import torch
import torch.distributed as dist
import torch.nn.functional as F


def _is_distributed_ready():
    # Check whether torch.distributed has already been initialized.
    return dist.is_available() and dist.is_initialized()


def _distributed_sync(tensor):
    # Gather the same shaped tensor from every worker.
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor, async_op=False)
    return torch.stack(gathered, dim=0)


def _resolve_device(device):
    # Normalize device inputs to a torch.device object.
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _pairwise_euclidean(x1, x2, pairwise=True):
    # Compute Euclidean distances, chunking when a full matrix would be large.
    if not pairwise:
        return torch.sqrt(torch.clamp((x1 - x2).pow(2).sum(dim=-1), min=0.0))
    split_size = min(4096, x1.size(0))
    chunks = []
    for chunk in x1.split(split_size, dim=0):
        chunks.append(torch.cdist(chunk, x2, p=2.0))
    return torch.cat(chunks, dim=0)


def _pairwise_cosine(x1, x2, pairwise=True):
    # Compute cosine distances, chunking when a full matrix would be large.
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    if not pairwise:
        return 1 - (x1 * x2).sum(dim=-1)
    split_size = min(4096, x1.size(0))
    chunks = []
    for chunk in x1.split(split_size, dim=0):
        chunks.append(1 - chunk.mm(x2.t()))
    return torch.cat(chunks, dim=0)


def _stable_cumsum(arr, dim=None, rtol=1e-5, atol=1e-8):
    # Build a numerically stable cumulative sum for k-means++ sampling.
    if dim is None:
        arr = arr.flatten()
        dim = 0
    out = torch.cumsum(arr, dim=dim, dtype=torch.float64)
    expected = torch.sum(arr, dim=dim, dtype=torch.float64)
    if not torch.all(torch.isclose(out[-1], expected, rtol=rtol, atol=atol, equal_nan=True)):
        return out
    return out


def _kmeans_plusplus(x, n_clusters, random_state, pairwise_distance, n_local_trials=None):
    # Initialize cluster centers with the k-means++ strategy.
    n_samples, n_features = x.size()
    generator = torch.Generator(device=x.device).manual_seed(random_state)
    centers = torch.empty((n_clusters, n_features), dtype=x.dtype, device=x.device)

    if n_local_trials is None:
        n_local_trials = 2 + int(torch.log(torch.tensor(float(n_clusters))).item())

    center_id = torch.randint(n_samples, (1,), generator=generator, device=x.device)
    indices = torch.full((n_clusters,), -1, dtype=torch.long, device=x.device)
    centers[0] = x[center_id]
    indices[0] = center_id

    closest_dist_sq = pairwise_distance(centers[0, None], x).view(-1).pow(2)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials, generator=generator, device=x.device) * current_pot
        candidate_ids = torch.searchsorted(_stable_cumsum(closest_dist_sq), rand_vals)
        candidate_ids.clamp_(max=closest_dist_sq.numel() - 1)

        distance_to_candidates = pairwise_distance(x[candidate_ids], x).pow(2)
        torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=-1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = x[best_candidate]
        indices[c] = best_candidate

    return centers, indices


class DistributedKMeans:
    @torch.no_grad()
    def __init__(
        self,
        metric='euclidean',
        init='k-means++',
        random_state=0,
        n_clusters=8,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        distributed=False,
        verbose=False,
        device=None,
    ):
        metric = metric.lower()
        if metric not in {'euclidean', 'cosine'}:
            raise ValueError('metric must be "euclidean" or "cosine"')

        self.metric = metric
        self.distance_metric = {
            'euclidean': _pairwise_euclidean,
            'cosine': _pairwise_cosine,
        }[metric]
        self.init = init
        self.random_state = 0 if random_state is None else random_state
        self.n_clusters = n_clusters
        self.n_init = n_init if not isinstance(init, torch.Tensor) else 1
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.device = _resolve_device(device)
        self.distributed = distributed and _is_distributed_ready()
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.stats = {'state': [], 'inertia': [], 'label': []}

    @torch.no_grad()
    def initialize(self, x, random_state):
        # Initialize the cluster centers for one restart.
        num_samples = x.size(0)
        if isinstance(self.init, str):
            generator = torch.Generator(device=x.device).manual_seed(random_state)
            if self.init == 'random':
                indices = torch.randperm(num_samples, generator=generator, device=x.device)[:self.n_clusters]
                return x[indices].clone()
            if self.init == 'k-means++':
                init_state, _ = _kmeans_plusplus(
                    x,
                    n_clusters=self.n_clusters,
                    random_state=random_state,
                    pairwise_distance=self.distance_metric,
                )
                return init_state
            raise NotImplementedError(f'Unknown init method: {self.init}')
        if isinstance(self.init, torch.Tensor):
            return self.init.to(device=x.device, dtype=x.dtype)
        raise NotImplementedError(f'Unsupported init type: {type(self.init)}')

    @torch.no_grad()
    def _predict(self, x, cluster_centers=None):
        # Assign each sample to its nearest cluster center.
        if cluster_centers is None:
            cluster_centers = self.cluster_centers_

        split_size = min(4096, x.size(0))
        all_labels = []
        inertia = 0.0

        for chunk in x.split(split_size, dim=0):
            dist_mat = self.distance_metric(chunk, cluster_centers)
            dists, labels = dist_mat.min(dim=1)
            inertia += dists.sum().item()
            all_labels.append(labels)

        return torch.cat(all_labels, dim=0), inertia

    @torch.no_grad()
    def fit_predict(self, x):
        # Run KMeans and return the cluster assignment for every sample.
        x = x.float().to(self.device)
        if self.metric == 'cosine':
            x = F.normalize(x, dim=-1)

        tol = torch.mean(torch.var(x, dim=0)).item() * self.tol

        min_inertia = float('inf')
        best_states = None
        best_labels = None

        random_states = torch.arange(self.n_init * self.world_size, device=x.device) + self.random_state
        random_states = random_states[self.rank::self.world_size]

        self.stats = {'state': [], 'inertia': [], 'label': []}

        for n_init_idx in range(self.n_init):
            random_state = int(random_states[n_init_idx].item())
            old_state = self.initialize(x, random_state=random_state)
            old_labels, inertia = self._predict(x, old_state)
            labels = old_labels.clone()

            for _ in range(self.max_iter):
                state = torch.zeros_like(old_state)
                counts = torch.zeros(self.n_clusters, dtype=x.dtype, device=x.device)
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=x.dtype))
                state.index_add_(0, labels, x)

                non_empty = counts > 0
                safe_counts = counts.clone()
                safe_counts[~non_empty] = 1.0
                state = state / safe_counts.view(-1, 1)
                state[~non_empty] = old_state[~non_empty]
                if self.metric == 'cosine':
                    state = F.normalize(state, dim=-1)

                labels, inertia = self._predict(x, state)

                if inertia < min_inertia:
                    min_inertia = inertia
                    best_states = state.clone()
                    best_labels = labels.clone()

                if torch.equal(labels, old_labels):
                    old_state = state
                    old_labels = labels
                    break

                center_shift = self.distance_metric(old_state, state, pairwise=False).sum().item()
                old_state = state
                old_labels = labels

                if center_shift <= tol:
                    break

            self.stats['state'].append(old_state.clone())
            self.stats['inertia'].append(inertia)
            self.stats['label'].append(old_labels.clone())

        self.stats['state'] = torch.stack(self.stats['state'])
        self.stats['inertia'] = torch.tensor(self.stats['inertia'], device=x.device)
        self.stats['label'] = torch.stack(self.stats['label'])

        if self.distributed:
            local_min = torch.tensor([min_inertia], dtype=torch.float32, device=x.device)
            gathered = _distributed_sync(local_min).view(-1)
            best_rank = int(torch.argmin(gathered).item())
            dist.broadcast(best_labels, src=best_rank)
            dist.broadcast(best_states, src=best_rank)
            self.stats['state'] = _distributed_sync(self.stats['state'])
            self.stats['inertia'] = _distributed_sync(self.stats['inertia'])
            self.stats['label'] = _distributed_sync(self.stats['label'])
            min_inertia = float(gathered[best_rank].item())

        self.cluster_centers_ = best_states
        self.labels_ = best_labels
        self.inertia_ = min_inertia
        return best_labels

    @torch.no_grad()
    def predict(self, x, soft=False):
        # Predict hard or soft assignments from fitted cluster centers.
        if self.cluster_centers_ is None:
            raise RuntimeError('Must call fit_predict before predict.')

        x = x.float().to(self.device)
        if self.metric == 'cosine':
            x = F.normalize(x, dim=-1)

        split_size = min(4096, x.size(0))
        outputs = []
        for chunk in x.split(split_size, dim=0):
            dists = self.distance_metric(chunk, self.cluster_centers_)
            if soft:
                outputs.append((-dists).softmax(dim=-1))
            else:
                outputs.append(dists.argmin(dim=-1))
        return torch.cat(outputs, dim=0)


@torch.no_grad()
def kmeans(
    X,
    num_clusters,
    distance='euclidean',
    tol=1e-4,
    device=torch.device('cuda'),
    init='k-means++',
    random_state=0,
    n_init=10,
    max_iter=300,
    distributed=False,
    verbose=False,
):
    # Backward-compatible function wrapper around DistributedKMeans.
    clustering_model = DistributedKMeans(
        metric=distance,
        init=init,
        random_state=random_state,
        n_clusters=num_clusters,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        distributed=distributed,
        verbose=verbose,
        device=device,
    )
    labels = clustering_model.fit_predict(X)
    return labels.cpu(), clustering_model.cluster_centers_.cpu()


@torch.no_grad()
def kmeans_predict(
    X,
    cluster_centers,
    distance='euclidean',
    device=torch.device('cuda'),
    soft=False,
):
    # Backward-compatible prediction wrapper around DistributedKMeans.
    clustering_model = DistributedKMeans(
        metric=distance,
        n_clusters=cluster_centers.size(0),
        n_init=1,
        max_iter=1,
        device=device,
    )
    clustering_model.cluster_centers_ = cluster_centers.float().to(clustering_model.device)
    if distance == 'cosine':
        clustering_model.cluster_centers_ = F.normalize(clustering_model.cluster_centers_, dim=-1)
    outputs = clustering_model.predict(X, soft=soft)
    return outputs.cpu()
