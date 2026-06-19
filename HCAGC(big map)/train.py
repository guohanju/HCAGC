import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *
from tqdm import tqdm
from torch import optim
from model import MLP
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='pokec', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--k', type=int, default=3, help='KNN')
parser.add_argument('--batch_size', type=int, default=4096, help='mini-batch size for large graphs')
parser.add_argument('--eval_batch_size', type=int, default=8192, help='inference batch size for large graphs')
parser.add_argument('--lambda_h', type=float, default=1000, help='weight of sampled hamiltonian loss for large graphs')
parser.add_argument('--hamiltonian_gamma', type=float, default=1.0, help='gamma in sampled hamiltonian loss')
parser.add_argument('--hamiltonian_chunk_size', type=int, default=1024, help='chunk size for block hamiltonian loss on large graphs')


args = parser.parse_args()
args.dataset = normalize_dataset_name(args.dataset)


def init_distributed_mode():
    # Initialize DDP from torchrun environment variables when available.
    if not dist.is_available():
        return False, 0, 1, 0

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def split_train_indices(num_nodes, rank, world_size, seed=None):
    # Split the node permutation across workers for distributed training.
    perm = torch.randperm(num_nodes)
    usable = (num_nodes // world_size) * world_size
    if usable == 0:
        return perm
    perm = perm[:usable]
    per_rank = usable // world_size
    start = rank * per_rank
    end = start + per_rank
    return perm[start:end]


def split_eval_indices(num_nodes, rank, world_size):
    # Split node ids across workers for distributed inference.
    indices = torch.arange(num_nodes)
    return torch.tensor_split(indices, world_size)[rank]


@torch.no_grad()
def distributed_collect_embeddings(model, features, device, rank, world_size, num_nodes, batch_size=8192):
    # Compute local embeddings and gather the full embedding matrix on rank 0.
    local_indices = split_eval_indices(num_nodes, rank, world_size)
    local_feats = features[local_indices]
    z1, z2 = batch_inference_embeddings(model, local_feats, batch_size=batch_size, device=device)
    local_hidden = ((z1 + z2) / 2).cpu()

    if world_size == 1:
        return local_hidden

    payload = (local_indices.cpu(), local_hidden)
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)

    if rank != 0:
        return None

    hidden_dim = local_hidden.size(1)
    full_hidden = torch.empty((num_nodes, hidden_dim), dtype=local_hidden.dtype)
    for indices, emb in gathered:
        full_hidden[indices.long()] = emb
    return full_hidden


def block_hamiltonian_loss(z, adj, gamma=1.0, chunk_size=1024):
    # Compute a chunked Hamiltonian loss on one batch-local adjacency block.
    if z.size(0) == 0:
        return z.new_tensor(0.0)

    adj_local = adj[:z.size(0), :z.size(0)].toarray()
    adj_local = torch.from_numpy(adj_local).to(z.device, dtype=z.dtype)
    total_loss = z.new_tensor(0.0)
    total_count = 0

    for start_i in range(0, z.size(0), chunk_size):
        end_i = min(start_i + chunk_size, z.size(0))
        zi = z[start_i:end_i]
        ai = adj_local[start_i:end_i]
        for start_j in range(0, z.size(0), chunk_size):
            end_j = min(start_j + chunk_size, z.size(0))
            zj = z[start_j:end_j]
            aj = ai[:, start_j:end_j]
            sim = torch.mm(zi, zj.t())
            block_loss = (aj - gamma * (1 - aj)) * sim
            total_loss = total_loss + block_loss.sum()
            total_count += block_loss.numel()

    return -0.5 * total_loss / max(total_count, 1)


def should_eval_and_log_epoch(epoch, total_epochs, interval=50):
    # Only evaluate on checkpoints to reduce large-graph overhead.
    return (epoch + 1) % interval == 0 or epoch == total_epochs - 1


def should_update_best(acc, nmi, ari, f1, best_acc, best_nmi, best_ari, best_f1, eps=1e-12):
    # Update the best checkpoint by comparing metrics in priority order.
    if acc > best_acc + eps:
        return True
    if abs(acc - best_acc) <= eps and nmi > best_nmi + eps:
        return True
    if abs(acc - best_acc) <= eps and abs(nmi - best_nmi) <= eps and ari > best_ari + eps:
        return True
    if abs(acc - best_acc) <= eps and abs(nmi - best_nmi) <= eps and abs(ari - best_ari) <= eps and f1 > best_f1 + eps:
        return True
    return False


distributed, rank, world_size, local_rank = init_distributed_mode()
if distributed:
    args.device = f'cuda:{local_rank}'


# for args.dataset in ["cora", "citeseer", "bat", "eat", "uat"]:
for args.dataset in [args.dataset]:
    if rank == 0:
        print("Using {} dataset".format(args.dataset))
        file = open("result_baseline.csv", "a+")
        print(args.dataset, file=file)
        file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 4
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 4
        args.lr = 5e-5
        args.dims = [500]
    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 5
        args.lr = 1e-5
        args.dims = [500]
    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 4
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'uat':
        args.cluster_num = 4
        args.gnnlayers = 5
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'corafull':
        args.cluster_num = 70
        args.gnnlayers = 2
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'arxiv':
        args.cluster_num = 40
        args.gnnlayers = 6
        args.lr = 1e-3
        args.dims = [500]
    elif args.dataset == 'pokec':
        args.cluster_num = 183
        args.gnnlayers = 6
        args.lr = 1e-3
        args.dims = [500]

    # load data
    lambda_f = 0.8
    X, y, A, node_num = load_graph_data(args.dataset, show_details=True)
    if args.dataset in ['arxiv', 'pokec']:
        if rank == 0:
            print("Using large-graph training branch for {}...".format(args.dataset))
        features = X
        true_labels = y
        adj = sp.csr_matrix(A)
        if rank == 0:
            print('Sparse Laplacian Smoothing...')
        adj_norm_s = sparse_preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
        path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
        sm_fea_s = smooth_features_sparse(features, adj_norm_s, cache_path=path)
        sm_fea_s = torch.FloatTensor(sm_fea_s)

        lambda1 = [1000]
        for i in lambda1:
            acc_list = []
            nmi_list = []
            ari_list = []
            f1_list = []
            for seed in range(10):
                setup_seed(seed)
                if rank == 0:
                    initial_acc, initial_nmi, initial_ari, initial_f1, prediect_labels = clustering(
                        sm_fea_s, true_labels, args.cluster_num
                    )
                    print('Initial Acc: {:.4f}, Initial NMI: {:.4f}, Initial ARI: {:.4f}, Initial F1: {:.4f}'.format(
                        initial_acc, initial_nmi, initial_ari, initial_f1
                    ))
                best_acc, best_nmi, best_ari, best_f1 = -1.0, -1.0, -1.0, -1.0
                best_epoch = -1
                best_loss = None
                model = MLP([features.shape[1]] + args.dims)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                model = model.to(args.device)
                if distributed:
                    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                train_model = model.module if distributed else model
                if rank == 0:
                    print('Start Large-Graph Training...')
                epoch_iter = tqdm(range(args.epochs), disable=rank != 0)
                for epoch in epoch_iter:
                    model.train()
                    epoch_loss = 0.0
                    batch_count = 0
                    local_perm = split_train_indices(sm_fea_s.size(0), rank, world_size, seed=seed)
                    for start in range(0, local_perm.size(0), args.batch_size):
                        end = min(start + args.batch_size, local_perm.size(0))
                        batch_idx = local_perm[start:end]
                        inx = sm_fea_s[batch_idx].to(args.device)
                        batch_adj = adj[batch_idx.cpu().numpy()][:, batch_idx.cpu().numpy()].tocsr()
                        optimizer.zero_grad()
                        z1, z2 = model(inx)
                        loss2 = batch_contrastive_loss(z1, z2, batch_size=min(args.batch_size, 2048), temperature=0.5)
                        z = F.normalize((z1 + z2) / 2, dim=1)
                        loss1 = block_hamiltonian_loss(
                            z,
                            batch_adj,
                            gamma=args.hamiltonian_gamma,
                            chunk_size=args.hamiltonian_chunk_size,
                        )
                        loss = loss2 + args.lambda_h * loss1
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        batch_count += 1

                    local_epoch_loss = torch.tensor(
                        [epoch_loss, float(batch_count)],
                        dtype=torch.float32,
                        device=args.device,
                    )
                    if distributed:
                        dist.all_reduce(local_epoch_loss, op=dist.ReduceOp.SUM)
                    epoch_avg_loss = local_epoch_loss[0].item() / max(local_epoch_loss[1].item(), 1.0)

                    if should_eval_and_log_epoch(epoch, args.epochs):
                        model.eval()
                        hidden_emb = distributed_collect_embeddings(
                            train_model,
                            sm_fea_s,
                            args.device,
                            rank,
                            world_size,
                            sm_fea_s.size(0),
                            batch_size=args.eval_batch_size,
                        )

                        if rank == 0:
                            acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                            if should_update_best(acc, nmi, ari, f1, best_acc, best_nmi, best_ari, best_f1):
                                best_acc = acc
                                best_nmi = nmi
                                best_ari = ari
                                best_f1 = f1
                                best_epoch = epoch
                                best_loss = epoch_avg_loss
                            tqdm.write(
                                'epoch: {}, acc: {}, nmi: {}, ari: {}, f1: {}'.format(
                                    epoch + 1,
                                    best_acc,
                                    best_nmi,
                                    best_ari,
                                    best_f1,
                                )
                            )

                        if distributed:
                            best_tensor = torch.tensor(
                                [best_acc, best_nmi, best_ari, best_f1, float(best_epoch), best_loss if best_loss is not None else float('nan')],
                                dtype=torch.float64,
                                device=args.device,
                            )
                            dist.broadcast(best_tensor, src=0)
                            if rank != 0:
                                best_acc = float(best_tensor[0].item())
                                best_nmi = float(best_tensor[1].item())
                                best_ari = float(best_tensor[2].item())
                                best_f1 = float(best_tensor[3].item())
                                best_epoch = int(best_tensor[4].item())
                                best_loss = float(best_tensor[5].item())

                if rank == 0:
                    tqdm.write('seed {} -> acc: {}, nmi: {}, ari: {}, f1: {}'.format(
                        seed, best_acc, best_nmi, best_ari, best_f1
                    ))
                    file = open("result_baseline.csv", "a+")
                    print(best_acc, best_nmi, best_ari, best_f1, file=file)
                    file.close()
                    acc_list.append(best_acc)
                    nmi_list.append(best_nmi)
                    ari_list.append(best_ari)
                    f1_list.append(best_f1)

            if rank == 0:
                acc_list = np.array(acc_list)
                nmi_list = np.array(nmi_list)
                ari_list = np.array(ari_list)
                f1_list = np.array(f1_list)
                file = open("result_baseline.csv", "a+")
                print('acc mean/std:', round(acc_list.mean(), 2), round(acc_list.std(), 2))
                print('nmi mean/std:', round(nmi_list.mean(), 2), round(nmi_list.std(), 2))
                print('ari mean/std:', round(ari_list.mean(), 2), round(ari_list.std(), 2))
                print('f1 mean/std:', round(f1_list.mean(), 2), round(f1_list.std(), 2))
                print('gnnlayers: {}, lr: {}, dims: {}, lamda: {}'.format(args.gnnlayers, args.lr, args.dims, i), file=file)
                print('acc mean/std:', round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
                print('nmi mean/std:', round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
                print('ari mean/std:', round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
                print('f1 mean/std:', round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
                file.close()
        continue
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)
    adj_f = build_knn_graph(features, args.k)
    adj = lambda_f * adj + (1-lambda_f) * adj_f
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros() 
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()

    path = "dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    # if os.path.exists(path):
    #     sm_fea_s = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
    # else:
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    np.save(path, sm_fea_s, allow_pickle=True)

    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

    lambda1 = [1000]
    for i in lambda1:
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []
        for seed in range(5):
            setup_seed(seed)
            initial_acc, initial_nmi, initial_ari, initial_f1, prediect_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
            print('Initial Acc: {:.4f}, Initial NMI: {:.4f}, Initial ARI: {:.4f}, Initial F1: {:.4f}'.format(initial_acc, initial_nmi, initial_ari, initial_f1))
            best_acc, best_nmi, best_ari, best_f1 = -1.0, -1.0, -1.0, -1.0
            best_epoch = -1
            best_loss = None
            model = MLP([features.shape[1]] + args.dims)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            model = model.to(args.device)
            inx = sm_fea_s.to(args.device)
            adj_o = None if sp.issparse(A) else torch.FloatTensor(A).to(args.device)

            print('Start Training...')
            for epoch in tqdm(range(args.epochs)):
                model.train()
                optimizer.zero_grad()
                z1, z2 = model(inx)
                Z = (z1 + z2) / 2
                if adj_o is not None:
                    if args.dataset == 'uat':
                        loss1 = hamiltonian_loss(Z, adj_o, 0.1)
                    else:
                        loss1 = hamiltonian_loss(Z, adj_o, 1)
                loss2 = contrastive_loss(z1, z2, 0.5)
                loss = loss2 + args.lambda_h * loss1
                loss.backward()
                optimizer.step()
                epoch_avg_loss = loss.item()
                if should_eval_and_log_epoch(epoch, args.epochs):
                    model.eval()
                    z1, z2 = model(inx)
                    hidden_emb = (z1 + z2) / 2

                    acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                    if should_update_best(acc, nmi, ari, f1, best_acc, best_nmi, best_ari, best_f1):
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
                        best_epoch = epoch
                        best_loss = epoch_avg_loss
                    tqdm.write(
                        'epoch: {}, acc: {}, nmi: {}, ari: {}, f1: {}'.format(
                            epoch + 1,
                            best_acc,
                            best_nmi,
                            best_ari,
                            best_f1,
                        )
                    )

            tqdm.write('seed {} -> acc: {}, nmi: {}, ari: {}, f1: {}'.format(
                seed, best_acc, best_nmi, best_ari, best_f1
            ))
            file = open("result_baseline.csv", "a+")
            print(best_acc, best_nmi, best_ari, best_f1, file=file)
            file.close()
            acc_list.append(best_acc)
            nmi_list.append(best_nmi)
            ari_list.append(best_ari)
            f1_list.append(best_f1)

        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)
        file = open("result_baseline.csv", "a+")
        print('acc mean/std:', round(acc_list.mean(), 2), round(acc_list.std(), 2))
        print('nmi mean/std:', round(nmi_list.mean(), 2), round(nmi_list.std(), 2))
        print('ari mean/std:', round(ari_list.mean(), 2), round(ari_list.std(), 2))
        print('f1 mean/std:', round(f1_list.mean(), 2), round(f1_list.std(), 2))
        print('gnnlayers: {}, lr: {}, dims: {}, lamda: {}'.format(args.gnnlayers, args.lr, args.dims, i), file=file)
        print('acc mean/std:', round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
        print('nmi mean/std:', round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
        print('ari mean/std:', round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
        print('f1 mean/std:', round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
        file.close()

if distributed and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
