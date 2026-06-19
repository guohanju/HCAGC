import os
import gzip
import zipfile
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.neighbors import NearestNeighbors

from kmeans_gpu import kmeans


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    # Build the original smoothing operators for the small-graph branch.
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    adj_ = adj + ident if renorm else adj

    rowsum = np.array(adj_.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv.dot(adj_).tocoo()
        laplacian = ident - adj_normalized
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    adjs = []
    for _ in range(layer):
        adjs.append(2 * ident - laplacian)
    return adjs


def sparse_preprocess_graph(adj, layer, norm='sym', renorm=True):
    # Build sparse smoothing operators for large-graph preprocessing.
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0], format='coo')
    adj_ = adj + ident if renorm else adj

    rowsum = np.array(adj_.sum(1)).flatten()
    rowsum = np.maximum(rowsum, 1e-12)
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_inv_sqrt).tocsr()
        laplacian = ident.tocsr() - adj_normalized
    elif norm == 'left':
        degree_mat_inv = sp.diags(np.power(rowsum, -1.0))
        adj_normalized = degree_mat_inv.dot(adj_).tocsr()
        laplacian = ident.tocsr() - adj_normalized
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    adjs = []
    ident_csr = ident.tocsr()
    for _ in range(layer):
        adjs.append((2 * ident_csr - laplacian).tocsr())
    return adjs


def cluster_acc(y_true, y_pred):
    # Align predicted clusters to labels and compute ACC/F1.
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i not in l2:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('Warning: number of classes mismatch')
        return 0.0, 0.0

    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    indexes = Munkres().compute(cost.__neg__().tolist())
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, show_details=True):
    # Aggregate the clustering metrics used during evaluation.
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def normalize_dataset_name(dataset_name):
    # Normalize dataset aliases used in the training script.
    dataset_name = dataset_name.lower()
    if dataset_name == 'ogbn-arxiv':
        return 'arxiv'
    if dataset_name in ['pokec-regions', 'pokec']:
        return 'pokec'
    return dataset_name


def _get_dataset_file_paths(dataset_name):
    # Resolve the standard local cache paths for one dataset.
    dataset_name = normalize_dataset_name(dataset_name)
    dataset_dir = os.path.join("dataset", dataset_name)
    load_path = os.path.join(dataset_dir, dataset_name)
    return dataset_name, dataset_dir, load_path + "_feat.npy", load_path + "_label.npy", load_path + "_adj.npy"


def _save_sparse_matrix_as_npy(path, matrix):
    # Store sparse adjacency in the existing .npy-based cache format.
    container = np.empty((), dtype=object)
    container[()] = matrix
    np.save(path, container, allow_pickle=True)


def _load_saved_adj_matrix(path):
    # Load either dense arrays or cached sparse adjacency.
    adj = np.load(path, allow_pickle=True)
    if isinstance(adj, np.ndarray) and adj.dtype == object:
        if adj.shape == ():
            return adj.item()
        if adj.size == 1:
            return adj.reshape(()).item()
    return adj


def _has_invalid_values(array):
    # Check whether a dense feature array still contains NaN or Inf.
    return np.isnan(array).any() or np.isinf(array).any()


def _ensure_arxiv_raw_from_local_zip(dataset_dir):
    # Extract the local arxiv package if raw files are not ready.
    zip_path = os.path.join("dataset", "arxiv.zip")
    raw_dir = os.path.join(dataset_dir, "raw")
    feat_raw_path = os.path.join(raw_dir, "node-feat.csv.gz")
    edge_raw_path = os.path.join(raw_dir, "edge.csv.gz")
    label_raw_path = os.path.join(raw_dir, "node-label.csv.gz")
    num_node_raw_path = os.path.join(raw_dir, "num-node-list.csv.gz")

    raw_files_ready = all(os.path.exists(path) for path in [
        feat_raw_path,
        edge_raw_path,
        label_raw_path,
        num_node_raw_path,
    ])
    if raw_files_ready:
        return raw_dir

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            "dataset/arxiv.zip was not found. Please place the local arxiv zip file under the dataset directory."
        )

    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall("dataset")

    if not os.path.exists(feat_raw_path):
        raise FileNotFoundError(
            "dataset/arxiv.zip was found but the extracted OGB raw files are missing."
        )
    return raw_dir


def _read_gzip_csv(path, dtype=None, header=None):
    # Read one gzip-compressed CSV file from the local dataset package.
    with gzip.open(path, 'rt') as handle:
        return pd.read_csv(handle, header=header, dtype=dtype)


def _load_local_arxiv_raw_graph(dataset_dir):
    # Parse raw arxiv features, labels and edges from the local package.
    raw_dir = _ensure_arxiv_raw_from_local_zip(dataset_dir)

    num_nodes = int(_read_gzip_csv(
        os.path.join(raw_dir, "num-node-list.csv.gz"),
        header=None,
    ).iloc[0, 0])

    feat = _read_gzip_csv(
        os.path.join(raw_dir, "node-feat.csv.gz"),
        header=None,
        dtype=np.float32,
    ).to_numpy(dtype=np.float32)

    label = _read_gzip_csv(
        os.path.join(raw_dir, "node-label.csv.gz"),
        header=None,
        dtype=np.int64,
    ).to_numpy(dtype=np.int64).reshape(-1)

    edge_df = _read_gzip_csv(
        os.path.join(raw_dir, "edge.csv.gz"),
        header=None,
        dtype=np.int64,
    )
    edge_index = edge_df.to_numpy(dtype=np.int64).T

    graph = {
        "node_feat": feat,
        "edge_index": edge_index,
        "num_nodes": num_nodes,
    }
    return graph, label


def _load_ogb_node_dataset_class():
    # Import the OGB node dataset loader with a fallback path.
    try:
        from ogb.nodeproppred import NodePropPredDataset
        return NodePropPredDataset
    except Exception:
        import importlib.util
        import ogb

        dataset_module_path = os.path.join(
            os.path.dirname(ogb.__file__),
            "nodeproppred",
            "dataset.py",
        )
        spec = importlib.util.spec_from_file_location(
            "ogb_nodeproppred_dataset_fallback",
            dataset_module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.NodePropPredDataset


def _parse_simple_yaml(path):
    # Parse the simple key/list structure used by GraphLand info.yaml files.
    data = {}
    current_key = None

    def _convert_scalar(value):
        if value == 'true':
            return True
        if value == 'false':
            return False
        return value

    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith('#'):
                continue

            if line.startswith('- '):
                if current_key is None:
                    raise ValueError(f"Invalid list item in yaml file: {path}")
                data[current_key].append(_convert_scalar(line[2:].strip()))
                continue

            if ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if value == '':
                data[key] = []
                current_key = key
            else:
                data[key] = _convert_scalar(value)
                current_key = None

    return data


def _impute_graphland_features(features_df, info):
    # Fill GraphLand numerical NaNs before converting features to numpy arrays.
    df = features_df.copy()
    numeric_columns = []
    numeric_columns.extend(info.get('numerical_features_names', []))
    numeric_columns.extend(info.get('fraction_features_names', []))
    numeric_columns = [col for col in dict.fromkeys(numeric_columns) if col in df.columns]

    for col in numeric_columns:
        series = pd.to_numeric(df[col], errors='coerce')
        series = series.replace([np.inf, -np.inf], np.nan)
        if series.isna().any():
            valid_values = series.dropna()
            fill_value = float(valid_values.median()) if not valid_values.empty else 0.0
            series = series.fillna(fill_value)
        df[col] = series.astype(np.float32)

    return df


def _load_graphland_local_graph(dataset_dir):
    # Parse one locally extracted GraphLand dataset directory.
    info_path = os.path.join(dataset_dir, "info.yaml")
    features_path = os.path.join(dataset_dir, "features.csv")
    targets_path = os.path.join(dataset_dir, "targets.csv")
    split_path = os.path.join(dataset_dir, "split_masks_TH.csv")
    edge_path = os.path.join(dataset_dir, "edgelist.csv")

    required_paths = [info_path, features_path, targets_path, split_path, edge_path]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(
            "The local GraphLand dataset directory is incomplete: {}".format(", ".join(missing_paths))
        )

    info = _parse_simple_yaml(info_path)

    features_df = pd.read_csv(features_path, index_col=0)
    features_df = _impute_graphland_features(features_df, info)
    targets_df = pd.read_csv(targets_path, index_col=0)
    targets = targets_df[info['target_name']].to_numpy()

    masks_df = pd.read_csv(split_path, index_col=0)
    masks = {k: np.array(v, dtype=bool) for k, v in masks_df.to_dict('list').items()}

    edges_df = pd.read_csv(edge_path)
    edge_index = edges_df.to_numpy(dtype=np.int64).T

    feat = features_df.to_numpy(dtype=np.float32, copy=True)
    feat[np.isinf(feat)] = 0.0

    graph = {
        "node_feat": feat,
        "edge_index": edge_index,
        "num_nodes": feat.shape[0],
        "masks": masks,
        "info": info,
    }
    return graph, targets.reshape(-1)


def prepare_arxiv_graph_data(force_reload=False):
    # Convert arxiv to the local HCGC feat/label/adj cache format.
    dataset_name, dataset_dir, feat_path, label_path, adj_path = _get_dataset_file_paths('arxiv')
    files_ready = all(os.path.exists(path) for path in [feat_path, label_path, adj_path])
    if files_ready and not force_reload:
        return feat_path, label_path, adj_path

    os.makedirs(dataset_dir, exist_ok=True)
    local_zip_path = os.path.join("dataset", "arxiv.zip")
    if os.path.exists(local_zip_path):
        graph, labels = _load_local_arxiv_raw_graph(dataset_dir)
    else:
        try:
            NodePropPredDataset = _load_ogb_node_dataset_class()
        except ImportError as exc:
            raise ImportError(
                "Preparing the arxiv dataset requires either dataset/arxiv.zip or the 'ogb' package."
            ) from exc
        dataset = NodePropPredDataset(root=os.path.join("dataset", "ogb"), name='ogbn-arxiv')
        graph, labels = dataset[0]

    feat = graph["node_feat"].astype(np.float32)
    label = labels.reshape(-1).astype(np.int64)
    edge_index = graph["edge_index"]
    row = edge_index[0]
    col = edge_index[1]
    sym_row = np.concatenate([row, col], axis=0)
    sym_col = np.concatenate([col, row], axis=0)
    sym_values = np.ones(sym_row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix(
        (sym_values, (sym_row, sym_col)),
        shape=(graph["num_nodes"], graph["num_nodes"]),
    ).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0

    np.save(feat_path, feat, allow_pickle=True)
    np.save(label_path, label, allow_pickle=True)
    _save_sparse_matrix_as_npy(adj_path, adj)
    return feat_path, label_path, adj_path


def prepare_pokec_graph_data(force_reload=False):
    # Convert a local GraphLand Pokec directory to the HCGC feat/label/adj cache format.
    dataset_name, dataset_dir, feat_path, label_path, adj_path = _get_dataset_file_paths('pokec')
    files_ready = all(os.path.exists(path) for path in [feat_path, label_path, adj_path])
    if files_ready and not force_reload:
        feat = np.load(feat_path, allow_pickle=True)
        if not _has_invalid_values(feat):
            return feat_path, label_path, adj_path
        force_reload = True
        reset_dataset_cache('pokec')

    graph, labels = _load_graphland_local_graph(dataset_dir)
    feat = graph["node_feat"].astype(np.float32)
    label = labels.astype(np.float32)

    labeled_mask = ~np.isnan(label)
    if labeled_mask.any():
        valid_labels = label[labeled_mask].astype(np.int64)
        unique_labels = np.unique(valid_labels)
        label_mapping = {label_value: idx for idx, label_value in enumerate(unique_labels)}
        mapped = np.full(label.shape[0], -1, dtype=np.int64)
        mapped[labeled_mask] = np.array([label_mapping[v] for v in valid_labels], dtype=np.int64)
        label = mapped
    else:
        label = np.full(label.shape[0], -1, dtype=np.int64)

    edge_index = graph["edge_index"]
    row = edge_index[0]
    col = edge_index[1]
    sym_row = np.concatenate([row, col], axis=0)
    sym_col = np.concatenate([col, row], axis=0)
    sym_values = np.ones(sym_row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix(
        (sym_values, (sym_row, sym_col)),
        shape=(graph["num_nodes"], graph["num_nodes"]),
    ).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0

    np.save(feat_path, feat, allow_pickle=True)
    np.save(label_path, label, allow_pickle=True)
    _save_sparse_matrix_as_npy(adj_path, adj)
    reset_dataset_smoothed_cache('pokec')
    return feat_path, label_path, adj_path


def load_graph_data(dataset_name, show_details=False):
    # Load one dataset from the local cache, preparing arxiv on demand.
    dataset_name, _, feat_path, label_path, adj_path = _get_dataset_file_paths(dataset_name)
    if dataset_name == 'arxiv':
        prepare_arxiv_graph_data()
    elif dataset_name == 'pokec':
        prepare_pokec_graph_data()

    feat = np.load(feat_path, allow_pickle=True)
    label = np.load(label_path, allow_pickle=True)
    adj = _load_saved_adj_matrix(adj_path)
    node_num = feat.shape[0]
    if show_details:
        edge_num = int(adj.nnz / 2) if sp.issparse(adj) else int(np.nonzero(adj)[0].shape[0] / 2)
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", edge_num)
        valid_label = label[label >= 0]
        print("category num:          ", max(valid_label) - min(valid_label) + 1 if valid_label.size > 0 else 0)
        print("category distribution: ")
        for i in range(int(valid_label.max()) + 1 if valid_label.size > 0 else 0):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj, node_num


def reset_dataset_cache(dataset_name):
    # Remove cached feat/label/adj files so the dataset can be rebuilt.
    _, dataset_dir, feat_path, label_path, adj_path = _get_dataset_file_paths(dataset_name)
    for path in [feat_path, label_path, adj_path]:
        if os.path.exists(path):
            os.remove(path)
    reset_dataset_smoothed_cache(dataset_name)


def reset_dataset_smoothed_cache(dataset_name):
    # Remove cached smoothed features so they can be regenerated from fresh inputs.
    _, dataset_dir, _, _, _ = _get_dataset_file_paths(dataset_name)
    sm_pattern = os.path.join(dataset_dir, f"{dataset_name}_feat_sm_*.npy")
    import glob
    for path in glob.glob(sm_pattern):
        os.remove(path)


def setup_seed(seed):
    # Make training and clustering runs reproducible.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clustering(feature, true_labels, cluster_num):
    # Run KMeans and immediately evaluate the predicted clustering.
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    return round(100 * acc, 2), round(100 * nmi, 2), round(100 * ari, 2), round(100 * f1, 2), predict_labels


def hamiltonian_loss(Z, adj, gamma):
    # Original full-batch Hamiltonian loss used by small graphs.
    sim_matrix = torch.mm(Z, Z.t())
    A = adj
    J = 1 - A
    mask = torch.ones_like(A) - torch.eye(A.size(0), device=A.device)
    H_matrix = (A - gamma * J) * sim_matrix * mask
    return -0.5 * torch.mean(H_matrix)


def contrastive_loss(z1, z2, temperature=0.5):
    # Original full-batch contrastive loss used by small graphs.
    sim_pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    sim_neg = torch.exp(torch.mm(z1, z2.t()) / temperature)
    return -torch.log(sim_pos / (torch.sum(sim_neg, dim=-1) - sim_pos)).mean()


def batch_contrastive_loss(z1, z2, batch_size=4096, temperature=0.5):
    # Chunked contrastive loss used by the large-graph branch.
    total_loss = 0.0
    total_count = 0
    z2_t = z2.t().contiguous()
    for start in range(0, z1.size(0), batch_size):
        end = min(start + batch_size, z1.size(0))
        z1_chunk = z1[start:end]
        z2_chunk = z2[start:end]
        sim_pos = torch.exp(torch.sum(z1_chunk * z2_chunk, dim=-1) / temperature)
        sim_neg = torch.exp(torch.mm(z1_chunk, z2_t) / temperature)
        chunk_loss = -torch.log(sim_pos / (torch.sum(sim_neg, dim=-1) - sim_pos + 1e-12))
        total_loss += chunk_loss.sum()
        total_count += chunk_loss.numel()
    return total_loss / max(total_count, 1)


@torch.no_grad()
def batch_inference_embeddings(model, features, batch_size=8192, device='cuda'):
    # Generate embeddings in batches to keep inference memory bounded.
    outputs_1 = []
    outputs_2 = []
    device = torch.device(device)
    model.eval()
    for start in range(0, features.size(0), batch_size):
        end = min(start + batch_size, features.size(0))
        batch_x = features[start:end].to(device)
        z1, z2 = model(batch_x)
        outputs_1.append(z1.cpu())
        outputs_2.append(z2.cpu())
    return torch.cat(outputs_1, dim=0), torch.cat(outputs_2, dim=0)


def smooth_features_sparse(features, adj_ops, cache_path=None):
    # Cache sparse-smoothed features for the large-graph branch.
    if cache_path is not None and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if not _has_invalid_values(cached):
            return cached
    smoothed = np.asarray(features, dtype=np.float32)
    for op in adj_ops:
        smoothed = op.dot(smoothed)
    smoothed = np.asarray(smoothed, dtype=np.float32)
    smoothed[np.isinf(smoothed)] = np.nan
    if cache_path is not None:
        np.save(cache_path, smoothed, allow_pickle=True)
    return smoothed


def build_knn_graph(x, k):
    # Build the auxiliary KNN graph used by the small-graph branch.
    x1 = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x1)
    _, indices = nbrs.kneighbors(x1)
    rows = np.repeat(np.arange(x1.shape[0]), k)
    cols = indices.flatten()
    values = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((values, (rows, cols)), shape=(x1.shape[0], x1.shape[0]))
    return ((adj + adj.T) * 0.5).tocoo()
