# Engineering Notes

## train.py

- `--dataset` now controls which dataset is used at runtime.
- `arxiv` and `ogbn-arxiv` are treated as the same dataset alias.
- `python train.py --dataset arxiv` will trigger local dataset preparation first.
- `arxiv` now uses a separate large-graph training branch inspired by PyAGC:
  sparse smoothing, mini-batch training, and batched inference for clustering.
- `arxiv` large-graph training now supports multi-GPU distributed execution via `torchrun`.
- In distributed mode, each rank trains on a shard of node batches and rank 0 gathers embeddings for evaluation/logging.

## utils.py

- `normalize_dataset_name()` maps `ogbn-arxiv` to `arxiv`.
- `prepare_arxiv_graph_data()` downloads `ogbn-arxiv` through OGB and converts it into the local HCGC dataset layout.
- The arxiv loader uses OGB's generic `NodePropPredDataset` path, so it does not depend on `torch_geometric`.
- If `dataset/arxiv.zip` exists, arxiv is prepared directly from that local zip file and no online download is needed.
- `load_graph_data("arxiv")` prepares the local files automatically on first use.
- `load_arxiv_splits()` reads the original OGB time split files from the local arxiv package.
- `sparse_preprocess_graph()` and `smooth_features_sparse()` avoid dense full-graph preprocessing for arxiv.
- `batch_contrastive_loss()` and `batch_inference_embeddings()` are added for large-graph arxiv training/inference.
- Large-graph Hamiltonian regularization now uses block-wise computation on each batch instead of random pair sampling.

## Local arxiv files

- `dataset/arxiv.zip`
- `dataset/arxiv/arxiv_feat.npy`
- `dataset/arxiv/arxiv_label.npy`
- `dataset/arxiv/arxiv_adj.npy`

## Storage detail

- Existing HCGC datasets use the `_feat.npy`, `_label.npy`, `_adj.npy` naming pattern.
- `ogbn-arxiv` is too large for a dense `N x N` adjacency array, so `arxiv_adj.npy` stores a SciPy CSR adjacency object inside the same `.npy` filename convention.

## Dependencies

- `ogb`

## kmeans_gpu.py

- The old single-function GPU KMeans has been replaced by a class-based implementation named `DistributedKMeans`.
- The new implementation keeps the old `kmeans(...)` and `kmeans_predict(...)` function entrypoints for backward compatibility.
- The new KMeans supports:
  - `k-means++` initialization
  - multiple random restarts via `n_init`
  - chunked distance computation to reduce peak memory
  - optional `torch.distributed` synchronization across workers
- Existing code in `utils.py` can keep calling `kmeans(X=feature, num_clusters=cluster_num, ...)` without changes.
- Distributed execution is enabled only when `torch.distributed` has already been initialized by the runtime.
