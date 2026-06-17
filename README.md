# HCAGC
The code for HCAGC: Hamiltonian-Inspired Contrastive Learning for Attribute Graph Clustering



**Before running the code, download and extract the corresponding dataset into the */dataset* directory**



When you need to run code on the [arxiv](http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip) dataset, execute the HCAGC (big map) code.

You can run the code using the following command:

```python
python train.py --dataset arxiv
```

or

```python
torchrun --nproc_per_node=num train.py --dataset arxiv
```

--num: the number of GPUs used during distributed training



If you want to run the code on datasets other than the [arxiv](http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip) dataset, you can use the following command in HCAGC:

```python
python train.py --dataset cora
```


