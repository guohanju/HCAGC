import os
import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import MLP
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--k', type=int, default=3, help='KNN')


args = parser.parse_args()


# for args.dataset in ["cora", "citeseer", "bat", "eat", "uat"]:
for args.dataset in ["cora"]:
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

    # load data
    lambda_f = 0.8
    X, y, A, node_num = load_graph_data(args.dataset, show_details=True)
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)
    adj_f = KNN(features, args.k)
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
        for seed in range(10):
            setup_seed(seed)
            best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
            print('Best Acc: {:.4f}, Best NMI: {:.4f}, Best ARI: {:.4f}, Best F1: {:.4f}'.format(best_acc, best_nmi, best_ari, best_f1))
            model = MLP([features.shape[1]] + args.dims)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            model = model.to(args.device)
            inx = sm_fea_s.to(args.device)
            adj_o = torch.FloatTensor(A).to(args.device)

            print('Start Training...')
            for epoch in tqdm(range(args.epochs)):
                model.train()
                z1, z2 = model(inx)
                Z = (z1 + z2) / 2
                if args.dataset == 'uat':
                    loss1 = hamiltonian_loss(Z, adj_o, 0.1)
                else:
                    loss1 = hamiltonian_loss(Z, adj_o, 1)
                loss2 = contrastive_loss(z1, z2, 0.5)
                loss = loss2 + 100 * loss1
                loss.backward()
                optimizer.step()
                if epoch % 50 == 0:
                    model.eval()
                    z1, z2 = model(inx)
                    hidden_emb = (z1 + z2) / 2

                    acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1

            tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
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
        print(round(acc_list.mean(), 2), round(acc_list.std(), 2))
        print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2))
        print(round(ari_list.mean(), 2), round(ari_list.std(), 2))
        print(round(f1_list.mean(), 2), round(f1_list.std(), 2))
        print('gnnlayers: {}, lr: {}, dims: {}, lamda: {}'.format(args.gnnlayers, args.lr, args.dims, i), file=file)
        print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
        print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
        print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
        print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
        file.close()
