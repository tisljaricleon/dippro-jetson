import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

def get_chebyshev_polynomials(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

def scaled_Laplacian(W):
    if isinstance(W, np.matrix):
        W = np.array(W)
         
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def load_graph_data(adj_path):
    print(f"Učitavam graf: {adj_path}")
    
    raw = np.load(adj_path)
    mat_shape = tuple(raw['shape'])  
    sparse_mx = sp.csr_matrix((raw['data'], raw['indices'], raw['indptr']), shape=mat_shape)
    adj_mx = sparse_mx.toarray()
    
    print(f"Uspješno učitano. Dimenzije: {adj_mx.shape}")
    
    return adj_mx
    
def load_traffic_data(
    vel_csv_path,
    batch_size,
    seq_len=12,
    pred_len=12,
    train_ratio=0.8,
    client_id=None,
    num_clients=None
):
    df = pd.read_csv(vel_csv_path, header=None)
    data = df.values

    if client_id is not None and num_clients is not None:
        chunk = len(data) // num_clients
        start = client_id * chunk
        end = (client_id + 1) * chunk
        data = data[start:end]

    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / std

    X, Y = [], []
    for i in range(len(data_norm) - seq_len - pred_len):
        X.append(np.expand_dims(data_norm[i:i+seq_len], axis=0))
        Y.append(data_norm[i+seq_len:i+seq_len+pred_len])

    X = np.array(X).transpose(0, 3, 1, 2)
    Y = np.array(Y).transpose(0, 2, 1)

    split = int(len(X) * train_ratio)
    train_x, test_x = X[:split], X[split:]
    train_y, test_y = Y[:split], Y[split:]

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_x), torch.FloatTensor(test_y)
    )

    return (
        torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False),
        mean,
        std
    )