
import torch
import torch.nn as nn
import numpy as np
import os
import json
import shutil
import warnings
from collections import OrderedDict
from torch_geometric.utils import k_hop_subgraph

# Uvoz tvojih modula

from files.model import make_model
from files.utils import load_graph_data, scaled_Laplacian, get_chebyshev_polynomials, load_traffic_data
import files.config as config
# Uvoz tvoje logike za particioniranje
from files.splitDataKralj import partition_nodes_to_cloudlets_by_range_proximity, get_cloudlet_location_info_from_json
from scipy.sparse import csr_matrix, save_npz
import pandas as pd

# --- 1. GLOBALNA INICIJALIZACIJA (Kao u tvom config/server dijelu) ---

def run_data_splitting():
    """
    Ova funkcija replicira logiku iz tvog originalnog fl_server.py.
    Briše stare mape, računa particije i sprema podatke za klijente.
    """
    if os.path.exists(os.path.join("data", "client_0", "vel.csv")):
        print("[Task] Podaci za klijente već postoje. Preskačem splitanje.")
        return

    print("[Task] Pokrećem particioniranje podataka (splitDataKralj logic)...")
    
    # Učitaj JSON
    with open(config.CLOUDLET_LOC_JSON) as f:
        cloudlet_info_json = json.load(f)
    
    cloudlets, radius_km = get_cloudlet_location_info_from_json(config.CLOUDLET_EXP, cloudlet_info_json)
    
    # Izračunaj particije
    client_nodes_list = partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_name="")
    
    # Učitaj originalne podatke za kopiranje
    vel_df = pd.read_csv(config.DATA_PATH, header=None)
    vel_data = vel_df.values
    adj_original = load_graph_data(config.ADJ_PATH)

    # Priprema mapa
    data_dir = "data"
    for name in os.listdir(data_dir):
        if name.startswith("client_"):
            shutil.rmtree(os.path.join(data_dir, name))
            
    # Spremanje
    for cid, nodes in enumerate(client_nodes_list):
        client_dir = os.path.join(data_dir, f"client_{cid}")
        os.makedirs(client_dir, exist_ok=True)
        
        # 1. Spremi nodes.npy (Tvoj popis senzora)
        np.save(os.path.join(client_dir, "nodes.npy"), np.array(nodes))
        
        # 2. Spremi maskirani vel.csv (Ostalo su nule)
        client_vel = vel_data.copy()
        mask = np.ones(client_vel.shape[1], dtype=bool)
        mask[nodes] = False
        client_vel[:, mask] = 0.0
        pd.DataFrame(client_vel).to_csv(os.path.join(client_dir, "vel.csv"), index=False, header=False)
        
        # 3. Spremi adj.npz
        save_npz(os.path.join(client_dir, "adj.npz"), csr_matrix(adj_original))
        
    print(f"[Task] Podaci pripremljeni za {len(client_nodes_list)} klijenata.")


# Globalno učitavanje grafa za model
try:
    adj_mx = load_graph_data(config.ADJ_PATH)
    L = scaled_Laplacian(adj_mx)
    cheb_poly = get_chebyshev_polynomials(L, config.K)
    # Edge index za k-hop subgraph (trebat će nam kasnije)
    edge_index_global = torch.tensor(adj_mx.nonzero(), dtype=torch.long)
except Exception as e:
    warnings.warn(f"Greška pri globalnom učitavanju grafa: {e}")
    adj_mx, cheb_poly, edge_index_global = None, None, None


# --- 2. DEFINICIJA MODELA (Net) ---

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.device = config.DEVICE
        
        # Priprema polinoma
        if cheb_poly is not None:
            self.cheb_tensors = [torch.FloatTensor(c).to(self.device) for c in cheb_poly]
        else:
            self.cheb_tensors = []

        # Tvoja make_model funkcija
        self.astgcn = make_model(
            self.device,
            config.NB_BLOCK,
            config.IN_CHANNELS,
            config.K,
            config.NB_CHEV_FILTER,
            config.NB_TIME_FILTER,
            config.TIME_STRIDES,
            self.cheb_tensors,
            config.NUM_FOR_PREDICT,
            config.LEN_INPUT,
            config.NUM_NODES
        )

    def forward(self, x):
        return self.astgcn(x)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# --- 3. UČITAVANJE PODATAKA (Logika iz tvog fl_client.py) ---

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """
    Ova funkcija radi isto što i __init__ u tvom ASTGCNClientu:
    1. Učitava 'nodes.npy'
    2. Računa k_hop_subgraph
    3. Vraća loader s podacima
    """
    run_data_splitting()
    print(f"--> [Task] Loading data for partition {partition_id}")
    
    client_dir = os.path.join("data", f"client_{partition_id}")
    vel_path = os.path.join(client_dir, "vel.csv")
    nodes_path = os.path.join(client_dir, "nodes.npy")
    
    # 1. Učitaj lokalne čvorove
    cln_nodes_numpy = np.load(nodes_path)
    cln_nodes = torch.tensor(cln_nodes_numpy, dtype=torch.long)

    # 2. Izračunaj k-hop subgraph (Ghost nodes)
    # Ovo je direktno iz tvog fl_client.py
    H = config.NB_BLOCK * (config.K - 1)
    
    sub_nodes, sub_edge_index, node_map, _ = k_hop_subgraph(
        node_idx=cln_nodes,
        num_hops=H,
        edge_index=edge_index_global, # Koristimo globalni edge_index učitan na vrhu
        relabel_nodes=True,
        num_nodes=config.NUM_NODES
    )
    
    # 3. Učitaj podatke (već maskirane u split fazi)
    # Koristimo tvoj load_traffic_data
    # Stavljamo client_id=0 i num_clients=1 jer učitavamo specifičan fajl
    trainloader, testloader, _, _ = load_traffic_data(
        vel_csv_path=vel_path,
        batch_size=batch_size,
        seq_len=config.LEN_INPUT,
        pred_len=config.NUM_FOR_PREDICT,
        train_ratio=0.8
    )
    
    # 4. TRICK: Spremamo maske u dataset da ih 'train' funkcija može vidjeti
    # Orchestrator ne dopušta dodatne argumente u train funkciji, pa ih "švercamo" ovdje
    trainloader.dataset.cln_nodes = cln_nodes
    trainloader.dataset.node_map = node_map
    
    # Isto za test
    testloader.dataset.cln_nodes = cln_nodes
    testloader.dataset.node_map = node_map
    
    return trainloader, testloader


# --- 4. TRENIRANJE (Logika iz tvog ASTGCNClient.fit) ---

def train(net, trainloader, valloader, epochs, learning_rate, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Dohvati maske iz dataseta
    try:
        cln_nodes = trainloader.dataset.cln_nodes.to(device)
        node_map = trainloader.dataset.node_map.to(device)
    except AttributeError:
        print("[Warning] Maske nisu nađene, koristim full graph loss.")
        cln_nodes = None

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            out = net(x).transpose(1, 2) # Fix dimenzija
            
            # MASKIRANJE (Tvoja logika)
            if cln_nodes is not None:
                # 1. Izvuci izlaz za 'ghost' podgraf (koristeći node_map)
                # Napomena: Tvoj k_hop vraća node_map koji mapira tvoje čvorove u podgraf.
                # Ali, budući da tvoj model ovdje radi na GLOBALNOM grafu (228 čvorova),
                # izlaz 'out' ima dimenziju 228.
                # Zato možemo direktno koristiti 'cln_nodes' (originalne indekse).
                
                out_local = out[:, cln_nodes, :]
                y_local = y[:, cln_nodes, :]
                
                loss = criterion(out_local, y_local)
            else:
                loss = criterion(out, y)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

    # Validacija
    val_loss, val_mae = test(net, valloader, device)
    return {"val_loss": val_loss, "val_mae": val_mae}


# --- 5. TESTIRANJE (Logika iz tvog ASTGCNClient.evaluate) ---

def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    
    try:
        cln_nodes = testloader.dataset.cln_nodes.to(device)
    except AttributeError:
        cln_nodes = None

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)

            out = net(x).transpose(1, 2)
            
            if cln_nodes is not None:
                out_local = out[:, cln_nodes, :]
                y_local = y[:, cln_nodes, :]
                
                loss = criterion(out_local, y_local)
                mae = torch.mean(torch.abs(out_local - y_local))
            else:
                loss = criterion(out, y)
                mae = torch.mean(torch.abs(out - y))

            total_loss += loss.item()
            total_mae += mae.item()

    return total_loss / len(testloader), total_mae / len(testloader)