import torch

DATA_PATH = "data/vel.csv"                 # csv putanja do podataka
ADJ_PATH = "data/adj.npz"                  # putanja do matrice susjedstva
LOCATIONS_PATH = "data/locations-raw.csv"  # putanja do lokacija senzora
CLOUDLET_LOC_JSON = "data/cloudlets.json"  # putanja do json datoteke s podacima o cloudlet lokacijama
CLOUDLET_EXP = "experiment_2"              # ime eksperimenta za dohvat cloudlet podataka iz json datoteke

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

LEARNING_RATE = 0.001 

NUM_NODES = 228       # Broj čvorova u grafu
LEN_INPUT = 12        # Duljina ulaznog vremenskog prozora, 12 * 5min = 1 sat
NUM_FOR_PREDICT = 12  # Duljina izlaznog vremenskog prozora za predviđanje, 12 * 5min = 1 sat
IN_CHANNELS = 1       # Broj ulaza po čvoru
NUM_CLIENTS = 2       # Broj klijenata za federativno učenje

BATCH_SIZE = 32        
EPOCHS = 1          
NB_BLOCK = 2          # Broj ASTGCN blokova
K = 1                 # Red Chebyshev polinoma 
NB_CHEV_FILTER = 16   # Broj filtera u graf konvoluciji 
NB_TIME_FILTER = 16   # Broj filtera u vremenskoj konvoluciji 
TIME_STRIDES = 1      # Stride u vremenskoj konvoluciji