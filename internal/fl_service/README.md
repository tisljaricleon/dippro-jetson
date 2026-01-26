# FL service - Flower Wrapper with support for HFL 

This is a wrapper for Flower framework to support hierarchical federated learning (HFL). 
It serves as a basis for the [FL orchestrator](https://github.com/AIoTwin/fl-orchestrator) also developed by AIoTwin.

## Deployment example

This example deploys a HFL pipeline with one global server, two local servers and four clients (two per cluster).
It trains a CNN over the CIFAR-10 dataset, as defined in `client/task.py`.

### Run Global Server

```
cd global_server
```

Build container image:
```
docker build -t hfl-global-server:1.0 . 
```

Run the server:
```
docker run \
    -p 8080:8080
    -v $(pwd)/task.py:/home/task.py \
    -v $(pwd)/global_server_config.yaml:/home/global_server_config.yaml \
    hfl-global-server:1.0 
```

### Run Local Servers: 

```
cd local_server
```

First, change IP address in `local_server_config.yaml` to match IP address of your host.

Build container image:
```
docker build -t hfl-local-server:1.0 .  
```

Run two local servers:
```
docker run \
    -p 8081:8080 \
    -v $(pwd)/local_server_config.yaml:/home/local_server_config.yaml \
    hfl-local-server:1.0
```

```
docker run \
    -p 8082:8080 \
    -v $(pwd)/local_server_config.yaml:/home/local_server_config.yaml \
    hfl-local-server:1.0
```

### Run Clients:

First, change IP address in configuration YAMLs (`client_config.yaml` and other three) to match IP address of your host.

```
cd client
```

```
docker build -t hfl-client:1.0 . 
```

Run four clients:
```
docker run \
    -v $(pwd)/task.py:/home/task.py \
    -v $(pwd)/client_config.yaml:/home/client_config.yaml \
    hfl-client:1.0
```
```
docker run \
    -v $(pwd)/task.py:/home/task.py \
    -v $(pwd)/client_config_2.yaml:/home/client_config.yaml \
    hfl-client:1.0
```
```
docker run \
    -v $(pwd)/task.py:/home/task.py \
    -v $(pwd)/client_config_3.yaml:/home/client_config.yaml \
    hfl-client:1.0
```
```
docker run \
    -v $(pwd)/task.py:/home/task.py \
    -v $(pwd)/client_config_4.yaml:/home/client_config.yaml \
    hfl-client:1.0
```