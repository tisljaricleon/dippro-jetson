# global aggregator
kubectl label --overwrite nodes raspberrypi fl/type=global_aggregator

# Jetson (ARM64) client node
kubectl label --overwrite nodes sem2-desktop fl/type=client kubernetes.io/arch=arm64
kubectl label --overwrite nodes sem2-desktop comm/raspberrypi=100
kubectl label --overwrite nodes sem2-desktop fl/num-partitions=2
kubectl label --overwrite nodes sem2-desktop fl/partition-id=0

# Normal (x86_64) client node
kubectl label --overwrite nodes worker-desktop fl/type=client kubernetes.io/arch=amd64
kubectl label --overwrite nodes worker-desktop comm/raspberrypi=100
kubectl label --overwrite nodes worker-desktop fl/num-partitions=2
kubectl label --overwrite nodes worker-desktop fl/partition-id=1