cd ~/Desktop/dippro-jetson
sudo bash scripts/cleanup.sh default
cd ~/Desktop
rm -rf dippro-jetson
git clone https://github.com/tisljaricleon/dippro-jetson.git
sudo cp /etc/rancher/k3s/k3s.yaml ~/Desktop/dippro-jetson/configs/cluster/kube_config.yaml
sudo chmod 644 ~/Desktop/dippro-jetson/configs/cluster/kube_config.yaml