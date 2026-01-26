cd ~/Desktop/dippro
sudo bash scripts/cleanup.sh default
cd ~/Desktop
rm -rf dippro
git clone https://github.com/tisljaricleon/dippro.git
sudo cp /etc/rancher/k3s/k3s.yaml ~/Desktop/dippro/configs/cluster/kube_config.yaml
sudo chmod 644 ~/Desktop/dippro/configs/cluster/kube_config.yaml