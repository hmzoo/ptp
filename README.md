Pour installer Python 3.11 sur Ubuntu 24.04, ouvrez un terminal et exécutez :
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

Une fois Python installé, il est recommandé d'utiliser un environnement virtuel pour isoler les dépendances de votre projet.

Pour créer un environnement cloisonné (virtuel) pour le projet `ptp`, placez-vous dans le dossier du projet et lancez :
```bash
python3.11 -m venv venv
```
Activez ensuite l'environnement :
```bash
source venv/bin/activate
```


Pour installer toutes les dépendances du projet listées dans requirements.txt :
```bash
pip install -r requirements.txt
```

Pour installer torch et torchvision avec support GPU (CUDA), utilisez la commande officielle PyTorch :
```bash
# Pour CUDA 12.1 (adaptez selon votre version CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Consultez https://pytorch.org/get-started/locally/ pour les instructions adaptées à votre version CUDA et votre OS.

Vous pouvez maintenant installer les dépendances du projet sans affecter les autres projets Python de votre système.