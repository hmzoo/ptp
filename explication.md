# 🚀 Explication détaillée du script `train.py`

## 🎯 **Objectif du script**
Le script `train.py` implémente un modèle d'intelligence artificielle de type **image-to-image** utilisant l'architecture **U-Net++** avec des fonctionnalités avancées de rectification géométrique. Il transforme des images d'entrée en images de sortie en apprenant les correspondances entre des paires d'images.

---

## 📁 **Architecture générale**

### **1. Structure des données**
```
dataset/
├── from/          # Images d'entrée (sources)
├── to/            # Images de sortie (cibles)
├── inter/         # Images intermédiaires générées pendant l'entraînement
└── results/       # Résultats finaux
```

### **2. Flux principal**
```
Images FROM → Modèle U-Net++ → Images TO
     ↓              ↓              ↓
  512x512        Traitement     512x512
   (RGB)         IA avancé      (RGB)
```

---

## 🧠 **Architecture du modèle : U-Net++**

### **Composants principaux :**

#### **A. Spatial Transformer Network (STN)**
```python
class SpatialTransformerNetwork(nn.Module)
```
- **Rôle** : Corrige automatiquement les déformations géométriques
- **Fonctionnement** : 
  - Analyse l'image d'entrée
  - Calcule une transformation affine (rotation, translation, échelle)
  - Applique la correction géométrique avant le traitement principal

#### **B. U-Net++ avec connexions denses**
```python
class UNetPP(nn.Module)
```
- **Encodeur** : Réduit progressivement la taille (512→256→128→64→32)
- **Décodeur** : Augmente progressivement la taille (32→64→128→256→512)
- **Skip connections** : Préserve les détails fins
- **Batch Normalization** : Stabilise l'entraînement
- **Dropout** : Évite le sur-apprentissage

---

## 📊 **Dataset et chargement des données**

### **ImagePairDataset**
```python
class ImagePairDataset(Dataset):
    def __init__(self, from_dir, to_dir, transform=None):
        self.filenames = sorted(os.listdir(from_dir))  # Ordre alphabétique
```

**Fonctionnement :**
1. **Correspondance par nom** : `img_0001.png` dans `from/` ↔ `img_0001.png` dans `to/`
2. **Chargement automatique** : PIL charge et convertit en RGB
3. **Transformation** : Conversion en tenseurs PyTorch [0,1]

### **DataLoader**
```python
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```
- **Batch size 8** : Traite 8 paires d'images simultanément
- **Shuffle=True** : Ordre aléatoire à chaque époque pour éviter le sur-apprentissage
- **Parallélisation GPU** : Traitement efficace sur CUDA

---

## 🎯 **Fonctions de perte avancées**

### **1. Loss L1 (MAE)**
```python
loss_l1 = l1_criterion(output, target)
```
- **Mesure** : Différence absolue pixel par pixel
- **Avantage** : Préserve les détails fins

### **2. Perceptual Loss**
```python
class PerceptualLoss(nn.Module):
    def __init__(self):
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
```
- **Principe** : Utilise un réseau VGG16 pré-entraîné
- **Mesure** : Similarité des caractéristiques de haut niveau
- **Avantage** : Résultats plus naturels visuellement

### **3. SSIM Loss**
```python
class SSIMLoss(nn.Module):
```
- **Mesure** : Similarité structurelle entre images
- **Évalue** : Luminance, contraste, structure
- **Avantage** : Correspond mieux à la perception humaine

### **Loss combinée**
```python
loss = loss_l1 + 0.1 * loss_perc + 0.1 * loss_ssim
```
- **Pondération** : L1 (poids 1.0), Perceptual (poids 0.1), SSIM (poids 0.1)
- **Équilibre** : Détails fins + naturalité visuelle + structure

---

## ⚙️ **Processus d'entraînement**

### **1. Initialisation**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetPP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### **2. Boucle d'entraînement (10 000 époques)**
```python
for epoch in range(start_epoch, epochs):
    for idx, (from_img, to_img) in enumerate(dataloader):
```

**Étapes par batch :**
1. **Chargement** : 8 paires d'images sur GPU
2. **Normalisation** : `to_img_tanh = to_img * 2.0 - 1.0` (pour activation Tanh)
3. **Forward pass** : `output = model(from_img)`
4. **Calcul des pertes** : L1 + Perceptual + SSIM
5. **Backpropagation** : Mise à jour des poids
6. **Accumulation** : Statistiques pour affichage

### **3. Optimisations GPU**
```python
if device.type == 'cuda':
    with torch.amp.autocast('cuda'):  # Mixed precision
        output = model(from_img)
        loss = ...
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
- **Mixed Precision** : Utilise FP16 pour accélérer l'entraînement
- **Scaler** : Évite l'underflow numérique

---

## 💾 **Système de checkpoints robuste**

### **Sauvegarde automatique**
```python
if (epoch + 1) % 20 == 0:  # Toutes les 20 époques
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
    }
    torch.save(checkpoint, ckpt_path)
```

### **Gestion intelligente**
- **Limite** : Garde seulement les 3 derniers checkpoints
- **Nettoyage automatique** : Supprime les anciens pour économiser l'espace
- **Récupération** : Détecte et supprime les checkpoints corrompus

### **Reprise automatique**
```python
if last_ckpt and os.path.exists(last_ckpt):
    checkpoint = torch.load(last_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
```
- **Détection automatique** : Trouve le dernier checkpoint
- **Compatibilité** : Gère les anciens et nouveaux formats
- **Choix utilisateur** : Option pour repartir de zéro

---

## 🖼️ **Sauvegarde des résultats**

### **Images intermédiaires (chaque époque)**
```python
if idx == 0:  # Premier batch de l'époque
    save_image(output_save.cpu(), f'epoch{epoch+1}_output.png')
    save_image(from_img[0].cpu(), f'epoch{epoch+1}_input.png')
    save_image(to_img[0].cpu(), f'epoch{epoch+1}_target.png')
```
- **Suivi visuel** : Voir l'évolution du modèle
- **Comparaison** : Input → Output → Target côte à côte

### **Normalisation pour sauvegarde**
```python
output_save = (output[0] + 1) / 2  # [-1,1] → [0,1]
```
- **Conversion** : De l'espace Tanh vers l'espace image standard

---

## 📈 **Métriques et monitoring**

### **Affichage détaillé**
```python
print(f"Epoch {epoch+1}/{epochs}, Total Loss: {running_loss/len(dataset):.4f} | 
       L1: {running_l1/len(dataset):.4f} | 
       Perceptual: {running_perc/len(dataset):.4f} | 
       SSIM: {running_ssim/len(dataset):.4f}")
```

### **Calcul des moyennes**
- **running_loss** : Accumule la perte totale
- **Division par len(dataset)** : Moyenne par image
- **Suivi séparé** : Chaque composante de la perte

---

## 🔧 **Paramètres clés**

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **Époques** | 10 000 | Nombre total d'itérations |
| **Batch size** | 8 | Images traitées simultanément |
| **Learning rate** | 1e-3 | Taux d'apprentissage Adam |
| **Checkpoints** | Toutes les 20 époques | Fréquence de sauvegarde |
| **Rétention** | 3 derniers | Nombre de checkpoints gardés |
| **Résolution** | 512×512 | Taille des images d'entrée/sortie |

---

## 🚀 **Workflow complet**

1. **Préparation** : Images cropées → `dataset/from` et `dataset/to`
2. **Initialisation** : Chargement du modèle/checkpoint
3. **Entraînement** : 10 000 époques avec loss combinée
4. **Monitoring** : Sauvegarde d'images + métriques
5. **Checkpoints** : Sauvegarde régulière + nettoyage
6. **Résultats** : Images générées dans `dataset/inter`

---

## 🎯 **Objectif final**
Apprendre une fonction de transformation **f: FROM → TO** qui peut généraliser à de nouvelles images non vues pendant l'entraînement, en préservant les détails fins tout en produisant des résultats visuellement naturels.
