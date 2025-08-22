import os
import shutil
from PIL import Image

def clear_and_create_directories():
    """Supprime et recrée les dossiers dataset/from et dataset/to"""
    from_dir = "dataset/from"
    to_dir = "dataset/to"
    
    # Supprimer les dossiers s'ils existent
    if os.path.exists(from_dir):
        shutil.rmtree(from_dir)
        print(f"Supprimé: {from_dir}")
    
    if os.path.exists(to_dir):
        shutil.rmtree(to_dir)
        print(f"Supprimé: {to_dir}")
    
    # Recréer les dossiers
    os.makedirs(from_dir, exist_ok=True)
    os.makedirs(to_dir, exist_ok=True)
    print(f"Créé: {from_dir} et {to_dir}")

def copy_and_mirror_images():
    """Copie les images de images_crop vers dataset/from et leurs miroirs vers dataset/to"""
    crop_dir = "images_crop"
    from_dir = "dataset/from"
    to_dir = "dataset/to"
    
    if not os.path.exists(crop_dir):
        print(f"Erreur: Le dossier {crop_dir} n'existe pas!")
        return
    
    images = sorted([f for f in os.listdir(crop_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not images:
        print(f"Aucune image trouvée dans {crop_dir}")
        return
    
    print(f"Traitement de {len(images)} images...")
    
    for img_name in images:
        src_path = os.path.join(crop_dir, img_name)
        from_path = os.path.join(from_dir, img_name)
        to_path = os.path.join(to_dir, img_name)
        
        try:
            # Copier l'image originale vers dataset/from
            shutil.copy2(src_path, from_path)
            
            # Créer la version miroir et la sauver dans dataset/to
            with Image.open(src_path) as img:
                mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mirrored_img.save(to_path)
            
            print(f"Traité: {img_name}")
            
        except Exception as e:
            print(f"Erreur avec {img_name}: {e}")
    
    print(f"Terminé! {len(images)} images copiées dans dataset/from")
    print(f"Terminé! {len(images)} images miroir créées dans dataset/to")

def main():
    print("=== Script de préparation du dataset ===")
    print("1. Nettoyage des dossiers dataset/from et dataset/to")
    clear_and_create_directories()
    
    print("\n2. Copie des images et création des miroirs")
    copy_and_mirror_images()
    
    print("\n=== Terminé ===")

if __name__ == "__main__":
    main()
