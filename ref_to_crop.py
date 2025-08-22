import os
from PIL import Image
import face_recognition
import numpy as np

def is_valid_image(file_path):
    """
    Vérifie si une image est valide et peut être traitée
    """
    try:
        # Vérifier que le fichier existe et n'est pas vide
        if not os.path.exists(file_path):
            print(f"❌ Fichier inexistant: {file_path}")
            return False
            
        if os.path.getsize(file_path) == 0:
            print(f"❌ Fichier vide: {file_path}")
            return False
        
        # Vérifier que PIL peut ouvrir l'image
        with Image.open(file_path) as img:
            # Vérifier les dimensions minimales
            if img.size[0] < 100 or img.size[1] < 100:
                print(f"❌ Image trop petite ({img.size[0]}x{img.size[1]}): {file_path}")
                return False
            
            # Vérifier le format d'image
            if img.format not in ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF']:
                print(f"❌ Format non supporté ({img.format}): {file_path}")
                return False
            
            # Vérifier que l'image n'est pas corrompue
            img.verify()
        
        # Double vérification : ré-ouvrir pour s'assurer que verify() n'a pas fermé l'image
        with Image.open(file_path) as img:
            # Tenter de charger quelques pixels pour détecter la corruption
            img.load()
        
        # Vérifier avec face_recognition
        test_img = face_recognition.load_image_file(file_path)
        if test_img is None or test_img.size == 0:
            print(f"❌ Impossible de charger avec face_recognition: {file_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la validation de {file_path}: {e}")
        return False

input_dir = "images_ref"
output_dir = "images_crop"
os.makedirs(output_dir, exist_ok=True)

def create_face_centered_crop(img, face_location, target_size=512):
    """
    Crée un crop carré centré sur le visage en gardant le maximum de fond
    """
    h, w = img.shape[:2]
    top, right, bottom, left = face_location
    
    # Centre du visage
    face_center_y = (top + bottom) // 2
    face_center_x = (left + right) // 2
    
    # Taille du visage
    face_width = right - left
    face_height = bottom - top
    face_size = max(face_width, face_height)
    
    # Calculer la taille du crop pour inclure suffisamment de fond
    # On veut que le visage occupe environ 40-60% du crop final
    crop_size = int(face_size * 2.5)  # Le visage fera ~40% de l'image finale
    crop_size = min(crop_size, min(w, h))  # Ne pas dépasser la taille de l'image
    
    # Centrer le crop sur le visage
    crop_left = max(0, face_center_x - crop_size // 2)
    crop_top = max(0, face_center_y - crop_size // 2)
    crop_right = min(w, crop_left + crop_size)
    crop_bottom = min(h, crop_top + crop_size)
    
    # Ajuster si on est en bordure d'image
    if crop_right - crop_left < crop_size:
        if crop_left == 0:
            crop_right = min(w, crop_size)
        else:
            crop_left = max(0, w - crop_size)
    
    if crop_bottom - crop_top < crop_size:
        if crop_top == 0:
            crop_bottom = min(h, crop_size)
        else:
            crop_top = max(0, h - crop_size)
    
    return img[crop_top:crop_bottom, crop_left:crop_right]

images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
count = 1
valid_count = 0
invalid_count = 0

print(f"🔍 Vérification de {len(images)} images dans {input_dir}...")

for fname in images:
    path = os.path.join(input_dir, fname)
    
    # Vérifier la validité de l'image AVANT traitement
    if not is_valid_image(path):
        invalid_count += 1
        print(f"⏭️  Ignoré: {fname}")
        continue
    
    try:
        img = face_recognition.load_image_file(path)
        faces = face_recognition.face_locations(img)

        if faces:
            # Prendre le plus grand visage détecté
            face_location = max(faces, key=lambda box: (box[2]-box[0])*(box[1]-box[3]))
            crop = create_face_centered_crop(img, face_location)
        else:
            # Si aucun visage détecté, recadrer au centre en carré
            h, w = img.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            crop = img[start_y:start_y+size, start_x:start_x+size]

        pil_img = Image.fromarray(crop)
        pil_img = pil_img.resize((512, 512), Image.LANCZOS)
        out_name = f"img_{count:04d}.png"
        pil_img.save(os.path.join(output_dir, out_name))
        print(f"✅ Saved {out_name} - Face detected: {'Yes' if faces else 'No'}")
        count += 1
        valid_count += 1
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {fname}: {e}")
        invalid_count += 1

print(f"\n📊 Résumé du traitement:")
print(f"✅ Images traitées avec succès: {valid_count}")
print(f"❌ Images ignorées/erreurs: {invalid_count}")
print(f"📁 Total images générées: {count-1}")