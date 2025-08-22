import os
from PIL import Image
import face_recognition

def is_valid_image(file_path):
    """
    Vérifie si une image est valide et peut être traitée
    """
    try:
        # Vérifier la taille du fichier
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
            
            # Tenter de charger les données de l'image
            img.verify()
        
        # Double vérification avec face_recognition
        test_img = face_recognition.load_image_file(file_path)
        if test_img is None or test_img.size == 0:
            print(f"❌ Impossible de charger avec face_recognition: {file_path}")
            return False
        
        print(f"✅ Image valide: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la validation de {file_path}: {e}")
        return False

input_dir = "images_ref"
output_dir = "images_crop"
os.makedirs(output_dir, exist_ok=True)

images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
count = 1
valid_count = 0
invalid_count = 0

print(f"🔍 Vérification de {len(images)} images...")

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
            top, right, bottom, left = max(faces, key=lambda box: (box[2]-box[0])*(box[1]-box[3]))
            # Ajouter une marge autour du visage
            h, w = img.shape[:2]
            margin = int(0.3 * max(bottom-top, right-left))
            top = max(0, top - margin)
            bottom = min(h, bottom + margin)
            left = max(0, left - margin)
            right = min(w, right + margin)
            crop = img[top:bottom, left:right]
        else:
            # Si aucun visage détecté, utiliser l'image entière
            crop = img

        pil_img = Image.fromarray(crop)
        pil_img = pil_img.resize((512, 512), Image.LANCZOS)
        out_name = f"img_{count:04d}.png"
        pil_img.save(os.path.join(output_dir, out_name))
        print(f"💾 Saved {out_name}")
        count += 1
        valid_count += 1
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {fname}: {e}")
        invalid_count += 1

print(f"\n📊 Résumé:")
print(f"✅ Images traitées avec succès: {valid_count}")
print(f"❌ Images ignorées/erreurs: {invalid_count}")
print(f"📁 Total images générées: {count-1}")