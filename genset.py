

import os
import random
from PIL import Image, ImageDraw

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def draw_random_shape(draw, width, height):
    shape_type = random.choice(['rectangle', 'ellipse', 'line'])
    x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
    x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
    # Assure l'ordre des coordonnées
    x0, x1_sorted = sorted([x1, x2])
    y0, y1_sorted = sorted([y1, y2])
    color = random_color()
    if shape_type == 'rectangle':
        draw.rectangle([x0, y0, x1_sorted, y1_sorted], fill=color, outline=random_color())
    elif shape_type == 'ellipse':
        draw.ellipse([x0, y0, x1_sorted, y1_sorted], fill=color, outline=random_color())
    elif shape_type == 'line':
        draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 10))

def generate_shape_images(num_images=10, width=512, height=512, output_dir='./dataset/from'):
    os.makedirs(output_dir, exist_ok=True)
    to_dir = output_dir.replace('/from', '/to')
    os.makedirs(to_dir, exist_ok=True)
    for i in range(num_images):
        img = Image.new('RGB', (width, height), color=random_color())
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(5, 15)):
            draw_random_shape(draw, width, height)
        filename = f"img_{i+1:04d}.png"
        img.save(os.path.join(output_dir, filename))
        # Génère l'image symétrique
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flipped.save(os.path.join(to_dir, filename))
    print(f'{num_images} images avec formes générées dans {output_dir} et {to_dir}.')

if __name__ == '__main__':
    generate_shape_images(num_images=10)