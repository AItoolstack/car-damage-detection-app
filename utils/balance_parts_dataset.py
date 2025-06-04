import json
import os
import shutil
from collections import defaultdict
import random
from tqdm import tqdm

def create_balanced_dataset(source_json, source_img_dir, target_dir, min_samples=50):
    """
    Create a balanced dataset for parts detection by sampling images with different parts.
    
    Args:
        source_json (str): Path to source COCO JSON file
        source_img_dir (str): Path to source images directory
        target_dir (str): Path to target directory for balanced dataset
        min_samples (int): Minimum number of samples per class
    """
    # Create target directories
    os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels'), exist_ok=True)
    
    # Load COCO annotations
    with open(source_json, 'r') as f:
        coco = json.load(f)
    
    # Group images by parts they contain
    images_by_part = defaultdict(set)
    image_to_anns = defaultdict(list)
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        images_by_part[cat_id].add(img_id)
        image_to_anns[img_id].append(ann)
    
    # Find images with balanced representation
    selected_images = set()
    for part_images in images_by_part.values():
        # Sample min_samples images for each part
        sample_size = min(min_samples, len(part_images))
        selected_images.update(random.sample(list(part_images), sample_size))
    
    # Copy selected images and create labels
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    
    print(f"Creating balanced dataset with {len(selected_images)} images...")
    for img_id in tqdm(selected_images):
        # Copy image
        src_img = os.path.join(source_img_dir, id_to_filename[img_id])
        dst_img = os.path.join(target_dir, 'images', id_to_filename[img_id])
        shutil.copy2(src_img, dst_img)
        
        # Create YOLO label
        base_name = os.path.splitext(id_to_filename[img_id])[0]
        label_file = os.path.join(target_dir, 'labels', f"{base_name}.txt")
        
        # Convert annotations to YOLO format
        anns = image_to_anns[img_id]
        label_lines = []
        
        # Get image dimensions
        from PIL import Image
        im = Image.open(src_img)
        w, h = im.size
        
        for ann in anns:
            cat_id = ann['category_id']
            # Convert segmentation to YOLO format
            for seg in ann['segmentation']:
                seg_norm = [str(x/w) if i%2==0 else str(x/h) for i,x in enumerate(seg)]
                label_lines.append(f"{cat_id} {' '.join(seg_norm)}")
        
        # Write label file
        with open(label_file, 'w') as f:
            f.write('\n'.join(label_lines))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    # Process training set
    create_balanced_dataset(
        source_json=os.path.join(base_dir, "damage_detection_dataset", "train", "COCO_mul_train_annos.json"),
        source_img_dir=os.path.join(base_dir, "damage_detection_dataset", "img"),
        target_dir=os.path.join(base_dir, "data", "parts", "balanced", "train"),
        min_samples=50
    )
    
    # Process validation set
    create_balanced_dataset(
        source_json=os.path.join(base_dir, "damage_detection_dataset", "val", "COCO_mul_val_annos.json"),
        source_img_dir=os.path.join(base_dir, "damage_detection_dataset", "img"),
        target_dir=os.path.join(base_dir, "data", "parts", "balanced", "val"),
        min_samples=10
    )