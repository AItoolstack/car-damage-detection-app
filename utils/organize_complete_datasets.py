import os
import shutil
import json
from collections import defaultdict
import random
from tqdm import tqdm
from PIL import Image

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, class_map, split='train'):
    """Convert COCO format annotations to YOLO format"""
    if not os.path.exists(coco_json_path):
        print(f"Warning: JSON file not found: {coco_json_path}")
        return set()
        
    if not os.path.exists(images_dir):
        print(f"Warning: Images directory not found: {images_dir}")
        return set()
    
    print(f"\nProcessing {split} split...")
    
    # Create output directories
    labels_dir = os.path.join(output_dir, 'labels', split)
    images_dir_out = os.path.join(output_dir, 'images', split)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir_out, exist_ok=True)
    
    # Load COCO annotations
    try:
        with open(coco_json_path, 'r') as f:
            coco = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {coco_json_path}")
        return set()
    
    # Create id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    
    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # Process each image
    processed_images = set()
    for img_id, anns in tqdm(img_to_anns.items(), desc=f"Converting {split} set"):
        img_file = id_to_filename[img_id]
        img_path = os.path.join(images_dir, img_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping...")
            continue
        
        try:
            # Copy image
            shutil.copy2(img_path, os.path.join(images_dir_out, img_file))
            
            # Get image dimensions
            with Image.open(img_path) as im:
                w, h = im.size
            
            # Convert annotations
            label_lines = []
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in class_map:
                    print(f"Warning: Unknown category ID {cat_id} in {img_file}")
                    continue
                yolo_cls = class_map[cat_id]
                
                # Convert segmentation points
                for seg in ann['segmentation']:
                    coords = [str(x/w) if i%2==0 else str(x/h) for i,x in enumerate(seg)]
                    label_lines.append(f"{yolo_cls} {' '.join(coords)}")
            
            # Write label file
            label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))
            
            processed_images.add(img_id)
            
        except (IOError, OSError) as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    return processed_images

def create_balanced_dataset(source_json, images_dir, output_dir, class_map, min_samples=50, split='train'):
    """Create balanced dataset by sampling equal number of images per class"""
    print(f"\nCreating balanced dataset for {split} split...")
    
    # Create output directories
    labels_dir = os.path.join(output_dir, 'labels', split)
    images_dir_out = os.path.join(output_dir, 'images', split)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir_out, exist_ok=True)
    
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
    
    # Sample images for balanced dataset
    selected_images = set()
    for part_images in images_by_part.values():
        sample_size = min(min_samples, len(part_images))
        selected_images.update(random.sample(list(part_images), sample_size))
    
    # Convert selected images to YOLO format
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    
    print(f"Processing {len(selected_images)} images for balanced {split} set...")
    for img_id in tqdm(selected_images):
        img_file = id_to_filename[img_id]
        img_path = os.path.join(images_dir, img_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping...")
            continue
        
        # Copy image
        shutil.copy2(img_path, os.path.join(images_dir_out, img_file))
        
        # Get image dimensions
        with Image.open(img_path) as im:
            w, h = im.size
        
        # Convert annotations
        label_lines = []
        for ann in image_to_anns[img_id]:
            cat_id = ann['category_id']
            yolo_cls = class_map[cat_id]
            
            # Convert segmentation points
            for seg in ann['segmentation']:
                coords = [str(x/w) if i%2==0 else str(x/h) for i,x in enumerate(seg)]
                label_lines.append(f"{yolo_cls} {' '.join(coords)}")
        
        # Write label file
        label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        with open(label_file, 'w') as f:
            f.write('\n'.join(label_lines))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(base_dir, 'damage_detection_dataset')
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Set up output directories
    car_damage_dir = os.path.join(base_dir, 'data', 'data_yolo_for_training', 'car_damage_dataset')
    car_parts_dir = os.path.join(base_dir, 'data', 'data_yolo_for_training', 'car_parts_damage_dataset')
    
    # Class mappings
    damage_class_map = {1: 0}  # Assuming damage is class 1 in COCO format
    parts_class_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}  # headlamp, front_bumper, hood, door, rear_bumper
    
    # Process car damage dataset (full dataset)
    print("\nProcessing Car Damage Dataset...")
    for split in ['train', 'val', 'test']:
        json_name = 'COCO_train_annos.json' if split == 'train' else 'COCO_val_annos.json'
        json_path = os.path.join(source_dir, split, json_name)
        images_dir = os.path.join(source_dir, split)
        
        if os.path.exists(json_path):
            convert_coco_to_yolo(
                json_path,
                images_dir,
                car_damage_dir,
                damage_class_map,
                split
            )
        else:
            print(f"Warning: JSON file not found for {split} split: {json_path}")
    
    # Process car parts dataset (balanced training, original val/test)
    print("\nProcessing Car Parts Dataset...")
    # Training set - balanced
    train_json = os.path.join(source_dir, 'train', 'COCO_mul_train_annos.json')
    if os.path.exists(train_json):
        create_balanced_dataset(
            train_json,
            os.path.join(source_dir, 'train'),
            car_parts_dir,
            parts_class_map,
            min_samples=50,
            split='train'
        )
    else:
        print(f"Warning: Training JSON file not found: {train_json}")
    
    # Validation and test sets - original
    for split in ['val', 'test']:
        json_path = os.path.join(source_dir, split, 'COCO_mul_val_annos.json')
        images_dir = os.path.join(source_dir, split)
        
        if os.path.exists(json_path):
            convert_coco_to_yolo(
                json_path,
                images_dir,
                car_parts_dir,
                parts_class_map,
                split
            )
        else:
            print(f"Warning: JSON file not found for {split} split: {json_path}")

if __name__ == '__main__':
    main()
