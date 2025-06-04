# Inference and visualization for YOLOv8 damage segmentation on unseen images
from ultralytics import YOLO
import os
from glob import glob
import sys

def run_inference():    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'damage', 'weights', 'weights', 'best.pt')
    img_dir = os.path.join(base_dir, 'damage_detection_dataset', 'img')
    out_dir = os.path.join(base_dir, 'inference_results', 'damage')
    
    # Validate paths
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
    
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at {img_dir}")
        return
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all images in the dataset
    all_imgs = sorted(glob(os.path.join(img_dir, '*.jpg')))
    if not all_imgs:
        print(f"No images found in {img_dir}")
        return
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Run inference and save results
        for img_path in all_imgs:
            try:
                results = model.predict(
                    source=img_path,
                    save=True,
                    project=out_dir,
                    name='',
                    imgsz=640,
                    conf=0.25
                )
                print(f'Processed: {os.path.basename(img_path)}')
            except Exception as e:
                print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
                continue
        
        print(f'Inference complete. Results saved to {out_dir}')
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

if __name__ == '__main__':
    run_inference()
