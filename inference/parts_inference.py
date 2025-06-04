# Inference on unseen images for YOLOv8 parts segmentation
from ultralytics import YOLO
import os
from glob import glob

def run_inference():    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'parts', 'weights', 'weights', 'best.pt')
    img_dir = os.path.join(base_dir, 'damage_detection_dataset', 'img')
    train_dir = os.path.join(base_dir, 'data', 'data_yolo_for_training', 'car_parts_damage_dataset', 'images', 'train')
    out_dir = os.path.join(base_dir, 'inference_results', 'parts')
    
    # Validate paths
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
    
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at {img_dir}")
        return
    
    if not os.path.exists(train_dir):
        print(f"Warning: Training directory not found at {train_dir}")
        print("Will run inference on all images instead of just unseen ones")
        train_imgs = set()
    else:
        # Get all images used for training
        train_imgs = set(os.listdir(train_dir))
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all images in original dataset
    all_imgs = set(os.listdir(img_dir))
    # Select images not used in training
    unseen_imgs = sorted(list(all_imgs - train_imgs))
    
    if not unseen_imgs:
        print(f"No images found for inference in {img_dir}")
        return
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Class names for visualization
        class_names = ['headlamp', 'front_bumper', 'hood', 'door', 'rear_bumper']
        
        # Run inference on each unseen image
        for img_name in unseen_imgs:
            try:
                img_path = os.path.join(img_dir, img_name)
                results = model.predict(
                    source=img_path,
                    save=True,
                    project=out_dir,
                    name='',
                    imgsz=640,
                    conf=0.25,
                    classes=list(range(len(class_names)))  # All classes
                )
                print(f'Processed: {img_name}')
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue
        
        print(f'Inference complete. Results saved to {out_dir}')
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

if __name__ == '__main__':
    run_inference()
