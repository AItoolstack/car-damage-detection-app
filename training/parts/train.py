# YOLOv8 segmentation training for car parts detection
from ultralytics import YOLO
import multiprocessing
import os

def train():
    # Start from YOLOv8 medium segmentation model
    model = YOLO('../../models/yolov8m-seg.pt')

    # Get the absolute path to the data.yaml file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'data.yaml')

    # Train with optimized parameters for parts detection
    model.train(
        data=data_yaml_path,  # Path to data configuration file
        epochs=100,  # Number of epochs
        imgsz=640,  # Image size
        batch=4,    # Batch size
        workers=4,  # Number of workers
        project='../../models/parts/weights',  # Save directory
        name='yolov8_parts_final',  # Run name
        
        # Learning rate strategy
        lr0=0.0002,  # Initial learning rate
        lrf=0.000001,  # Final learning rate
        warmup_epochs=20,  # Fewer warmup epochs for parts
        warmup_momentum=0.8,
        cos_lr=True,  # Use cosine learning rate scheduler
        
        # Loss weights
        box=8.0,  # Box loss gain
        cls=4.0,  # Class loss gain
        dfl=2.5,  # DFL loss gain
        
        # Augmentation settings
        augment=True,
        mosaic=0.5,
        mixup=0.2,
        copy_paste=0.1,
        degrees=20.0,
        translate=0.2,
        scale=0.4,
        shear=10.0,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Other optimization settings
        overlap_mask=True,  # Overlap mask segments
        mask_ratio=4,  # Mask downsampling ratio
        single_cls=False,  # Multiple classes for parts
        rect=False,  # Rectangular training
        cache=False,  # Cache images for faster training
        patience=50,  # Early stopping patience
        close_mosaic=10,  # Close mosaic augmentation epochs
        deterministic=True,  # Deterministic mode
        seed=42,  # Random seed
        device=0  # GPU device
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
