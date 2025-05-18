import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ✅ Set Seed for Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ✅ Check CUDA Availability
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"✅ Using Device: {DEVICE}")

# ✅ Define Kaggle Dataset Paths
# Kaggle paths are different from local paths
DATASET_ROOT = "/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset"
WORK_DIR = "/kaggle/working/violence_detection"
os.makedirs(WORK_DIR, exist_ok=True)

# ✅ Create Directories for Images & Labels
IMAGE_DIR = os.path.join(WORK_DIR, "images")
LABEL_DIR = os.path.join(WORK_DIR, "labels")
TRAIN_IMG_DIR = os.path.join(IMAGE_DIR, "train")
VAL_IMG_DIR = os.path.join(IMAGE_DIR, "val")
TRAIN_LABEL_DIR = os.path.join(LABEL_DIR, "train")
VAL_LABEL_DIR = os.path.join(LABEL_DIR, "val")

for path in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR]:
    os.makedirs(path, exist_ok=True)

# ✅ Training Parameters
TRAIN_RATIO = 0.8
IMG_SIZE = 640  # Reduced from 1280 for better speed/performance balance
BATCH_SIZE = 16  # Increased from 8
EPOCHS = 50
LEARNING_RATE = 0.001
FRAME_SKIP = 5  # Extract every 5th frame instead of 10th for more training data
PATIENCE = 10  # Early stopping patience

# Helper function to log progress and save output
def log_progress(message):
    print(f"✅ {message}")
    with open(os.path.join(WORK_DIR, "training_log.txt"), "a") as f:
        f.write(f"{message}\n")

# ✅ Optimized Frame Extraction with tqdm
def extract_frames(input_dir, output_dir, label, frame_skip=FRAME_SKIP):
    """Extract frames from videos and save them as images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # List all MP4 files in the directory
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
    
    log_progress(f"Found {len(video_files)} videos in {input_dir}")
    
    def process_video(video_path):
        if not os.path.exists(video_path):
            return 0
            
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_frames = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a base name for frames from this video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                # Save frame with a unique name
                frame_name = f"{label}_{video_name}_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                extracted_frames += 1
                
            frame_count += 1
            
        cap.release()
        return extracted_frames
    
    # Use ThreadPoolExecutor for parallel processing
    total_extracted = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(
            executor.map(process_video, video_files),
            total=len(video_files),
            desc=f"Extracting {label} frames"
        ))
        total_extracted = sum(results)
    
    log_progress(f"Extracted {total_extracted} frames from {len(video_files)} {label} videos")
    return total_extracted

# ✅ Extract Frames from Violence and Non-Violence Videos
violence_path = os.path.join(DATASET_ROOT, "Violence")
nonviolence_path = os.path.join(DATASET_ROOT, "NonViolence")

violence_frames = extract_frames(violence_path, os.path.join(IMAGE_DIR, "violence"), "violence")
nonviolence_frames = extract_frames(nonviolence_path, os.path.join(IMAGE_DIR, "nonviolence"), "nonviolence")

log_progress(f"Total frames extracted: {violence_frames + nonviolence_frames}")

# ✅ Split Images into Train & Val Sets with tqdm
def split_train_val(image_dir, train_dir, val_dir, train_ratio=TRAIN_RATIO):
    """Split images into training and validation sets"""
    if not os.path.exists(image_dir):
        log_progress(f"Warning: Directory {image_dir} does not exist, skipping split...")
        return
    
    images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
    if not images:
        log_progress(f"Warning: No images found in {image_dir}, skipping split...")
        return
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=SEED)
    
    log_progress(f"Splitting {len(images)} images: {len(train_images)} train, {len(val_images)} validation")
    
    # Process training images
    for img in tqdm(train_images, desc=f"Moving train images to {train_dir}"):
        src = os.path.join(image_dir, img)
        dest = os.path.join(train_dir, img)
        if os.path.exists(src):
            os.rename(src, dest)
    
    # Process validation images
    for img in tqdm(val_images, desc=f"Moving validation images to {val_dir}"):
        src = os.path.join(image_dir, img)
        dest = os.path.join(val_dir, img)
        if os.path.exists(src):
            os.rename(src, dest)
    
    log_progress(f"Train/Val split completed for {image_dir}")

# Split the extracted frames
for category in ["violence", "nonviolence"]:
    split_train_val(
        os.path.join(IMAGE_DIR, category),
        os.path.join(TRAIN_IMG_DIR, category),
        os.path.join(VAL_IMG_DIR, category)
    )

# ✅ Create YOLO Labels for both classes
def create_labels(image_dir, label_dir, class_id):
    """Create YOLO format labels for images"""
    os.makedirs(label_dir, exist_ok=True)
    
    images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
    log_progress(f"Creating labels for {len(images)} images in {image_dir}")
    
    for img_name in tqdm(images, desc=f"Creating labels for class {class_id}"):
        txt_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        txt_path = os.path.join(label_dir, txt_name)
        
        # For violence detection, we're using a scene-level label
        # The bounding box covers the entire image
        with open(txt_path, "w") as f:
            # Format: class_id center_x center_y width height
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Create labels for both classes
for category, class_id in zip(["violence", "nonviolence"], [0, 1]):
    create_labels(
        os.path.join(TRAIN_IMG_DIR, category),
        os.path.join(TRAIN_LABEL_DIR, category),
        class_id
    )
    create_labels(
        os.path.join(VAL_IMG_DIR, category),
        os.path.join(VAL_LABEL_DIR, category),
        class_id
    )

log_progress("Labels created for all images")

# ✅ Create YOLO dataset configuration file (data.yaml)
yaml_content = f"""
# Violence Detection Dataset
train: {TRAIN_IMG_DIR}
val: {VAL_IMG_DIR}

# Number of classes
nc: 2

# Class names
names:
  0: violence
  1: nonviolence
"""

yaml_path = os.path.join(WORK_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

log_progress("Dataset configuration (data.yaml) created")

# ✅ Prepare for YOLOv8 Training
log_progress("Preparing for YOLOv8 training...")

# Clear GPU memory before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ✅ Train YOLOv8 Model with optimal parameters
def train_yolo_model(model_size="s"):
    """Train YOLOv8 model with optimized parameters"""
    model_path = f"yolov8{model_size}.pt"
    log_progress(f"Training YOLOv8-{model_size.upper()} model")
    
    # Initialize the model
    model = YOLO(model_path)
    
    # Define file paths
    results_dir = os.path.join(WORK_DIR, "results")
    best_model_path = os.path.join(results_dir, "best.pt")
    
    # Start training
    try:
        results = model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            optimizer="AdamW",  # AdamW optimizer
            cos_lr=True,        # Cosine learning rate scheduler
            patience=PATIENCE,  # Early stopping patience
            dropout=0.2,        # Dropout for regularization
            
            # Data augmentation params
            augment=True,
            hsv_h=0.015,        # HSV Hue augmentation
            hsv_s=0.7,          # HSV Saturation augmentation
            hsv_v=0.4,          # HSV Value augmentation
            flipud=0.5,         # Vertical flip prob
            fliplr=0.5,         # Horizontal flip prob
            mosaic=1.0,         # Mosaic augmentation prob
            mixup=0.3,          # Mixup augmentation prob
            
            # Training and loss params
            iou=0.5,            # IoU threshold
            lr0=LEARNING_RATE,  # Initial learning rate
            lrf=0.0001,         # Final learning rate factor
            label_smoothing=0.2,# Label smoothing
            
            # Saving and visualization
            save=True,          # Save training results
            save_period=5,      # Save model every 5 epochs
            project=results_dir,
            name="violence_detection",
            exist_ok=True,
            plots=True,         # Generate plots
            
            # Performance optimization
            workers=8,          # Number of worker threads
            cache=True,         # Cache images for faster training
        )
        
        # Save the best model to a predictable location
        if os.path.exists(os.path.join(results_dir, "violence_detection", "weights", "best.pt")):
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            os.rename(
                os.path.join(results_dir, "violence_detection", "weights", "best.pt"),
                best_model_path
            )
            log_progress(f"Best model saved to {best_model_path}")
        
        return results, best_model_path
    
    except Exception as e:
        log_progress(f"Error during training: {str(e)}")
        return None, None

# Try different model sizes based on available resources
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
    # Use larger model if GPU has >8GB memory
    log_progress("Using YOLOv8m model (medium size)")
    results, best_model_path = train_yolo_model("m")
else:
    # Use smaller model for limited resources
    log_progress("Using YOLOv8s model (small size)")
    results, best_model_path = train_yolo_model("s")

# ✅ Evaluate the trained model
def evaluate_model(model_path):
    """Evaluate the trained model on validation data"""
    if not os.path.exists(model_path):
        log_progress(f"Model not found at {model_path}, skipping evaluation")
        return
    
    log_progress("Evaluating model on validation data...")
    try:
        model = YOLO(model_path)
        results = model.val(
            data=yaml_path,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            plots=True,
        )
        
        # Log key metrics
        metrics = results.box
        log_progress(f"Model Evaluation Results:")
        log_progress(f"- mAP@0.5: {metrics.map50:.4f}")
        log_progress(f"- mAP@0.5:0.95: {metrics.map:.4f}")
        log_progress(f"- Precision: {metrics.p:.4f}")
        log_progress(f"- Recall: {metrics.r:.4f}")
        log_progress(f"- F1-Score: {metrics.f1:.4f}")
        
        return results
    
    except Exception as e:
        log_progress(f"Error during evaluation: {str(e)}")
        return None

# Evaluate the best model
if best_model_path and os.path.exists(best_model_path):
    evaluation_results = evaluate_model(best_model_path)
else:
    log_progress("No best model found for evaluation")

# ✅ Create a simple inference function for future use
def create_inference_script():
    """Create a script for running inference with the trained model"""
    script_path = os.path.join(WORK_DIR, "run_inference.py")
    
    script_content = '''
import os
import cv2
import torch
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def process_video(model_path, video_path, output_path=None, conf_threshold=0.5):
    """Process a video with the trained violence detection model"""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Load the model
    model = YOLO(model_path)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video if needed
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    violence_frames = 0
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            results = model(frame, conf=conf_threshold)[0]
            
            # Get detections
            is_violence = False
            for detection in results.boxes.data.tolist():
                class_id = int(detection[5])
                confidence = detection[4]
                
                if class_id == 0 and confidence >= conf_threshold:  # Violence detected
                    is_violence = True
                    violence_frames += 1
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add label
                    label = f"Violence: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add overall status to the frame
            status = "VIOLENCE DETECTED" if is_violence else "NO VIOLENCE"
            color = (0, 0, 255) if is_violence else (0, 255, 0)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Write the frame to output if needed
            if output_path:
                out.write(frame)
                
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    
    # Report results
    violence_percentage = (violence_frames / frame_count) * 100 if frame_count > 0 else 0
    print(f"Video analysis complete:")
    print(f"- Total frames: {frame_count}")
    print(f"- Violence frames: {violence_frames} ({violence_percentage:.2f}%)")
    print(f"- Overall classification: {'VIOLENT' if violence_percentage > 30 else 'NON-VIOLENT'}")
    
    return violence_frames, frame_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run violence detection on a video')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--output', type=str, help='Path to save the output video (optional)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    process_video(args.model, args.video, args.output, args.conf)
'''
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    log_progress(f"Inference script created at {script_path}")
    log_progress(f"Usage: python run_inference.py --model [MODEL_PATH] --video [VIDEO_PATH] --output [OUTPUT_PATH]")

# Create the inference script
create_inference_script()

log_progress("✅ Violence detection model training and evaluation complete!")
log_progress(f"All outputs saved to: {WORK_DIR}")
