import os
import shutil
import random

SOURCE_IMAGES_DIR = r"D:\projeto_placentas_clayton\dataset\Placentas\train\images"
SOURCE_LABELS_DIR = r"D:\projeto_placentas_clayton\dataset\Placentas\train\labels"

DEST_DIR = r"D:\projeto_placentas_clayton\dataset_ready_for_yolo"

SPLIT_RATIOS = (0.85, 0.15) 
# -------------------------------------

def split_dataset():
    print("--- Starting Dataset Split ---")
    
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"CRITICAL ERROR: Source image path not found:\n{SOURCE_IMAGES_DIR}")
        return
    if not os.path.exists(SOURCE_LABELS_DIR):
        print(f"CRITICAL ERROR: Source label path not found:\n{SOURCE_LABELS_DIR}")
        return

    pairs = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
    
    print(f"Scanning: {SOURCE_IMAGES_DIR}")
    files = os.listdir(SOURCE_IMAGES_DIR)
    
    for filename in files:
        name, ext = os.path.splitext(filename)
        if ext.lower() in extensions:
            label_filename = f"{name}.txt"
            label_path = os.path.join(SOURCE_LABELS_DIR, label_filename)
            
            if os.path.exists(label_path):
                pairs.append(filename)
            else:
                print(f"Warning: Missing label for {filename}")

    if not pairs:
        print("Error: No valid image/label pairs found.")
        return

    random.seed(42)
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * SPLIT_RATIOS[0])
    train_imgs = pairs[:split_idx]
    val_imgs = pairs[split_idx:]
    
    print(f"Found {len(pairs)} total pairs.")
    print(f"Splitting into: {len(train_imgs)} Train / {len(val_imgs)} Validation")

    if os.path.exists(DEST_DIR):
        try:
            shutil.rmtree(DEST_DIR)
        except:
            print("Warning: Could not clean previous destination folder.")

    for split, files in zip(['train', 'valid'], [train_imgs, val_imgs]):
        img_dest = os.path.join(DEST_DIR, split, 'images')
        lbl_dest = os.path.join(DEST_DIR, split, 'labels')
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        for filename in files:
            name = os.path.splitext(filename)[0]
            label_filename = name + '.txt'
            
            shutil.copy(
                os.path.join(SOURCE_IMAGES_DIR, filename), 
                os.path.join(img_dest, filename)
            )

            shutil.copy(
                os.path.join(SOURCE_LABELS_DIR, label_filename), 
                os.path.join(lbl_dest, label_filename)
            )

    print(f"\nSUCCESS! Dataset created at:\n{DEST_DIR}")
    print("Please copy this path into your data.yaml file.")

if __name__ == '__main__':
    split_dataset()