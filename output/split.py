import os
import shutil
import random
from sklearn.model_selection import train_test_split
 
# Path to the dataset folder
dataset_dir = os.path.dirname(os.path.abspath(__file__))
categories = ["rock", "paper", "scssisors",]
 
# Output folders
output_dirs = {
    "train": "output/train/",
    "test": "output/test/",
    "val": "output/val/"
}
 
# Allowed image extensions
valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
 
# Max images per category
max_images = 200
 
# Create output directories
for output_dir in output_dirs.values():
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)
 
# Split images
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    images = []
 
    # Walk through all subfolders inside this category
    for root, _, files in os.walk(category_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_exts:
                images.append(os.path.join(root, f))
 
    # Shuffle and take only up to 200 images
    random.shuffle(images)
    images = images[:max_images]
 
    if len(images) < 30:
        print(f"⚠ Warning: Category '{category}' has only {len(images)} images. Splitting anyway.")
 
    # First split → Train (70%) and Temp (30%)
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
 
    # Second split → from Temp → Validate (10%) and Test (20%)
    validate, test = train_test_split(temp, test_size=0.67, random_state=42)
 
    # Copy images into their respective folders
    for img in train:
        shutil.copy(img, os.path.join(output_dirs["train"], category))
 
    for img in validate:
        shutil.copy(img, os.path.join(output_dirs["val"], category))
 
    for img in test:
        shutil.copy(img, os.path.join(output_dirs["test"], category))
 
    print(f"✅ {category}: {len(train)} train | {len(validate)} val | {len(test)} test")
 
print("\n🎉 Dataset split completed with max 200 images per category.")
