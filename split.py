import os
import random
import shutil

dataset_path = "VOCdevkit/VOC2012/JPEGImages"
train_path = "VOCdevkit/VOC2012/train"
test_path = "VOCdevkit/VOC2012/test"

# Create train/test directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# List all image files
images = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]
random.shuffle(images)

# Split 80-20
split_idx = int(len(images) * 0.8)
train_images, test_images = images[:split_idx], images[split_idx:]

# Move files
for img in train_images:
    shutil.move(os.path.join(dataset_path, img), os.path.join(train_path, img))

for img in test_images:
    shutil.move(os.path.join(dataset_path, img), os.path.join(test_path, img))

print(f"Train: {len(train_images)}, Test: {len(test_images)}")
