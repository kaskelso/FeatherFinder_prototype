#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:53:29 2025

@author: kennyaskelson
"""

import json
import os
from labelme import utils
import json
import os
import random
from shutil import copy2

def labelme_to_coco(labelme_folder, output_json_path):
    # Set empty objects for loop
    images = []
    annotations = []
    categories = [{"id": 0, "name": "feather"}]
    ann_id = 0
    img_id = 0

    for filename in os.listdir(labelme_folder):
        # Skips files that are not .json
        if not filename.endswith(".json"):
            continue
        # Gets full file path of .json and opens
        labelme_path = os.path.join(labelme_folder, filename)
        with open(labelme_path) as f:
            data = json.load(f)
        # Assigns image information
        img = {
            "id": img_id,
            "file_name": data["imagePath"],
            "height": data["imageHeight"],
            "width": data["imageWidth"]
        }
        images.append(img)
        # loops over shapes in .json
        for shape in data["shapes"]:
            # grabs points of polygon
            points = shape["points"]
            # Flatten polygon points for COCO segmentation LabelMe has [[x1, y1], [x2, y2] COCO wants [x1, y1, x2, y2]
            segmentation = []
            for point in points:
                for coord in point:
                    segmentation.append(coord)
            # Compute bbox from points
            # Grab x values
            xs = [p[0] for p in points]
            # Grab y values
            ys = [p[1] for p in points]
            # Find minimums
            x_min, y_min = min(xs), min(ys)
            # Get width and height
            width, height = max(xs)-x_min, max(ys)-y_min
            # Assign box
            bbox = [x_min, y_min, width, height]
            
            # assign annotation values
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "segmentation": [segmentation],
                "bbox": bbox,
                "iscrowd": 0 # this means it is not overlapping
            }
            annotations.append(ann)
            ann_id += 1 # get ready for next one

        img_id += 1 # get ready for next one
    # Build final dictionary
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f)

# Usage example:
labelme_to_coco("feather_images", "feather_dataset_coco.json")


# assign files and directories
input_json = "feather_images/feather_dataset_coco.json"
images_dir = "feather_images"
train_dir = "feather_images/feather_train"
val_dir = "feather_images/feather_val"
train_json = os.path.join(train_dir, "train.json")
val_json = os.path.join(val_dir, "val.json")
# 80% training, 20% validation
train_ratio = 0.8  

# Create output folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Load json
with open(input_json) as f:
    coco_data = json.load(f)
# Get list of images, annotation, and categories
images = coco_data["images"]
annotations = coco_data["annotations"]
categories = coco_data["categories"]

# Shuffle and split images
random.shuffle(images)
train_cutoff = int(len(images) * train_ratio)
train_images = images[:train_cutoff]
val_images = images[train_cutoff:]

# Index images by ID
train_image_ids = {img["id"] for img in train_images}
val_image_ids = {img["id"] for img in val_images}

# Split annotations by image_id
train_annotations = []
for ann in annotations: 
    if ann["image_id"] in train_image_ids:
        train_annotations.append(ann)
        
val_annotations = [] 
for ann in annotations: 
    if ann["image_id"] in val_image_ids:
        val_annotations.append(ann)

# Copy images into train/val folders
for img in train_images:
    copy2(os.path.join(images_dir, img["file_name"]), os.path.join(train_dir, img["file_name"]))
for img in val_images:
    copy2(os.path.join(images_dir, img["file_name"]), os.path.join(val_dir, img["file_name"]))

# Save split COCO files
train_dict = {"images": train_images, "annotations": train_annotations, "categories": categories}
val_dict = {"images": val_images, "annotations": val_annotations, "categories": categories}

# Makes json files
with open(train_json, "w") as f:
    json.dump(train_dict, f)

with open(val_json, "w") as f:
    json.dump(val_dict, f)





