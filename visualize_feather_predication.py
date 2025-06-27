#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:35:40 2025

@author: kennyaskelson
"""

import cv2
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

# This tells detectron2 where the datasets are (name, extra metadata, .json, data folder paths). 
register_coco_instances("feather_train", {}, "feather_images/feather_train/train.json", "feather_images/feather_train")
register_coco_instances("feather_val", {}, "feather_images/feather_val/val.json", "feather_images/feather_val")

# Creates a new configuration with all the model & training settings
cfg = get_cfg()
#Important! This brings in the pretrained model as our starting point
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# ONLY one class: "feather"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  
# use CPU, I'm on mac so GPU not supported
cfg.MODEL.DEVICE = "cpu"  

cfg.OUTPUT_DIR = "./feather_output"
# Use the trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  
# This says only use predictions with this confidence or higher
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
#load model for prediction
predictor = DefaultPredictor(cfg)

# Load an image
image_path = "NOGO_tail_ad.jpg" 
im = cv2.imread(image_path)

# Make prediction
outputs = predictor(im)

# Visualize, im[:, :, ::-1] converts from BGR (OpenCV) to RGB (for visualization). And uses class metadata from feather_val
v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("feather_val"))
# Draws boxes on image
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

output_img = v.get_image()

plt.figure(figsize=(12,8))
plt.imshow(output_img)
plt.axis('off')
plt.show()



### The following is used to automatically grab feathers in photos and crop them out into 
### their own photo

import cv2
import os
import glob

# Set your threshold (e.g., 0.7 for 70%)
score_threshold = 0.8

folder_path = '.'
pattern = os.path.join(folder_path, '*test.jpg*')  # matches anything with the string
file_list = glob.glob(pattern)

for i in range(len(file_list)):
    im = cv2.imread(glob.glob(pattern)[i])
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    scores = instances.scores if instances.has("scores") else None
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    masks = instances.pred_masks if instances.has("pred_masks") else None

    os.makedirs("feather_crops_filtered", exist_ok=True)
    
    name = glob.glob(pattern)[i]
    
    short_name = name.replace('./', '').replace('.jpg', '')

    for i in range(len(instances)):
        if scores[i] >= score_threshold:
            box = boxes[i].tensor.numpy().astype(int)[0]
            x1, y1, x2, y2 = box
            crop = im[y1:y2, x1:x2]
    
            if masks is not None:
                mask = masks[i].numpy()
                mask_crop = mask[y1:y2, x1:x2]
                crop[mask_crop == 0] = 255  # Make background white
    
            cv2.imwrite(f"feather_crops_filtered/{short_name}_{i}_score_{scores[i]:.2f}.png", crop)
            print(f"{short_name}_{i} with score {scores[i]:.2f}")