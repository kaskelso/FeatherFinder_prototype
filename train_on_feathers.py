#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:55:13 2025

@author: kennyaskelson
"""


import os
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

# Set training dataset
cfg.DATASETS.TRAIN = ("feather_train",)
# Set validation dataset
cfg.DATASETS.TEST = ("feather_val",)

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2
# Learning rate
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 300  
# This gives a list of steps where the learning rate should decrease
cfg.SOLVER.STEPS = []

# parameter that is used to sample a subset of proposals coming out of RPN to calculate cls and reg loss during training
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

# ONLY one class: "feather"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

cfg.MODEL.DEVICE = "cpu"  # use CPU, or "cuda" if you have GPU

cfg.OUTPUT_DIR = "./feather_output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()








