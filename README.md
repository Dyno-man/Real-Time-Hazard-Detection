# Real-Time Hazard Detection

This repo contains machine vision experiments for detecting sidewalk and road hazards in images and video. It focuses on object-detection models that can identify hazards such as potholes, open manholes, traffic cones, poles, signs, trash cans, roadblocks, tactile paving, surface damage, and other sidewalk obstacles.

## Contents

- `Yolo/` - YOLO12n training and video testing scripts.
- `F_RCNN/` - Faster R-CNN notebook experiments.
- `RTM-Det/` - RTMDet Colab notebooks plus a dataset creation script.

## Dataset

The training scripts expect a COCO-style dataset shaped like:

```text
sidewalk_hazard_dataset/
  annotations/
    train.json
    val.json
  images/
    train/
    val/
  class_map.json
```

`RTM-Det/colabscripts/CreateDatasets.py` builds this structure by combining VOC-style source datasets and mapping labels into the shared sidewalk hazard class list.

## YOLO Usage

Prepare YOLO labels and train:

```bash
python Yolo/train.py --dataset-root sidewalk_hazard_dataset
```

Only prepare labels/manifests:

```bash
python Yolo/train.py --dataset-root sidewalk_hazard_dataset --prepare-only
```

Run detection and tracking on a video:

```bash
python Yolo/test_video.py path/to/video.mp4 --model best.pt
```

## Notes

Large datasets, trained weights, and generated run outputs are not included here. The notebooks are intended for experimentation in Colab or a local Python environment with the required detection libraries installed.
