# Object Detection and Segmentation

## Classification vs Detection vs Segmentation

| Task | Output | Example |
|---|---|---|
| Classification | class label | "this is a cat" |
| Object Detection | class + bounding box per object | "cat at [x1,y1,x2,y2]" |
| Semantic Segmentation | class label per pixel | full pixel mask |

## Bounding Boxes

Format: `[x1, y1, x2, y2]` — top-left and bottom-right pixel coordinates, origin at top-left.

YOLO format (in label files): `class_id cx cy w h` — all normalized to [0, 1] relative to image size.

## IoU (Intersection over Union)

Measures overlap between a predicted box and the ground truth:

```
IoU = intersection area / union area
```

- IoU = 1.0: perfect overlap
- IoU = 0.0: no overlap
- Detection is typically considered "correct" at IoU ≥ 0.5

**Manual calculation:**
```python
x1 = max(gt[0], pred[0]);  y1 = max(gt[1], pred[1])
x2 = min(gt[2], pred[2]);  y2 = min(gt[3], pred[3])
intersection = max(0, x2-x1) * max(0, y2-y1)
union = gt_area + pred_area - intersection
iou = intersection / union
```

## YOLO (You Only Look Once)

Single-pass detector — predicts boxes, class probabilities, and confidence in one forward pass. Much faster than two-stage detectors (R-CNN family).

**YOLOv8** is used in the course. The nano variant (`yolov8n.pt`, ~6 MB) runs on CPU.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.predict(source="image.jpg", conf=0.35, iou=0.5)

boxes = results[0].boxes.xyxy   # [x1, y1, x2, y2]
confs = results[0].boxes.conf   # confidence scores
classes = results[0].boxes.cls  # class IDs
```

Key parameters:
- `conf`: minimum confidence threshold to keep a detection
- `iou`: IoU threshold for NMS (non-maximum suppression, removes duplicate boxes)
- `imgsz`: input resolution (larger = slower but more accurate on small objects)

## Semantic Segmentation

YOLOv8-seg adds a mask prediction head alongside detection. Returns a binary mask for each detected object.

```python
model = YOLO("yolov8n-seg.pt")
results = model("image.jpg", conf=0.35)

masks = results[0].masks.data    # [N, H, W] binary masks
```

Masks are low-resolution — upsample to original image size with bilinear interpolation before overlaying:

```python
mask_resized = F.interpolate(
    mask.unsqueeze(0).unsqueeze(0),
    size=image.shape[1:],
    mode="bilinear"
).squeeze()
```
