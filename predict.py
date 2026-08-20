import argparse
import json
import os
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from utils.DetectionLoss import DetectionLoss
from utils.ObjectDetectionDataset import ObjectDetectionDataset
from utils.SimpleDetector import SimpleDetector
from tqdm import tqdm
import numpy as np
import cv2


def non_max_suppression(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_others + area_i - inter + 1e-7)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def predict_one(model, image_path, classes, img_size, conf_thres=0.3, iou_thres=0.45,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    image = cv2.imread(image_path)
    if image is None:
        return []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h0, w0 = image.shape[:2]

    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)[0].cpu()

    S = pred.shape[0]
    boxes, scores, labels = [], [], []

    for j in range(S):
        for i in range(S):
            obj = torch.sigmoid(pred[j, i, 0]).item()
            if obj < conf_thres:
                continue
            cls_scores = torch.sigmoid(pred[j, i, 5:])
            cls_id = int(cls_scores.argmax().item())
            conf = obj * cls_scores[cls_id].item()
            if conf < conf_thres:
                continue

            cx, cy, w, h = torch.sigmoid(pred[j, i, 1:5]).tolist()
            # Dua toa do ve anh goc
            cx *= w0
            cy *= h0
            w *= w0
            h *= h0

            xmin = max(0, cx - w/2)
            ymin = max(0, cy - h/2)
            xmax = min(w0, cx + w/2)
            ymax = min(h0, cy + h/2)

            # Bỏ qua các khung rỗng (empty box)
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(conf)
            labels.append(cls_id)

    if not boxes:
        return []

    # NMS theo tung lop
    final = []
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        keep = non_max_suppression(boxes[idxs], scores[idxs], iou_threshold=iou_thres)
        for k in keep:
            final.append({
                "class": classes[c],
                "bbox": [
                    float(boxes[idxs][k][0]), float(boxes[idxs][k][1]),
                    float(boxes[idxs][k][2]), float(boxes[idxs][k][3])
                ],
                "score": float(scores[idxs][k])
            })
    return final


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load checkpoint
    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(args.checkpoint_dir, 'best.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.checkpoint_dir, 'last.pth')
    print(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)

    classes = checkpoint['classes']
    img_size = checkpoint.get('img_size', 416)
    grid_size = checkpoint.get('grid_size', 13)
    num_classes = checkpoint['num_classes']

    model = SimpleDetector(num_classes=num_classes, grid_size=grid_size, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Lay danh sach anh
    image_dir = Path(args.image_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in extensions])

    predictions = []
    for img_path in tqdm(image_files, desc='Predicting'):
        dets = predict_one(model, str(img_path), classes, img_size=img_size,
                           conf_thres=args.conf_thres, iou_thres=args.iou_thres)

        # Tạo mảng chứa các "đối tượng" bounding box
        boxes = []
        for d in dets:
            boxes.append({
                "bbox": d['bbox'],
                "confidence": d['score'],
                "class": d['class']
            })

        # Gán mảng các đối tượng đó vào key "boxes" của ảnh
        predictions.append(
            {
                "image_id": img_path.name,
                "boxes": boxes
            }
        )

    # Lưu kết quả
    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Thư mục chứa ảnh cần dự đoán")
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--checkpoint_dir", type=str, default="./models")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn cụ thể tới file .pth (ưu tiên hơn checkpoint_dir)")
    parser.add_argument("--conf_thres", type=float, default=0.3)
    parser.add_argument("--iou_thres", type=float, default=0.45)
    args = parser.parse_args()

    main(args)
