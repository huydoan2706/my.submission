import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ObjectDetectionDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=416, is_train=True):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        self.classes = data['classes']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.img_ids = list(self.images.keys())

        # Tang cuong du lieu
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        # Dung basename de tuong thich voi cau truc thu muc khac nhau
        img_path = os.path.join(self.img_dir, os.path.basename(img_info['file_name']))

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        bboxes = []
        labels = []
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                xmin, ymin, xmax, ymax = ann['bbox']
                # Clip de tranh loi
                xmin = max(0, min(xmin, w0-1))
                ymin = max(0, min(ymin, h0-1))
                xmax = max(xmin+1, min(xmax, w0))
                ymax = max(ymin+1, min(ymax, h0))
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[ann['class']])

        if len(bboxes) == 0:
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        image = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['labels']

        # Chuyen bbox ve dang [cx, cy, w, h] chuan hoa [0, 1]
        targets = []
        for box, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = box
            cx = ((xmin + xmax) / 2) / self.img_size
            cy = ((ymin + ymax) / 2) / self.img_size
            w = (xmax - xmin) / self.img_size
            h = (ymax - ymin) / self.img_size
            targets.append([cx, cy, w, h, label])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))

        return image, targets, img_id, (w0, h0)  # tra ve kich thuoc goc de inference

