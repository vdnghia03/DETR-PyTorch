
import torch
import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random

class Food67Dataset(Dataset):
    def __init__(self, split, im_sets, im_size=640):
        self.split = split
        self.im_size = im_size
        self.images = []
        
        # 1. Load danh sách ảnh
        for folder_path in im_sets:
            if not os.path.exists(folder_path):
                print(f"Warning: Path {folder_path} does not exist!")
                continue
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                self.images.extend(glob.glob(os.path.join(folder_path, ext)))
        self.images = sorted(self.images)

        # 2. Định nghĩa Class (69 class)
        self.idx2label = [
            "__background__", 
            "Banh canh", "Banh chung", "Banh cuon", "Banh khot", "Banh mi", "Banh trang",
            "Banh trang tron", "Banh xeo", "Bo kho", "Bo la lot", "Bong cai", "Bun",
            "Bun bo Hue", "Bun cha", "Bun dau", "Bun mam", "Bun rieu", "Ca", "Ca chua",
            "Ca phao", "Ca rot", "Canh", "Cha", "Cha gio", "Chanh", "Com", "Com tam",
            "Con nguoi", "Cu kieu", "Cua", "Dau hu", "Dua chua", "Dua leo",
            "Goi cuon", "Hamburger", "Heo quay", "Hu tieu", "Kho qua thit", "Khoai tay chien",
            "Lau", "Long heo", "Mi", "Muc", "Nam", "Oc", "Ot chuong", "Pho", "Pho mai",
            "Rau", "Salad", "Thit bo", "Thit ga", "Thit heo", "Thit kho", "Thit nuong",
            "Tom", "Trung", "Xoi", "Banh beo", "Cao lau", "Mi Quang",
            "Com chien Duong Chau", "Bun cha ca", "Com chien ga", "Chao long",
            "Nom hoa chuoi", "Nui xao bo", "Sup cua"
        ]
        
        # Transform ảnh
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)
    def load_yolo_label_as_xyxy(self, img_path):
        """
        Đọc YOLO (cx, cy, w, h) -> Chuyển thành (x1, y1, x2, y2) Normalized
        """
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    labels.append(cls_id + 1) # Class + 1
                    
                    # 1. Đọc YOLO (Normalized Center)
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # 2. Chuyển sang XYXY Normalized
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    # 3. Clip an toàn về 0-1
                    x1 = np.clip(x1, 0, 1)
                    y1 = np.clip(y1, 0, 1)
                    x2 = np.clip(x2, 0, 1)
                    y2 = np.clip(y2, 0, 1)
                    boxes.append([x1, y1, x2, y2])
                    
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # 1. Đọc ảnh
        im = cv2.imread(img_path)
        if im is None:
            im = np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)
            boxes, labels = torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            # Load nhãn dạng XYXY
            boxes, labels = self.load_yolo_label_as_xyxy(img_path)

            # Augmentation Flip (Nếu lật ảnh thì lật box x1, x2)
            if self.split == 'train' and random.random() > 0.5:
                im = cv2.flip(im, 1) # Flip ảnh
                if len(boxes) > 0:
                    # Flip tọa độ X: x1_new = 1.0 - x2_old; x2_new = 1.0 - x1_old
                    old_x1 = boxes[:, 0].clone()
                    old_x2 = boxes[:, 2].clone()
                    boxes[:, 0] = 1.0 - old_x2
                    boxes[:, 2] = 1.0 - old_x1
           # Resize ảnh (Box normalized giữ nguyên tỉ lệ)
            im = cv2.resize(im, (self.im_size, self.im_size))
        
        # 2. Chuẩn hóa ảnh
        im_tensor = self.normalize(im)
        
        # 3. Lọc box rác
        if len(boxes) > 0:
            # Tính width, height của box để lọc
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep = (ws > 0.01) & (hs > 0.01)
            boxes = boxes[keep]
            labels = labels[keep]

        # 4. Đóng gói target
        target = {
            'boxes': boxes, # Dạng x1, y1, x2, y2 (Normalized)
            'labels': labels,
            'orig_size': torch.tensor([self.im_size, self.im_size]),
            'size': torch.tensor([self.im_size, self.im_size]),
            'image_id': torch.tensor([idx]),
            'difficult': torch.zeros_like(labels)
        }
        return im_tensor, target, img_path
