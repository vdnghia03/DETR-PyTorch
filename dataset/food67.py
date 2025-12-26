import torch
import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
# Import v2 và tv_tensors để xử lý Box
import torchvision.transforms.v2 as T
from torchvision import tv_tensors 

class Food67Dataset(Dataset):
    def __init__(self, split, im_sets, im_size=640):
        self.split = split
        self.im_size = im_size
        self.images = []
        
        # 1. Load ảnh (Giữ nguyên logic cũ)
        for folder_path in im_sets:
            if not os.path.exists(folder_path):
                print(f"Warning: Path {folder_path} does not exist!")
                continue
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
                self.images.extend(glob.glob(os.path.join(folder_path, ext)))
        self.images = sorted(self.images)
        
        # 2. Định nghĩa Class (Giữ nguyên)
        self.idx2label = [
            "__background__", 
            "Bánh canh", "Bánh chưng", "Bánh cuốn", "Bánh khọt", "Bánh mì", "Bánh tráng",
            "Bánh tráng trộn", "Bánh xèo", "Bò kho", "Bò lá lốt", "Bông cải", "Bún",
            "Bún bò Huế", "Bún chả", "Bún đậu", "Bún mắm", "Bún riêu", "Cá", "Cà chua",
            "Cà pháo", "Cà rốt", "Canh", "Chả", "Chả giò", "Chanh", "Cơm", "Cơm tấm",
            "Con người", "Củ kiệu", "Cua", "Đậu hũ", "Dưa chua", "Dưa leo",
            "Gỏi cuốn", "Hamburger", "Heo quay", "Hủ tiếu", "Khổ qua thịt", "Khoai tây chiên",
            "Lẩu", "Lòng heo", "Mì", "Mực", "Nấm", "Ốc", "Ớt chuông", "Phở", "Phô mai",
            "Rau", "Salad", "Thịt bò", "Thịt gà", "Thịt heo", "Thịt kho", "Thịt nướng",
            "Tôm", "Trứng", "Xôi", "Bánh bèo", "Cao lầu", "Mì Quảng",
            "Cơm chiên Dương Châu", "Bún chả cá", "Cơm chiên gà", "Cháo lòng",
            "Nộm hoa chuối", "Nui xào bò", "Súp cua"
        ]
        self.label2idx = {label: idx for idx, label in enumerate(self.idx2label)}

        # 3. Cấu hình Transforms (Đoạn bạn thích đây!)
        # Mean/Std chuẩn của ImageNet
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        # Mean để fill màu khi zoom out (màu xám trung tính)
        self.im_mean = [123.0, 117.0, 104.0] 

        self.transforms = {
            'train': T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                # T.RandomZoomOut(fill=self.im_mean), # Có thể bật lại nếu muốn
                # T.RandomIoUCrop(),                 # Có thể bật lại nếu muốn augmentation mạnh
                T.RandomPhotometricDistort(),        # Chỉnh màu sắc, độ sáng
                T.Resize(size=(self.im_size, self.im_size)),
                # Hàm sanitize giúp lọc bỏ các box bị lỗi sau khi crop/resize
                T.SanitizeBoundingBoxes(), 
                T.ToPureTensor(),
                T.ToDtype(torch.float32, scale=True), # Chia 255
                T.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
            ]),
            'test': T.Compose([
                T.Resize(size=(self.im_size, self.im_size)),
                T.ToPureTensor(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
            ]),
        }
        
        # Mapping split tên 'valid' sang key 'test' trong transforms
        self.transform_key = 'train' if split == 'train' else 'test'

        print(f"Loaded {len(self.images)} images for split '{split}'")

    def __len__(self):
        return len(self.images)

    def load_yolo_label(self, img_path, img_h, img_w):
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
                    labels.append(cls_id + 1) # Class + 1 vì Background=0
                    
                    # YOLO: cx, cy, w, h (normalized)
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Chuyển sang Absolute XYXY (x1, y1, x2, y2)
                    # Lý do: Transform v2 làm việc tốt nhất với XYXY tuyệt đối
                    x_c = cx * img_w
                    y_c = cy * img_h
                    w_abs = w * img_w
                    h_abs = h * img_h
                    
                    x1 = x_c - (w_abs / 2)
                    y1 = y_c - (h_abs / 2)
                    x2 = x_c + (w_abs / 2)
                    y2 = y_c + (h_abs / 2)
                    
                    boxes.append([x1, y1, x2, y2])
                    
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # 1. Đọc ảnh bằng TVTensors (giúp PyTorch hiểu đây là Ảnh)
        im = cv2.imread(img_path)
        if im is None:
            im = np.zeros((640, 640, 3), dtype=np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        
        # Chuyển ảnh sang TVTensor Image
        # Shape: (C, H, W)
        im_tv = tv_tensors.Image(torch.from_numpy(im).permute(2, 0, 1))

        # 2. Đọc Label & Wrap vào BoundingBoxes
        boxes, labels = self.load_yolo_label(img_path, h, w)
        
        # Quan trọng: Phải bọc boxes vào tv_tensors.BoundingBoxes
        # để transforms.v2 biết đây là hộp cần được resize/flip theo ảnh
        if len(boxes) > 0:
            boxes_tv = tv_tensors.BoundingBoxes(
                boxes, 
                format="XYXY", 
                canvas_size=(h, w)
            )
        else:
            # Nếu không có box nào, tạo dummy rỗng
            boxes_tv = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)), 
                format="XYXY", 
                canvas_size=(h, w)
            )

        # 3. Tạo dictionary target
        target = {
            "boxes": boxes_tv,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            # Thêm dummy fields để transform không lỗi (nếu dùng Sanitize nâng cao)
            "difficult": torch.zeros_like(labels) 
        }

        # 4. ÁP DỤNG TRANSFORMS (Magic happens here!)
        # Transform v2 nhận vào (Image, Target Dict) và trả về (Image, Target Dict) đã biến đổi
        im_transformed, target_transformed = self.transforms[self.transform_key](im_tv, target)

        # 5. Chuyển đổi ngược lại format cho DETR
        # DETR cần box dạng: (cx, cy, w, h) chuẩn hóa [0-1]
        
        final_boxes = target_transformed['boxes'] # Đang là XYXY Absolute sau khi resize
        final_labels = target_transformed['labels']
        
        if len(final_boxes) > 0:
            # Lấy kích thước ảnh sau khi resize (thường là 640x640)
            # im_transformed shape là (C, H, W)
            new_h, new_w = im_transformed.shape[-2:]
            
            # Convert XYXY -> CXCYWH
            # boxes là Tensor, ta dùng phép toán tensor
            x1 = final_boxes[:, 0]
            y1 = final_boxes[:, 1]
            x2 = final_boxes[:, 2]
            y2 = final_boxes[:, 3]
            
            w_box = x2 - x1
            h_box = y2 - y1
            cx = x1 + w_box / 2
            cy = y1 + h_box / 2
            
            # Chuẩn hóa về 0-1
            final_boxes_norm = torch.stack([
                cx / new_w,
                cy / new_h,
                w_box / new_w,
                h_box / new_h
            ], dim=1)
            
            # Clip về [0, 1] để an toàn
            final_boxes_norm = torch.clamp(final_boxes_norm, 0, 1)
        else:
            final_boxes_norm = torch.zeros((0, 4), dtype=torch.float32)

        # 6. Đóng gói kết quả cuối cùng
        final_target = {
            'boxes': final_boxes_norm,
            'labels': final_labels,
            'orig_size': torch.tensor([h, w]),
            'size': torch.tensor([self.im_size, self.im_size]),
            'image_id': torch.tensor([idx]),
            'difficult': target['difficult']
        }

        return im_transformed, final_target, img_path
