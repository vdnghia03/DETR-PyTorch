import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from model.detr import DETR
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Chọn thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

def collate_function(data):
    return tuple(zip(*data))

def train(args):
    # --- 1. CONFIG LOADING ---
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']

    # Thiết lập Seed để tái lập kết quả
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- 2. DATASET ---
    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])
    
    train_dataset = DataLoader(voc,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)

    # --- 3. MODEL SETUP ---
    model = DETR(
        config=model_config,
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx']
    )
    model.to(device)
    model.train()

    # Load checkpoint nếu có (Resume Training)
    ckpt_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print('Loading checkpoint as one exists')

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # --- 4. OPTIMIZER TỐI ƯU (BÍ MẬT SỐ 1) ---
    # Tách tham số: Backbone học chậm (0.1x), Transformer học thường (1x)
    backbone_params = []
    transformer_params = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params.append(p)
        else:
            transformer_params.append(p)

    param_dicts = [
        {'params': backbone_params, 'lr': train_config['lr'] * 0.1}, # Backbone LR nhỏ
        {'params': transformer_params, 'lr': train_config['lr']}     # Transformer LR chuẩn
    ]

    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

    lr_scheduler = MultiStepLR(optimizer,
                               milestones=train_config['lr_steps'],
                               gamma=0.1)

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    
    # --- 5. TRAINING LOOP ---
    print("Start Training...")
    for i in range(num_epochs):
        detr_classification_losses = []
        detr_localization_losses = []
        
        # Reset Gradient đầu epoch
        optimizer.zero_grad() 

        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset)):
            # A. Chuẩn bị dữ liệu
            for target in targets:
                target['boxes'] = target['boxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
            images = torch.stack([im.float().to(device) for im in ims], dim=0)

            # B. Forward Pass & Tính Loss
            # Model DETR tự chạy Hungarian Matching bên trong khi có targets
            output = model(images, targets)
            batch_losses = output['loss']

            # Tổng hợp loss từ Deep Supervision (cộng cả 4 lớp)
            loss_cls = sum(batch_losses['classification'])
            loss_bbox = sum(batch_losses['bbox_regression'])
            total_loss = loss_cls + loss_bbox

            # C. Gradient Accumulation (Chia nhỏ loss)
            loss = total_loss / acc_steps
            loss.backward()

            # Logging
            detr_classification_losses.append(loss_cls.item())
            detr_localization_losses.append(loss_bbox.item())

            # D. Bước Update (Chỉ chạy khi tích lũy đủ bước)
            if (idx + 1) % acc_steps == 0:
                # --- TỐI ƯU 2: GRADIENT CLIPPING (BÍ MẬT SỐ 2) ---
                # Cắt ngọn đạo hàm để tránh nổ loss (NaN)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Print log mỗi N bước
            if steps % train_config['log_steps'] == 0:
                current_lr = optimizer.param_groups[1]['lr'] # Lấy LR của Transformer
                print(f' Step {steps}: Cls Loss: {np.mean(detr_classification_losses):.4f} | '
                      f'Loc Loss: {np.mean(detr_localization_losses):.4f} | LR: {current_lr:.1e}')
            
            if torch.isnan(total_loss):
                print('Error: Loss is becoming NaN. Exiting.')
                exit(0)
            
            steps += 1

        # --- TỐI ƯU 3: XỬ LÝ SỐ DƯ (Leftover Batches) ---
        # Nếu số batch không chia hết cho acc_steps, update nốt phần còn dư
        if len(train_dataset) % acc_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            optimizer.zero_grad()

        # E. Cập nhật Scheduler cuối epoch
        lr_scheduler.step()
        
        print(f'Finished Epoch {i+1}')
        print(f'Avg Epoch Loss -> Cls: {np.mean(detr_classification_losses):.4f} | '
              f'Loc: {np.mean(detr_localization_losses):.4f}')
        
        # Save Checkpoint
        torch.save(model.state_dict(), ckpt_path)

    print('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for detr training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)
