
import torch
import argparse
import os
import numpy as np
import yaml
import random
import time  # <--- Import Time
from tqdm import tqdm
from model.detr import DETR
from dataset.food67 import Food67Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_function(data):
    return tuple(zip(*data))

def train(args):
    # --- [NEW LOGIC] 1. BẮT ĐẦU TÍNH GIỜ ---
    start_time = time.time()
    # Kaggle limit 12h = 43200s. Dừng ở 11.5h (41400s) để an toàn
    TIME_LIMIT = 11.5 * 3600 
    print(f"Training started. Will auto-stop after {TIME_LIMIT/3600:.1f} hours.")

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset_raw = Food67Dataset(
        split='train',
        im_sets=dataset_config['train_im_sets'],
        im_size=dataset_config['im_size']
    )
    
    train_dataset = DataLoader(train_dataset_raw,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)

    model = DETR(
        config=model_config,
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx']
    )
    model.to(device)
    model.train()

    # Tạo folder output (nếu chưa có)
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # --- [NEW LOGIC] 2. SMART CHECKPOINT LOADING ---
    # Đường dẫn file trong thư mục đang chạy (Working Dir)
    ckpt_path_working = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    
    # Đường dẫn file từ Input Dataset (Resume Path từ config)
    ckpt_path_input = train_config.get('resume_path', None)

    start_epoch = 0

    # ƯU TIÊN 1: Load từ Working Dir (Mới nhất trong session hiện tại)
    if os.path.exists(ckpt_path_working):
        print(f"🔄 Resuming from WORKING directory: {ckpt_path_working}")
        state_dict = torch.load(ckpt_path_working, map_location=device)
        model.load_state_dict(state_dict)
    
    # ƯU TIÊN 2: Load từ Input Dataset (Model cũ upload lên)
    elif ckpt_path_input and os.path.exists(ckpt_path_input):
        print(f"🔄 Resuming from INPUT DATASET: {ckpt_path_input}")
        state_dict = torch.load(ckpt_path_input, map_location=device)
        model.load_state_dict(state_dict)
    
    # ƯU TIÊN 3: Train từ đầu
    else:
        print("✨ No checkpoint found. Training from scratch (Pretrained ResNet).")

    # Optimizer setup (Tách LR)
    backbone_params = []
    transformer_params = []
    for n, p in model.named_parameters():
        if "backbone" in n and p.requires_grad:
            backbone_params.append(p)
        elif p.requires_grad:
            transformer_params.append(p)

    param_dicts = [
        {"params": backbone_params, "lr": train_config['lr'] * 0.1},
        {"params": transformer_params, "lr": train_config['lr']},
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    
    print("Start Training Loop...")
    for i in range(num_epochs):
        loss_cls_list = []
        loss_loc_list = []
        
        optimizer.zero_grad()
        
        # Dùng tqdm để hiện progress bar
        pbar = tqdm(train_dataset, desc=f"Epoch {i+1}/{num_epochs}")
        for idx, (ims, targets, _) in enumerate(pbar):
            for target in targets:
                target['boxes'] = target['boxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
            
            images = torch.stack([im.float().to(device) for im in ims], dim=0)

            output = model(images, targets)
            batch_losses = output['loss']
            
            loss_cls = sum(batch_losses['classification'])
            loss_bbox = sum(batch_losses['bbox_regression'])
            total_loss = loss_cls + loss_bbox
            
            loss = total_loss / acc_steps
            loss.backward()
            
            loss_cls_list.append(loss_cls.item())
            loss_loc_list.append(loss_bbox.item())

            if (idx + 1) % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar text
            if steps % train_config['log_steps'] == 0:
                pbar.set_postfix({
                    'Cls': f"{np.mean(loss_cls_list[-10:]):.4f}",
                    'Loc': f"{np.mean(loss_loc_list[-10:]):.4f}"
                })
            
            steps += 1

        if len(train_dataset) % acc_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()
        print(f"Finished Epoch {i+1}. Saving model...")
        
        # Save Model vào Working Dir
        torch.save(model.state_dict(), ckpt_path_working)
        
        # --- [NEW LOGIC] 3. KIỂM TRA THỜI GIAN ---
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT:
            print(f"⚠️ Time limit reached ({elapsed_time/3600:.2f} hours). Stopping training safely.")
            print("Model has been saved. Please download output or commit dataset.")
            break # Thoát vòng lặp Epoch -> Kết thúc train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/food67.yaml', type=str)
    args = parser.parse_args()
    train(args)
