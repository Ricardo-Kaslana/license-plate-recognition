import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime
from torchvision import transforms

# 导入自定义的模块
from model import LPR_Transformer
from dataset import CCPD_LPR_Dataset
from utils import (
    VOCAB_SIZE, 
    SPECIAL_TOKENS_DICT, 
    idx_to_char,
    CHARSET
)

# --- 0. 环境配置 (字体) ---
# 清除字体缓存
font_cache = os.path.join(os.path.expanduser('~'), '.cache', 'matplotlib')
os.system(f'rm -rf {font_cache}')

# 手动指定中文字体[ubuntu下]（系统中已安装的文泉驿字体 绝对路径）
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

# 注册字体
try:
    # 方法一：适用于 matplotlib 较新版本
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
except:  # noqa: E722
    # 方法二：兼容 matplotlib 旧版本
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

# 确保unicode支持
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 超参数和配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CCPD_PATH = r"../CCPD2019/ccpd_base"
SPLIT_RATIO = (0.8, 0.1, 0.1)
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
D_MODEL = 512
NHEAD = 8
NUM_DECODER_LAYERS = 6

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./LPR_Rec/training_run_{TIMESTAMP}"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.jsonl")
REPORT_FILE = os.path.join(OUTPUT_DIR, "final_report.txt")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
print(f"所有输出将被保存到: {os.path.abspath(OUTPUT_DIR)}")

# --- 2. 数据准备 ---
def prepare_dataloaders(ccpd_root, batch_size, split_ratio):
    all_files = [f for f in os.listdir(ccpd_root) if f.lower().endswith('.jpg')]
    random.shuffle(all_files)
    train_count = int(len(all_files) * split_ratio[0])
    val_count = int(len(all_files) * split_ratio[1])
    train_files = all_files[:train_count]
    val_files = all_files[train_count : train_count + val_count]
    test_files = all_files[train_count + val_count :]
    print(f"数据集划分: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    train_dataset = CCPD_LPR_Dataset(ccpd_root, train_files, is_train=True)
    val_dataset = CCPD_LPR_Dataset(ccpd_root, val_files, is_train=False)
    test_dataset = CCPD_LPR_Dataset(ccpd_root, test_files, is_train=False)
    collate_fn = CCPD_LPR_Dataset.collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, (len(train_files), len(val_files), len(test_files))

# --- 3. 可视化与日志函数 ---
def plot_confusion_matrix(cm, class_names, file_path):
    fig, ax = plt.subplots(figsize=(25, 25))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel('Predicted labels', fontsize=16)
    ax.set_ylabel('True labels', fontsize=16)
    ax.set_title('Confusion Matrix', fontsize=20)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_training_curves(history, file_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title('Training and Validation Loss', fontsize=16)
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history['val_char_accuracy'], 'go-', label='Val Char Accuracy')
    ax2.plot(epochs, history['val_seq_accuracy'], 'mo-', label='Val Sequence Accuracy')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_title('Validation Accuracy', fontsize=16)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def save_wrong_predictions_image(wrong_samples, file_path):
    if not wrong_samples: return  # noqa: E701
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    num_samples = len(wrong_samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
    if num_samples == 1: axes = [axes]  # noqa: E701
    for i, (img_tensor, true_text, pred_text) in enumerate(wrong_samples):
        img = inv_normalize(img_tensor).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_text}\nPred: {pred_text}", fontsize=14)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def log_to_file(data, log_file):
    with open(log_file, "a") as f:
        f.write(json.dumps(data) + "\n")

# --- 4. 训练、验证、测试核心逻辑 ---
def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
    for images, labels_padded, labels_for_loss in pbar:
        images, labels_padded, labels_for_loss = images.to(device), labels_padded.to(device), labels_for_loss.to(device)
        optimizer.zero_grad()
        outputs = model(images, labels_padded)
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), labels_for_loss.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device, is_test=False):
    model.eval()
    total_loss, all_preds_flat, all_labels_flat, wrong_samples = 0.0, [], [], []
    total_sequences, correct_sequences = 0, 0
    mode = "Testing" if is_test else "Validation"
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"[{mode}]")
        for images, labels_padded, labels_for_loss in pbar:
            images, labels_padded, labels_for_loss = images.to(device), labels_padded.to(device), labels_for_loss.to(device)
            outputs = model(images, labels_padded)
            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), labels_for_loss.reshape(-1))
            total_loss += loss.item()
            preds = outputs.argmax(-1)
            
            non_pad_mask_flat = (labels_for_loss != SPECIAL_TOKENS_DICT["<pad>"])
            all_preds_flat.append(preds[non_pad_mask_flat].cpu().numpy())
            all_labels_flat.append(labels_for_loss[non_pad_mask_flat].cpu().numpy())

            for i in range(labels_for_loss.size(0)):
                label_seq, pred_seq = labels_for_loss[i], preds[i]
                actual_len_mask = (label_seq != SPECIAL_TOKENS_DICT["<pad>"])
                is_correct = (pred_seq[actual_len_mask] == label_seq[actual_len_mask]).all()
                if is_correct:
                    correct_sequences += 1
                elif len(wrong_samples) < 10:
                    true_text = "".join([idx_to_char[idx.item()] for idx in label_seq if idx.item() != SPECIAL_TOKENS_DICT["<pad>"]])
                    pred_text = "".join([idx_to_char[idx.item()] for idx in pred_seq if idx.item() not in [SPECIAL_TOKENS_DICT["<s>"], SPECIAL_TOKENS_DICT["<pad>"], SPECIAL_TOKENS_DICT["</s>"]]])
                    wrong_samples.append((images[i].cpu(), true_text, pred_text))
            total_sequences += labels_for_loss.size(0)

    all_preds_flat = np.concatenate(all_preds_flat)
    all_labels_flat = np.concatenate(all_labels_flat)
    avg_loss = total_loss / len(data_loader)
    char_accuracy = (all_preds_flat == all_labels_flat).mean() * 100
    sequence_accuracy = (correct_sequences / total_sequences) * 100 if total_sequences > 0 else 0
    print(f"[{mode}] Avg Loss: {avg_loss:.4f}, Char Acc: {char_accuracy:.2f}%, Seq Acc: {sequence_accuracy:.2f}%")
    return avg_loss, char_accuracy, sequence_accuracy, all_preds_flat, all_labels_flat, wrong_samples

# --- 5. 主执行逻辑 ---
def main():
    train_loader, val_loader, test_loader, counts = prepare_dataloaders(CCPD_PATH, BATCH_SIZE, SPLIT_RATIO)
    model = LPR_Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_decoder_layers=NUM_DECODER_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS_DICT["<pad>"])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'val_char_accuracy': [], 'val_seq_accuracy': []}
    best_val_loss, best_epoch = float('inf'), -1
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_lpr_model.pth")
    
    print("\n--- 开始训练 ---")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_char_acc, val_seq_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_char_accuracy'].append(val_char_acc)
        history['val_seq_accuracy'].append(val_seq_acc)
        log_to_file({"type": "epoch_summary", "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_char_accuracy": val_char_acc, "val_seq_accuracy": val_seq_acc, "learning_rate": scheduler.get_last_lr()[0]}, LOG_FILE)
        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Val loss 降低，模型已更新并保存到 {best_model_path}")

    plot_training_curves(history, os.path.join(VISUALIZATION_DIR, "training_curves.png"))
    print("\n--- 训练完成 ---")
    
    print("\n--- 在测试集上进行最终评估 ---")
    # --- 修改点: 在 torch.load 中添加 weights_only=True ---
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    
    test_loss, test_char_acc, test_seq_acc, test_preds, test_labels, wrong_samples = evaluate(model, test_loader, criterion, DEVICE, is_test=True)
    log_to_file({"type": "final_test_summary", "test_loss": test_loss, "test_char_accuracy": test_char_acc, "test_seq_accuracy": test_seq_acc}, LOG_FILE)

    class_names = list(CHARSET)
    cm = confusion_matrix(test_labels, test_preds, labels=range(3, len(class_names) + 3))
    plot_confusion_matrix(cm, class_names, os.path.join(VISUALIZATION_DIR, "final_confusion_matrix.png"))
    save_wrong_predictions_image(wrong_samples, os.path.join(VISUALIZATION_DIR, "final_wrong_predictions.png"))

    print("\n--- 生成最终分析报告 ---")
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
    np.fill_diagonal(cm_normalized, 0)
    most_confused_indices = np.dstack(np.unravel_index(np.argsort(cm_normalized.ravel())[-5:], cm_normalized.shape))[0][::-1]
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("           车牌识别模型训练最终报告\n")
        f.write("====================================================\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"所有输出保存在: {os.path.abspath(OUTPUT_DIR)}\n\n")
        f.write("--- 1. 超参数配置 ---\n")
        f.write(f"  - 设备: {DEVICE}\n")
        f.write(f"  - 训练周期 (Epochs): {EPOCHS}\n")
        f.write(f"  - 批次大小 (Batch Size): {BATCH_SIZE}\n")
        f.write(f"  - 初始学习率: {LEARNING_RATE}\n")
        f.write(f"  - 模型维度 (d_model): {D_MODEL}\n")
        f.write(f"  - 多头注意力头数 (nhead): {NHEAD}\n")
        f.write(f"  - 解码器层数: {NUM_DECODER_LAYERS}\n\n")
        f.write("--- 2. 数据集信息 ---\n")
        f.write(f"  - 数据源: {CCPD_PATH}\n")
        f.write(f"  - 划分比例 (Train/Val/Test): {SPLIT_RATIO}\n")
        f.write(f"  - 样本数 (Train/Val/Test): {counts[0]}/{counts[1]}/{counts[2]}\n\n")
        f.write("--- 3. 训练摘要 ---\n")
        f.write(f"  - 最佳模型保存在第 {best_epoch + 1} 个周期\n")
        f.write(f"  - 最佳验证集损失: {best_val_loss:.4f}\n")
        f.write("  - 训练曲线图已保存至: visualizations/training_curves.png\n\n")
        f.write("--- 4. 最终测试性能 (在独立测试集上) ---\n")
        f.write(f"  - 测试集损失: {test_loss:.4f}\n")
        f.write(f"  - **字符识别准确率 (Char Accuracy): {test_char_acc:.2f}%**\n")
        f.write(f"  - **全序列准确率 (Sequence Accuracy): {test_seq_acc:.2f}%**\n\n")
        f.write("--- 5. 性能分析 ---\n")
        f.write("  - 混淆矩阵图已保存至: visualizations/final_confusion_matrix.png\n")
        f.write("  - 错误样本可视化图已保存至: visualizations/final_wrong_predictions.png\n")
        f.write("  - **Top 5 最易混淆字符对 (真实 -> 预测)**:\n")
        for r, c in most_confused_indices:
            f.write(f"    - '{class_names[r]}' -> '{class_names[c]}' (混淆率: {cm_normalized[r,c]:.2%})\n")
        f.write("\n")
        f.write("====================== 报告结束 ======================\n")

    print(f"最终报告已生成: {REPORT_FILE}")

if __name__ == '__main__':
    main()