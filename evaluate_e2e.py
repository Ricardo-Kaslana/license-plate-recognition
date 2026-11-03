import torch
import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO
from torchvision import transforms
from tqdm import tqdm

# 导入自定义的模块
from model import LPR_Transformer
from utils import (
    VOCAB_SIZE, 
    SPECIAL_TOKENS_DICT, 
    idx_to_char,
    parse_plate_number  # 从utils.py导入解析真实标签的函数
)

# --- 1. 加载模型与配置 ---
def load_models(yolo_pose_weights_path, lpr_weights_path, device):
    print("正在加载模型...")
    detection_model = YOLO(yolo_pose_weights_path)
    recognition_model = LPR_Transformer(
        vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_decoder_layers=6, img_height=48, img_width=160
    ).to(device)
    recognition_model.load_state_dict(torch.load(lpr_weights_path, map_location=device, weights_only=True))
    recognition_model.eval()
    print("模型加载成功。")
    return detection_model, recognition_model

# --- 2. 核心处理与推理函数 (与inference_with_pose.py类似) ---
def perspective_transform(image, src_pts):
    dst_h, dst_w = 48, 160
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    warped_img = cv2.warpPerspective(image, M, (dst_w, dst_h))
    return warped_img

def preprocess_plate_image(plate_image, device):
    plate_image_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(plate_image_rgb).unsqueeze(0).to(device)

def decode_prediction(prediction_indices):
    chars = [idx_to_char.get(idx.item(), '') for idx in prediction_indices if idx.item() not in [SPECIAL_TOKENS_DICT["<s>"], SPECIAL_TOKENS_DICT["<pad>"], SPECIAL_TOKENS_DICT["</s>"]]]
    return "".join(chars)

def get_true_label(filename):
    """从CCPD文件名解析真实车牌号"""
    try:
        parts = filename.strip().split('-')
        plate_indices = list(map(int, parts[4].split('_')))
        return parse_plate_number(plate_indices)
    except Exception:
        return None

# --- 3. 端到端评测主函数 ---
def evaluate_end_to_end(test_data_dir, detection_model, recognition_model, device):
    image_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith('.jpg')]
    
    total_images = len(image_files)
    detected_count = 0
    correctly_recognized_count = 0
    
    print(f"\n开始在 {total_images} 张图片上进行端到端评测...")
    
    for filename in tqdm(image_files, desc="E2E Evaluation"):
        image_path = os.path.join(test_data_dir, filename)
        
        # 1. 提取真实标签
        true_label = get_true_label(filename)
        if not true_label:
            tqdm.write(f"警告: 无法解析文件名 {filename} 的真实标签，跳过。")
            total_images -= 1 # 从总数中排除无效样本
            continue

        # 2. 端到端推理
        original_image = cv2.imread(image_path)
        if original_image is None:
            tqdm.write(f"警告: 无法读取图片 {filename}，跳过。")
            total_images -= 1
            continue
            
        # 使用YOLOv8-Pose进行检测
        results = detection_model.predict(source=original_image, verbose=False)
        
        # 检查是否检测到任何物体
        if not results or len(results[0].boxes) == 0:
            continue # 检测失败

        # 成功检测到车牌
        detected_count += 1
        
        # 选择置信度最高的一个结果
        best_result = results[0][0] # Ultralytics结果已按置信度排序
        
        # 检查是否有角点
        if best_result.keypoints is None or len(best_result.keypoints.xy[0]) != 4:
            continue # 虽然检测到，但无角点，无法进行准确识别

        # 提取角点并进行校正
        src_pts = best_result.keypoints.xy[0].cpu().numpy()
        aligned_plate = perspective_transform(original_image, src_pts)
        
        # 识别校正后的车牌
        processed_plate = preprocess_plate_image(aligned_plate, device)
        with torch.no_grad():
            prediction_indices = recognition_model.inference(
                src_img=processed_plate,
                start_token_idx=SPECIAL_TOKENS_DICT["<s>"],
                end_token_idx=SPECIAL_TOKENS_DICT["</s>"]
            )
        
        predicted_label = decode_prediction(prediction_indices)
        
        # 比较结果
        if predicted_label == true_label:
            correctly_recognized_count += 1

    # 4. 计算并返回最终指标
    detection_rate = (detected_count / total_images) * 100 if total_images > 0 else 0
    e2e_accuracy = (correctly_recognized_count / detected_count) * 100 if detected_count > 0 else 0
    
    return {
        "total_images": total_images,
        "detected_count": detected_count,
        "correctly_recognized_count": correctly_recognized_count,
        "detection_rate": detection_rate,
        "e2e_accuracy": e2e_accuracy
    }

# --- 4. 主执行块 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="端到端车牌识别系统性能评测脚本")
    parser.add_argument('--yolo_weights', type=str, help='YOLOv8-Pose模型权重路径', 
                        default="./checkpoints/best_yolov8_pose_model_latest.pt")
    parser.add_argument('--lpr_weights', type=str, help='LPR Transformer模型权重路径',
                        default="./checkpoints/best_lpr_model.pth")
    parser.add_argument('--test_dir', type=str, help='CCPD测试集图片文件夹路径 (例如 ccpd_challenge)', 
                        default="../CCPD2019/ccpd_challenge")
    
    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    detection_model, recognition_model = load_models(args.yolo_weights, args.lpr_weights, DEVICE)
    
    # 执行评测
    report = evaluate_end_to_end(args.test_dir, detection_model, recognition_model, DEVICE)
    
    # 打印最终报告
    print("\n====================================================")
    print("      端到端车牌识别系统性能评测报告")
    print("====================================================\n")
    print(f"测试数据集: {os.path.abspath(args.test_dir)}")
    print(f"总计有效图片数: {report['total_images']}\n")
    print("--- 性能指标 ---\n")
    print(f"成功检测到车牌的图片数: {report['detected_count']}")
    print(f"检测成功率 (Detection Rate): {report['detection_rate']:.2f}%\n")
    print(f"被完全正确识别的车牌数: {report['correctly_recognized_count']}")
    print(f"** 端到端准确率 (E2E Accuracy): {report['e2e_accuracy']:.2f}% **")
    print("   (计算公式: 正确识别数 / 成功检测数)\n")
    print("====================================================")