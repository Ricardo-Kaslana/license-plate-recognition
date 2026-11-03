import torch
import cv2
import argparse
import os
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 导入我们自己的模块
from model import LPR_Transformer
from utils import (
    VOCAB_SIZE,
    SPECIAL_TOKENS_DICT,
    idx_to_char
)


# --- 1. 加载模型与配置 ---
def load_models(yolo_pose_weights_path, lpr_weights_path, device):
    print("正在加载模型...")
    # 加载YOLOv8-Pose模型
    detection_model = YOLO(yolo_pose_weights_path)
    print("YOLOv8-Pose 检测模型加载成功。")

    # 加载LPR识别模型
    recognition_model = LPR_Transformer(
        vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_decoder_layers=6, img_height=48, img_width=160
    ).to(device)
    recognition_model.load_state_dict(torch.load(lpr_weights_path, map_location=device, weights_only=True))
    recognition_model.eval()
    print("LPR Transformer 识别模型加载成功。")
    return detection_model, recognition_model


# --- 2. 核心处理函数 ---
def perspective_transform(image, src_pts):
    """
    根据四个源点进行透视变换
    :param image: 原始图像
    :param src_pts: 4个角点坐标 (左上, 右上, 右下, 左下)
    :return: 校正后的水平车牌图像
    """
    dst_h, dst_w = 48, 160  # LPR模型的标准输入尺寸
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    warped_img = cv2.warpPerspective(image, M, (dst_w, dst_h))
    return warped_img


def preprocess_plate_image(plate_image, device):
    """对校正后的车牌图像进行预处理"""
    plate_image_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(plate_image_rgb).unsqueeze(0).to(device)


def decode_prediction(prediction_indices):
    """将模型输出的索引序列解码为字符串"""
    chars = [idx_to_char.get(idx.item(), '') for idx in prediction_indices if
             idx.item() not in [SPECIAL_TOKENS_DICT["<s>"], SPECIAL_TOKENS_DICT["<pad>"], SPECIAL_TOKENS_DICT["</s>"]]]
    return "".join(chars)


# --- 3. 端到端推理主函数 ---
def end_to_end_inference(image_path, detection_model, recognition_model, device):
    print(f"\n正在处理图片: {image_path}")
    original_image = cv2.imread(image_path)
    img_h, img_w, _ = original_image.shape

    # 1. 使用YOLOv8-Pose进行检测
    results = detection_model.predict(source=original_image, verbose=False)

    if not results or len(results[0].boxes) == 0:
        print("未检测到任何车牌。")
        return original_image

    # 2. 遍历所有检测结果
    for result in results[0]:
        # 提取边界框
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])

        # 提取角点 (最关键的一步)
        if result.keypoints is None or len(result.keypoints.xy[0]) != 4:
            print("警告: 检测到车牌但未找到4个角点，跳过。")
            continue

        # 反归一化角点坐标
        src_pts = result.keypoints.xy[0].cpu().numpy()

        # 3. 透视校正
        aligned_plate = perspective_transform(original_image, src_pts)

        # 4. 识别校正后的车牌
        processed_plate = preprocess_plate_image(aligned_plate, device)
        with torch.no_grad():
            prediction = recognition_model.inference(
                src_img=processed_plate,
                start_token_idx=SPECIAL_TOKENS_DICT["<s>"],
                end_token_idx=SPECIAL_TOKENS_DICT["</s>"]
            )

        plate_text = decode_prediction(prediction)
        print(f"  - 检测到车牌，识别结果: {plate_text}")

        # 5. 使用Pillow绘制中文文本
        # 转换BGR图像为RGB（Pillow使用RGB）
        pil_img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 加载中文字体（确保字体文件存在）
        try:
            font = ImageFont.truetype("simhei.ttf", 20)  # 使用黑体，可替换为其他支持中文的字体
        except IOError:
            font = ImageFont.load_default()  # 如果找不到字体，使用默认（可能不支持中文）
            print("警告: 未找到中文字体，可能无法正确显示中文。")

        # 绘制车牌边界框（绿色）
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)

        # 绘制角点（红色）
        for i, pt in enumerate(src_pts):
            draw.ellipse([(pt[0] - 5, pt[1] - 5), (pt[0] + 5, pt[1] + 5)], fill="red")
            draw.text((pt[0], pt[1]), str(i), fill="white")

        # 绘制车牌识别结果（黑色文字+绿色背景）
        # 计算文本尺寸（兼容Pillow >= 10.0.0）
        text_bbox = draw.textbbox((0, 0), plate_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([(x1, y1 - text_height - 10), (x1 + text_width, y1 - 10)], fill="green")
        draw.text((x1, y1 - text_height - 5), plate_text, font=font, fill="black")

        # 转换回OpenCV格式（BGR）
        original_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return original_image


# === 处理图片和视频函数 ===

def process_image(image_path, detection_model, recognition_model, device, save_result=True):
    """
    处理单张图片的车牌检测与识别
    :param image_path: 输入图片路径
    :param detection_model: 加载好的YOLO检测模型
    :param recognition_model: 加载好的LPR识别模型
    :param device: 计算设备 (cuda/cpu)
    :param save_result: 是否保存结果图片
    :return: (识别结果文本, 结果图片路径)
    """
    # 执行端到端推理
    result_image = end_to_end_inference(image_path, detection_model, recognition_model, device)

    # 保存结果
    output_path = None
    if save_result:
        output_dir = "processed/images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}")
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存到: {output_path}")

    # 结果路径
    return output_path


def process_video(video_path, detection_model, recognition_model, device, save_result=True):
    """
    处理视频的车牌检测与识别（逐帧处理）
    :param video_path: 输入视频路径
    :param detection_model: 加载好的YOLO检测模型
    :param recognition_model: 加载好的LPR识别模型
    :param device: 计算设备 (cuda/cpu)
    :param save_result: 是否保存结果视频
    :return: 结果视频路径
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = None
    writer = None
    if save_result:
        output_dir = "processed/videos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}")
        
        # 使用更浏览器友好的编码
        fourcc = cv2.VideoWriter_fourcc(*'h264')  # type: ignore # 或 'h264'
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理视频: {video_path} (FPS: {fps}, 分辨率: {width}x{height})")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        temp_image_path = f"temp_frame_{frame_count}.jpg"
        cv2.imwrite(temp_image_path, frame)
        result_frame = end_to_end_inference(temp_image_path, detection_model, recognition_model, device)
        
        if writer is not None:
            writer.write(result_frame)

        os.remove(temp_image_path)
        frame_count += 1
        print(f"已处理帧: {frame_count}", end='\r')

    cap.release()
    if writer is not None:
        writer.release()

    # 如果视频仍然有问题，可以尝试FFmpeg转换
    if output_path and os.path.exists(output_path):
        try:
            from utils import convert_video_for_web
            web_output_path = convert_video_for_web(output_path)
            print(f"\n视频处理完成，浏览器兼容版本已保存到: {web_output_path}")
            return web_output_path
        except:  # noqa: E722
            print("\n视频处理完成，但FFmpeg转换失败，返回原始视频")
    
    print(f"\n视频处理完成，总帧数: {frame_count}")
    return output_path


# === 主启动函数 ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="基于YOLOv8-Pose的车牌识别推理脚本")
    parser.add_argument('--yolo_weights', type=str, default="./checkpoints/best_yolov8_pose_model_latest.pt")
    parser.add_argument('--lpr_weights', type=str, default="./checkpoints/best_lpr_model.pth")
    parser.add_argument('--input', type=str, required=True, help='输入文件路径（图片或视频）')
    parser.add_argument('--mode', choices=['image', 'video'], required=True, help='处理模式：image或video')

    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    detection_model, recognition_model = load_models(args.yolo_weights, args.lpr_weights, DEVICE)

    # 根据模式选择处理函数
    if args.mode == 'image':
        result_path = process_image(args.input, detection_model, recognition_model, DEVICE)
        # 显示结果图片
        result_img = cv2.imread(result_path) # type: ignore
        cv2.imshow("LPR Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == 'video':
        result_path = process_video(args.input, detection_model, recognition_model, DEVICE)
        print(f"结果视频已保存到: {result_path}")