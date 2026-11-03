import os
import cv2
from tqdm import tqdm
import random
import shutil
import yaml

# --- 1. 配置参数 ---
CCPD_ROOT = r"../CCPD2019/ccpd_base"  # CCPD数据集路径
OUTPUT_DIR = r"./yolo_ccpd_pose_dataset"  # 输出YOLO-Pose格式数据集的目录
SPLIT_RATIO = (0.8, 0.1, 0.1)  # Train, Val, Test


# --- 2. 核心处理函数 ---
def parse_ccpd_for_pose(filename):
    """
    解析CCPD文件名，提取边界框和角点。
    返回一个包含class_id, bbox, keypoints的列表。
    """
    try:
        parts = filename.strip().split('-')
        if len(parts) < 4: return None  # noqa: E701

        # 边界框: 154&383_386&473 -> [xmin, ymin, xmax, ymax]
        bbox_str = parts[2]
        xmin, ymin, xmax, ymax = list(map(int, bbox_str.replace('&', '_').split('_')))

        # 角点: 386&473_177&454_154&383_363&402 (右下, 左下, 左上, 右上)
        corners_str = parts[3].split('_')
        corners_ccpd = [list(map(int, c.split('&'))) for c in corners_str]

        # 将CCPD的角点顺序 (右下, 左下, 左上, 右上) 转换为我们需要的 (左上, 右上, 右下, 左下)
        # CCPD index: [2]   [3]   [0]   [1]
        tl, tr, br, bl = corners_ccpd[2], corners_ccpd[3], corners_ccpd[0], corners_ccpd[1]

        # 打包成bbox和keypoints
        bbox = [xmin, ymin, xmax, ymax]
        keypoints = [tl, tr, br, bl]

        return bbox, keypoints
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None


def convert_to_yolo_pose_format(bbox, keypoints, img_width, img_height):
    """
    将bbox和keypoints转换为YOLO-Pose的归一化格式。
    格式: class_id x_center y_center width height kpt1_x kpt1_y ...
    """
    # 转换bbox
    xmin, ymin, xmax, ymax = bbox
    x_center_norm = ((xmin + xmax) / 2) / img_width
    y_center_norm = ((ymin + ymax) / 2) / img_height
    width_norm = (xmax - xmin) / img_width
    height_norm = (ymax - ymin) / img_height

    # 转换keypoints
    keypoints_norm = []
    for (kx, ky) in keypoints:
        keypoints_norm.extend([kx / img_width, ky / img_height])

    # class_id 为 0
    return [0, x_center_norm, y_center_norm, width_norm, height_norm] + keypoints_norm


# --- 3. 主执行脚本 ---
def main():
    print("--- 开始处理CCPD2019数据集，转换为YOLO-Pose格式 ---")

    if not os.path.exists(CCPD_ROOT):
        print(f"错误: CCPD根目录 '{CCPD_ROOT}' 不存在。")
        return

    print(f"创建输出目录: {OUTPUT_DIR}")
    subdirs = ['train', 'val', 'test']
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', subdir), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', subdir), exist_ok=True)

    all_images = [f for f in os.listdir(CCPD_ROOT) if f.lower().endswith('.jpg')]
    print(f"找到 {len(all_images)} 张图片。")
    random.shuffle(all_images)

    train_count = int(len(all_images) * SPLIT_RATIO[0])
    val_count = int(len(all_images) * SPLIT_RATIO[1])

    datasets = {
        'train': all_images[:train_count],
        'val': all_images[train_count: train_count + val_count],
        'test': all_images[train_count + val_count:]
    }

    for split_name, image_list in datasets.items():
        print(f"\n--- 正在处理 {split_name} 集 ({len(image_list)} 张图片) ---")

        for filename in tqdm(image_list, desc=f"处理 {split_name} 集"):
            parsed_data = parse_ccpd_for_pose(filename)
            if parsed_data is None: continue  # noqa: E701

            bbox, keypoints = parsed_data

            image_path = os.path.join(CCPD_ROOT, filename)
            try:
                img = cv2.imread(image_path)
                img_height, img_width, _ = img.shape
            except Exception as e:
                print(f"警告: 无法读取或处理图片 {filename}: {e}，跳过。")
                continue

            yolo_pose_label = convert_to_yolo_pose_format(bbox, keypoints, img_width, img_height)

            output_image_path = os.path.join(OUTPUT_DIR, 'images', split_name, filename)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            output_label_path = os.path.join(OUTPUT_DIR, 'labels', split_name, label_filename)

            shutil.copy(image_path, output_image_path)

            with open(output_label_path, 'w') as f:
                label_str = " ".join([f"{x:.6f}" for x in yolo_pose_label])
                f.write(label_str)

    # --- 4. 创建 data.yaml 文件 ---
    yaml_path = os.path.join(OUTPUT_DIR, 'ccpd_pose_data.yaml')
    print(f"\n--- 创建YOLO-Pose配置文件: {yaml_path} ---")

    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['license_plate'],
        'kpts_shape': [4, 2]  # 4个点, 每个点2个坐标 (x,y)
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("\n处理完成！")
    print(f"YOLO-Pose格式的数据集已生成在: {os.path.abspath(OUTPUT_DIR)}")
    print(f"YOLO训练配置文件为: {yaml_path}")
    print( "\n下一步: 运行训练脚本来训练YOLOv8-Pose模型。")


if __name__ == '__main__':
    main()