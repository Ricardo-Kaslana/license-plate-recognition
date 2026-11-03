import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# 从 utils.py 需要的变量和函数
from utils import (
    char_to_idx,
    idx_to_char,
    MAX_SEQ_LEN,
    SPECIAL_TOKENS_DICT,
    parse_plate_number
)


class CCPD_LPR_Dataset(Dataset):
    def __init__(self, ccpd_root, image_files, img_size=(48, 160), is_train=True):
        """
        初始化数据集
        :param ccpd_root: CCPD原始图片集的根目录
        :param image_files: 一个包含图片文件名的Python列表
        :param img_size: (height, width) 模型输入图片的尺寸
        :param is_train: 是否为训练模式 (决定是否应用数据增强)
        """
        self.ccpd_root = ccpd_root
        self.img_size = img_size
        self.is_train = is_train
        
        # 直接使用传入的文件列表
        self.image_files = image_files

        # 定义训练和测试时不同的图像变换
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)

    def parse_filename(self, filename):
        """
        解析文件名，获取四个角点和车牌号索引
        格式: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        """
        try:
            parts = filename.strip().split('-')
            # 角点坐标
            four_corners_str = parts[3].split('_')
            four_corners = np.array([list(map(int, c.split('&'))) for c in four_corners_str], dtype=np.float32)

            # 车牌号索引
            plate_indices = list(map(int, parts[4].split('_')))

            return four_corners, plate_indices
        except Exception as e:
            print(f"Warning: Could not parse filename {filename}. Error: {e}")
            return None, None

    def perspective_transform(self, image, four_corners):
        """
        对图像进行透视变换，校正车牌
        :param image: 原始图像 (OpenCV BGR格式)
        :param four_corners: 4个角点坐标 (左上, 右上, 右下, 左下)
        :return: 校正并缩放后的车牌图像
        """
        h, w = self.img_size
        # CCPD的角点顺序是 右下, 左下, 左上, 右上. 我们需要调整为 左上, 右上, 右下, 左下
        # 索引顺序: [2, 3, 0, 1]
        src_pts = four_corners[[2, 3, 0, 1], :]
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # 计算透视变换矩阵并应用
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(image, M, (w, h))

        return warped_img

    def __getitem__(self, index):
        while True:  # 循环直到找到一个可用的样本
            filename = self.image_files[index]
            image_path = os.path.join(self.ccpd_root, filename)

            # 1. 解析文件名
            four_corners, plate_indices = self.parse_filename(filename)

            if four_corners is None or plate_indices is None:
                # 如果解析失败，随机选择另一个样本
                index = random.randint(0, len(self) - 1)
                continue

            # 2. 读取和校正图像
            image = cv2.imread(image_path)
            if image is None:
                index = random.randint(0, len(self) - 1)
                continue

            # OpenCV读取的是BGR, transform需要RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 校正车牌
            cropped_plate = self.perspective_transform(image, four_corners)

            # 应用transform (增强, ToTensor, Normalize)
            transformed_image = self.transform(cropped_plate)

            # 3. 处理标签
            plate_str = parse_plate_number(plate_indices)

            # 将字符串转换为索引序列, 并添加<s>和</s>
            label = [SPECIAL_TOKENS_DICT["<s>"]]
            label.extend([char_to_idx[c] for c in plate_str])
            label.append(SPECIAL_TOKENS_DICT["</s>"])

            # 填充到最大长度
            # 注意：我们将返回两个版本的标签
            # label_padded: 用于Transformer的输入 (decoder input)
            # label_for_loss: 用于计算损失 (decoder target)

            # e.g., for "皖A123", label = [<s>, 皖, A, 1, 2, 3, </s>]

            # decoder_input: [<s>, 皖, A, 1, 2, 3, </s>, <pad>, <pad>]
            label_padded = torch.full((MAX_SEQ_LEN,), SPECIAL_TOKENS_DICT["<pad>"], dtype=torch.long)
            label_padded[:len(label)] = torch.tensor(label, dtype=torch.long)

            # decoder_target for loss: [皖, A, 1, 2, 3, </s>, <pad>, <pad>, <pad>]
            label_for_loss = torch.full((MAX_SEQ_LEN,), SPECIAL_TOKENS_DICT["<pad>"], dtype=torch.long)
            label_for_loss[:len(label) - 1] = torch.tensor(label[1:], dtype=torch.long)

            return transformed_image, label_padded, label_for_loss

    @staticmethod
    def collate_fn(batch):
        """
        自定义的collate_fn来处理批次数据。
        DataLoader会调用这个函数将Dataset的输出打包成一个batch。
        """
        images, labels_padded, labels_for_loss = zip(*batch)

        images = torch.stack(images, 0)
        labels_padded = torch.stack(labels_padded, 0)
        labels_for_loss = torch.stack(labels_for_loss, 0)

        return images, labels_padded, labels_for_loss


# --- 使用示例 ---
if __name__ == '__main__':
    # CCPD2019 数据集原始图片所在的根目录
    CCPD_DATA_PATH = r"../CCPD2019/ccpd_base"
    BATCH_SIZE = 8

    # 创建训练集和验证集
    train_dataset = CCPD_LPR_Dataset(ccpd_root=CCPD_DATA_PATH, is_train=True) # type: ignore
    val_dataset = CCPD_LPR_Dataset(ccpd_root=CCPD_DATA_PATH, is_train=False) # type: ignore
    test_dataset = CCPD_LPR_Dataset(ccpd_root=CCPD_DATA_PATH, is_train=False) # type: ignore

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # 使用多进程加载数据
        pin_memory=True,  # 加速数据从CPU到GPU的转移
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    # 从 DataLoader 中取一个批次的数据来检查
    print("\n--- 检查一个批次的数据 ---")
    imgs, lbls_padded, lbls_for_loss = next(iter(train_loader))

    print(f"图片批次尺寸: {imgs.shape}")
    print(f"填充后的标签 (用于decoder输入) 尺寸: {lbls_padded.shape}")
    print(f"用于计算损失的标签尺寸: {lbls_for_loss.shape}")

    print("\n--- 样本0 示例 ---")
    print(f"Decoder 输入: {lbls_padded[0].tolist()}")
    print(f"Decoder 输入 (字符): {''.join([idx_to_char[i.item()] for i in lbls_padded[0]])}")

    print(f"目标标签 (用于loss): {lbls_for_loss[0].tolist()}")
    print(f"目标标签 (字符): {''.join([idx_to_char[i.item()] for i in lbls_for_loss[0]])}")