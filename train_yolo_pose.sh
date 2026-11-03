#!/bin/bash

# 训练YOLOv8-Pose模型的脚本

# 定义模型和数据集配置文件
MODEL_CONFIG="./yolov8n-pose.pt"  # 使用YOLOv8n-pose预训练模型
DATA_CONFIG="./yolo_ccpd_pose_dataset/ccpd_pose_data.yaml"

# 定义训练参数
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
PROJECT_NAME="LPR_Pose_Detection"
RUN_NAME="yolov8n_ccpd_pose_e100"

# 开始训练
echo "开始训练YOLOv8-Pose模型..."
yolo train model=$MODEL_CONFIG data=$DATA_CONFIG epochs=$EPOCHS batch=$BATCH_SIZE imgsz=$IMG_SIZE project=$PROJECT_NAME name=$RUN_NAME

echo "训练完成!"