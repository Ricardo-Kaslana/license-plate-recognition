## 基于 CNN-Transformer 的端到端车牌识别系统

### 系统结构
```md
root/
├── checkpoints/                    # 存放训练好的 yolo-pose 检测模型和 trainsformer 模型
│   └── best_lpr_model.pth
│   └── best_yolov8_pose_model.pt
│   └── best_yolov8_pose_model_latest.pt
│ 
├── instance/
│   └── database.db                 # SQLite 数据库文件
│
├── lpr_app/
│   ├── __init__.py                 # 应用工厂函数 create_app()
│   ├── models.py                   # 数据库模型 (User, UploadRecord)
│   ├── auth.py                     # 认证蓝图 (登录、注册、登出路由)
│   ├── main.py                     # 主功能蓝图 (主页、上传、结果展示路由)
│   ├── forms.py                    # WTForms 表单类定义
│   ├── utils.py                    # 工具函数 (模型处理函数等)
│   │
│   ├── static/                     # 静态文件
│   │   ├── css/
│   │   │   └── style.css           # 自定义样式
│   │   ├── js/
│   │   │   └── script.js           # 自定义脚本
│   │   └── images/
│   │       └── logo.png            # 网站Logo等
│   │
│   └── templates/                  # Jinja2 模板
│       ├── base.html               # 基础模板，包含导航栏和页脚
│       ├── index.html              # 主页面 (上传表单)
│       ├── login.html              # 登录页面
│       ├── register.html           # 注册页面
│       └── result.html             # 显示处理结果的页面
│
├── LPR_Pose_Detection/             # 存放 yolo-pose 模型的训练结果
│
├── LPR_Rec/                        # 存放 trainsformer 识别模型的训练结果
│
├── processed/                      # 存放模型处理后的文件(图片 视频)
│   └── images/
│   └── videos/
│
├── template/                       # 存放 yolo 训练配置的模板 yaml 文件
│
├── ultralytics/                    # yolov8 源代码
│
├── uploads/                        # 存放用户上传的原始文件(图片 视频)
│   └── images/
│   └── videos/
│
├── yolo_ccpd_pose_dataset/         # yolo 格式的数据集
│   ├── images/
│   │   ├── test/                   # 测试集
│   │   ├── train                   # 训练集
│   │   └── val                     # 验证集
│   └── lables/
│       ├── test/                   # 测试集标签
│       ├── train                   # 训练集标签
│       └── val                     # 验证集标签
│
├── dateset.py                      # 数据集类
├── inferenec.py                    # 端到端推理脚本
├── model.py                        # transformer 模型类模块
├── prepare_yolo_pose_data.py.py    # 准备 yolo 格式数据集的脚本
├── run.py                          # 项目启动脚本
├── train_lpr.py                    # 训练 transformer 模型的脚本
├── utils.py                        # 工具类
├── train_lpr.py                    # 训练 yolo-pose 模型的脚本
├── yolo11n.pt                      # 预训练的 yolo 模型
├── yolov8n-pose                    # 预训练的 yolo-pose 模型
└── requirements.txt                # 项目依赖包列表
```

### 数据集
CCPD2019

### 训练环境(AutoDL)
```text
GPU RTX 4090(24GB) * 1
CPU 16 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
内存 120GB
硬盘 系统盘:30 GB
cuda <= 12.4
Python  3.12(ubuntu22.04)
PyTorch  2.5.1
```

### 服务运行环境
```text
GPU RTX 4060(24GB) * 1
CPU 12th Gen Intel(R) Core(TM) i7-12650H @ 2.30 GHz
内存 RAM 16.0 GB
Python  3.13(windows_x64)
Flask + Uvicorn
SQLite
```

### 开发IDE
Visual Studio Code

### 项目依赖
yolov8
ffmpeg

### 第三方库
```text
torch
torchvision
torchaudio
numpy
opencv
tqdm
ultralytics
pillow
scikit-learn 
matplotlib 
seaborn
chardet
bcrypt 
flask 
flask-login 
flask-sqlalchemy  
flask-wtf 
uvicorn  
werkzeug
```

### Ubuntu下训练时 注意安装中文字体
```bash
apt-get install fonts-wqy-zenhei
```

### 注意下载 yolo 预训练模型
```text
yolo11n.pt yolov8n-pose.pt
```

### 安装 torch 及其它环境
```bash
pip install -r requirements.txt
# 或者使用uv
uv sync
```

### 运行 prepare_yolo_pose_data.py 准备 yolo_pose 格式的数据集
```bash
python prepare_yolo_pose_data.py
```

### 将 template 文件夹下模板 yaml 文件复制到 yolo_ccpd_pose_dataset 文件夹下的 yaml 文件中
```yaml
path: /root/autodl-tmp/lpr/yolo_ccpd_pose_dataset
train: images/train
val: images/val
test: images/test

# Keypoints
kpt_shape: [4, 2] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)

# Classes dictionary
names:
    0: license_plate

flip_idx: []  # 空数组表示无对称性
fliplr: 0.0   # 显式关闭水平翻转
```

### 运行 train_yolo_pose.sh 脚本训练 yolo_pose 检测模型
```bash
./train_yolo_pose.sh
```
或直接运行 yolo 命令
```bash
yolo train model=./yolov8n-pose.pt data=./yolo_ccpd_pose_dataset/ccpd_pose_data.yaml epochs=100 batch=16 imgsz=640 project=LPR_Pose_Detection name=yolov8n_ccpd_pose_e100
```

### 运行 train_lpr.py 脚本训练识别模型
```bash
python train_lpr.py
```

### 端到端识别
```bash
python inference.py  --input test_car_1.jpg --mode image  
```

### 端到端识别准确率评估
```bash
python evaluate_e2e.py --test_dir your_test_dir
```

### 系统启动
```bash
python run.py # 运行 Flask 启动文件启动服务器
```

### 模型权重
`checkpoints/` 下的训练好的模型权重不随仓库分发，需要自行训练得到：
- `best_yolov8_pose_model_latest.pt` — 由 `train_yolo_pose.sh` 训练产生
- `best_lpr_model.pth` — 由 `train_lpr.py` 训练产生

### 国内 torch 加速安装
如果从 `download.pytorch.org` 下载 torch 很慢或连接被重置，可以先从国内镜像手动下载 wheel 再本地安装：
```bash
# 阿里云镜像
https://mirrors.aliyun.com/pytorch-wheels/cu121/
# 上海交大镜像
https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/
```
下载对应 Python 版本和平台的 `torch / torchvision / torchaudio` wheel 后，将 `pyproject.toml` 中 `[tool.uv.sources]` 的 torch 源临时指向本地 wheel 文件，再执行 `uv sync`。

## License
本项目基于 [MIT License](LICENSE) 开源。
