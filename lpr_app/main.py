# 处理应用核心功能

import os
import uuid

import torch
from flask import Blueprint, render_template, redirect, url_for, flash, current_app, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

import inference
from .forms import UploadForm
from .models import UploadRecord
from . import db


# 调用模型处理图片或视频
def process_file(input_path):
    """
    处理图片并保存 返回处理后的图片路径
    """
    import time
    # 模拟处理延迟
    time.sleep(2)

    # 从输入路径中提取文件名
    filename = os.path.basename(input_path)
    print(filename)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detection_model, recognition_model = inference.load_models(
        "./checkpoints/best_yolov8_pose_model_latest.pt",
        "./checkpoints/best_lpr_model.pth", device)

    # 根据文件类型创建子目录
    file_ext = filename.rsplit('.', 1)[1].lower()
    if file_ext in ['png', 'jpg', 'jpeg', 'gif']:
        # 处理图片
        output_path = inference.process_image(input_path, detection_model, recognition_model, device)
    else:
        # 处理视频
        output_path = inference.process_video(input_path, detection_model, recognition_model, device)

    return output_path


# --- 蓝图创建 ---
bp = Blueprint('main', __name__)


@bp.route('/', methods=['GET', 'POST'])
@login_required
def index():
    """主页路由，处理文件上传"""
    form = UploadForm()
    if form.validate_on_submit():
        f = form.file.data

        # 1. 保存文件
        # 使用 secure_filename 防止恶意文件名
        original_filename = secure_filename(f.filename)
        # 生成一个唯一的随机字符串 + 保留原始扩展名
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"

        # 根据文件类型确定保存的子目录
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if file_ext in ['png', 'jpg', 'jpeg', 'gif']:
            sub_folder = 'images'
        else:
            sub_folder = 'videos'

        save_dir = os.path.join(upload_folder, sub_folder)
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        file_path = os.path.join(save_dir, unique_filename)
        f.save(file_path)

        # 2. 调用处理函数
        process_file(file_path)

        # 保存记录到数据库时，使用正斜杠作为路径分隔符 否则无法在前端查看图片
        relative_upload_path = f"{sub_folder}/{unique_filename}"  # 使用正斜杠
        relative_processed_path = f"{sub_folder}/{unique_filename}"  # 使用正斜杠

        # 3. 将记录存入数据库
        record = UploadRecord(
            filename=original_filename, # type: ignore
            file_type='image' if sub_folder == 'images' else 'video', # type: ignore
            upload_path=relative_upload_path,  # 存相对路径 保证生成的url路径正确 # type: ignore
            processed_path=relative_processed_path,  # 存相对路径 保证生成的url路径正确 # type: ignore
            user_id=current_user.id # type: ignore
        )
        db.session.add(record)
        db.session.commit()

        flash('文件上传成功，正在处理中...', 'success')
        return redirect(url_for('main.result', record_id=record.id))

    return render_template('index.html', title='主页', form=form)


@bp.route('/result/<int:record_id>')
@login_required
def result(record_id):
    """显示处理结果页面"""
    record = UploadRecord.query.get_or_404(record_id)
    # 安全检查：确保当前用户只能查看自己的上传记录
    if record.user_id != current_user.id:
        flash('无权访问此记录', 'danger')
        return redirect(url_for('main.index'))
    return render_template('result.html', title='识别结果', record=record)


# 创建一个新的路由来提供对上传文件的访问
@bp.route('/uploads/<path:filename>')
@login_required
def get_upload_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


# 创建一个新的路由来提供对已处理文件的访问
@bp.route('/processed/<path:filename>')
@login_required
def get_processed_file(filename):
    return send_from_directory(current_app.config['PROCESSED_FOLDER'], filename)