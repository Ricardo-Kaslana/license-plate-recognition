from datetime import datetime
from . import db  # 从当前包（lpr_app）的 __init__.py 文件中导入 db 对象
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    """用户模型"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    # 定义与 UploadRecord 的一对多关系
    # 'uploads' 是一个虚拟字段，可以通过 user_instance.uploads 访问该用户的所有上传记录
    # backref='user' 会在 UploadRecord 模型中创建一个虚拟字段 user，可以通过 record_instance.user 访问对应的用户
    # lazy=True 表示在首次访问 uploads 属性时才从数据库加载相关记录
    uploads = db.relationship('UploadRecord', backref='user', lazy=True)

    def set_password(self, password):
        """设置密码，将明文密码哈希后存储"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """校验密码"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class UploadRecord(db.Model):
    """上传记录模型"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # 'image' or 'video'
    upload_path = db.Column(db.String(512), nullable=False)
    processed_path = db.Column(db.String(512), nullable=True)  # 处理后的文件路径，初始可以为空
    upload_timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # 设置外键，关联到 user 表的 id 字段
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<UploadRecord {self.filename}>'