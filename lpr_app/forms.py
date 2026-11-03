# 定义注册表单 (RegisterForm)，包含用户名、密码、确认密码字段，并设置验证规则。
# 定义登录表单 (LoginForm)，包含用户名、密码和“记住我”选项。

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
from .models import User
from flask_wtf.file import FileField, FileRequired, FileAllowed


class RegisterForm(FlaskForm):
    """注册表单"""
    username = StringField(
        '用户名',
        validators=[
            DataRequired(message="用户名不能为空"),
            Length(min=3, max=20, message="用户名长度必须在3到20个字符之间")
        ]
    )
    password = PasswordField(
        '密码',
        validators=[
            DataRequired(message="密码不能为空"),
            Length(min=6, max=20, message="密码长度必须在6到20个字符之间")
        ]
    )
    confirm_password = PasswordField(
        '确认密码',
        validators=[
            DataRequired(message="请再次输入密码"),
            EqualTo('password', message="两次输入的密码不一致")
        ]
    )
    submit = SubmitField('注册')

    def validate_username(self, username):
        """自定义验证函数，检查用户名是否已存在"""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('该用户名已被注册，请选择其他用户名。')


class LoginForm(FlaskForm):
    """登录表单"""
    username = StringField(
        '用户名',
        validators=[DataRequired(message="用户名不能为空")]
    )
    password = PasswordField(
        '密码',
        validators=[DataRequired(message="密码不能为空")]
    )
    remember_me = BooleanField('记住我')
    submit = SubmitField('登录')


# 文件上传
class UploadForm(FlaskForm):
    """文件上传表单"""
    file = FileField(
        '选择文件',
        validators=[
            FileRequired(message="请选择一个文件。"),
            # 'current_app' 是一个代理，指向当前处理请求的应用实例
            # 我们从应用的配置中获取允许的扩展名
            FileAllowed(['png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'], message="只允许上传图片和视频文件！")
        ]
    )
    submit = SubmitField('上传并识别')
