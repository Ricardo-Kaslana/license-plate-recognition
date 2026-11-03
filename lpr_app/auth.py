# 处理登录 认证 登出的逻辑

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from .forms import LoginForm, RegisterForm
from .models import User
from . import db

# 创建一个名为 'auth' 的蓝图
# 'auth' 是蓝图的名字，__name__ 帮助蓝图找到自己的位置
# url_prefix='/auth' 表示这个蓝图下所有的路由都会自动加上 /auth 前缀，
bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/register', methods=['GET', 'POST'])
def register():
    # 如果用户已登录，直接重定向到主页
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegisterForm()
    if form.validate_on_submit():
        # form.validate_on_submit() 会在 POST 请求时自动调用所有验证器
        # 如果所有验证通过，则返回 True
        user = User(username=form.username.data) # type: ignore
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('恭喜，您已成功注册！现在可以登录了。', 'success')
        return redirect(url_for('auth.login'))  # 重定向到登录页面

    # 如果是 GET 请求或表单验证失败，则渲染注册页面
    return render_template('register.html', title='注册', form=form)


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        # 检查用户是否存在，并且密码是否正确
        if user is None or not user.check_password(form.password.data):
            flash('用户名或密码无效', 'danger')
            return redirect(url_for('auth.login'))

        # 用户验证成功，使用 Flask-Login 的 login_user 函数记录用户登录状态
        # remember=form.remember_me.data 会根据用户是否勾选“记住我”来决定 session 的有效期
        login_user(user, remember=form.remember_me.data)

        # 获取用户登录后想要访问的页面（如果有）
        next_page = request.args.get('next')
        # 如果没有 next_page 或者 next_page 不是一个有效的相对路径，则重定向到主页
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('main.index')

        flash('登录成功！', 'success')
        return redirect(next_page)

    return render_template('login.html', title='登录', form=form)


@bp.route('/logout')
@login_required  # 确保只有登录的用户才能访问登出链接
def logout():
    logout_user()
    flash('您已成功退出登录。', 'info')
    return redirect(url_for('auth.login'))