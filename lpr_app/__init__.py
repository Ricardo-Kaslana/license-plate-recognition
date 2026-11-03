import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# 初始化扩展，但此时不关联到具体的 app 实例
# 以便可以在其他模块中（如 models.py）安全地导入它们
db = SQLAlchemy()
login_manager = LoginManager()

# login_manager.login_view 指定了当未登录用户尝试访问受保护页面时，
# 应该被重定向到的登录页面的端点（endpoint）。
# 'auth.login' 指的是名为 'auth' 的蓝图下的 'login' 视图函数。
login_manager.login_view = 'auth.login' # type: ignore
login_manager.login_message = "请先登录以访问此页面。"
login_manager.login_message_category = "info"


def create_app():
    """
    应用工厂函数
    """
    # 创建 Flask app 实例
    # instance_relative_config=True 表示配置文件可以从 instance 文件夹加载
    # instance 文件夹位于 lpr-flask-project/ 根目录下，用于存放不应被版本控制的敏感数据，如数据库文件、密钥等
    app = Flask(__name__, instance_relative_config=True)

    # --- 1. 配置应用 ---
    # a. 从默认配置开始
    app.config.from_mapping(
        # SECRET_KEY 用于保护 session 数据和 CSRF 令牌，生产环境中应使用更复杂的随机值
        SECRET_KEY='dev-secret-key-change-it-for-production',
        # 设置 SQLite 数据库的路径。app.instance_path 指向 instance 文件夹
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{os.path.join(app.instance_path, "database.db")}',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # 定义文件上传和处理的文件夹路径
        UPLOAD_FOLDER=os.path.join(app.root_path, '..', 'uploads'),  # app.root_path 是 lpr_app 目录
        PROCESSED_FOLDER=os.path.join(app.root_path, '..', 'processed'),
        # 允许上传的文件扩展名
        ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
    )

    # 确保 instance 文件夹存在，Flask 不会自动创建它
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass  # 如果文件夹已存在，则什么也不做

    # --- 2. 初始化扩展 ---
    # 将之前创建的扩展实例与 app 关联起来
    db.init_app(app)
    login_manager.init_app(app)

    # --- 3. 定义 user_loader 回调函数 ---
    # Flask-Login 需要这个函数来通过 session 中存储的 user_id 重新加载用户对象
    @login_manager.user_loader
    def load_user(user_id):
        # 导入 User 模型。这里使用函数内导入是为了避免循环依赖问题
        from .models import User
        return User.query.get(int(user_id))

    # --- 4. 注册蓝图 (Blueprints) ---
    from . import auth
    app.register_blueprint(auth.bp)

    from . import main
    app.register_blueprint(main.bp)

    # 将根 URL '/' 指向 main 蓝图的 index 视图
    # 这使得我们可以使用 url_for('index') 来生成主页的 URL
    app.add_url_rule('/', endpoint='index')

    # --- 5. 创建数据库表 ---
    with app.app_context():
        # 确保所有在 models.py 中定义的表都被创建
        # 注意：db.create_all() 不会更新已存在的表结构，
        # 如果模型发生变化，需要手动删除数据库文件或使用数据库迁移工具（如 Flask-Migrate）
        db.create_all()

    # 返回创建好的 app 实例
    return app