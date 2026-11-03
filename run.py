# 启动文件
from lpr_app import create_app

# 调用应用工厂函数创建一个 app 实例
app = create_app()

if __name__ == '__main__':
    # 启动 Flask 开发服务器
    # debug=True 会在代码变动后自动重启服务器，并提供详细的错误页面
    app.run(debug=True)
