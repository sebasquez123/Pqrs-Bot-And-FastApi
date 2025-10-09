from app import create_app
from app.router import bp

app = create_app()
app.register_blueprint(bp, url_prefix='/api')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
    