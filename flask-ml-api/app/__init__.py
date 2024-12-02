from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Middleware de ejemplo
    @app.before_request
    def middleware():
        print("Procesando solicitud...")

    return app