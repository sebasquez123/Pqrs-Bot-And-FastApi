from flask import Blueprint
from app.controller import predict, info

bp = Blueprint('api', __name__)

@bp.route('/')
def index():
    return 'API v1.0'

bp.route('/chatbot', methods=['POST'])(predict)
bp.route('/info', methods=['GET'])(info)