from flask import Blueprint
from app.server.module.controller import BotController

controller = BotController()
bp = Blueprint('api', __name__)

@bp.route('/version', methods=['GET'])
def index():
    return 'API v1.0'


bp.route('/chatbot', methods=['POST'])(controller.predict)
bp.route('/info', methods=['GET'])(controller.info)