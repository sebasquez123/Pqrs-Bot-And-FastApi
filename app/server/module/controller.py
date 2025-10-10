from flask import request, jsonify
from app.server.module.chat_bot_service import BotService
import logging

service = BotService()

class BotController:

    def predict():
        try:
            logging.info("Received request for prediction")
            data = request.json 
            
            if(data.get("message") is None):
                logging.warning("No question found in request")
                return jsonify({"error": "there is not questions"}), 400

            prediction = service.predict_model(str(data.get("message")))

            logging.info(f"Prediction made: {prediction}")
            return jsonify({"Prediction":prediction })
        
        except Exception as e:
            logging.error(str(e))
            return jsonify({"error": str(e)}), 500
        
        
    def info():
        try:
            params,test,train = service.model_info()
            return jsonify({"Model Parameters": params, "Test Accuracy": test, "Train Accuracy": train})
        except Exception as e:
            logging.error(f"Error occurred while fetching model info: {str(e)}")
            return jsonify({'error':f'Error fetching model information - {str(e)}'}), 500