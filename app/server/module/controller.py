from flask import request, jsonify
from app.server.module.chat_bot_service import BotService
import logging

service = BotService()

class BotController:

    def predict(self):
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
        
    def info(self):
        try:
            params,test,train,classification_report,confusion_matrix = service.model_info()
            return jsonify({"Model Parameters": str(params), "Test Accuracy": str(test), "Train Accuracy": str(train), "Classification Report": str(classification_report), "Confusion Matrix": str(confusion_matrix)})
        except Exception as e:
            logging.error(f"Error occurred while fetching model info: {str(e)}")
            return jsonify({'error':f'Error fetching model information - {str(e)}'}), 500