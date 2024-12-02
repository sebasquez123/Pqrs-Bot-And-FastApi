from flask import request, jsonify
from app.chatbotservice import predict_model,model_info



def predict():
    try:
        data = request.json  # Obtener datos enviados en JSON
        if(data.get("pregunta") is None):
            return jsonify({"error": "No se ha enviado una pregunta"}), 400
        prediction = predict_model(str(data.get("pregunta")))
        return jsonify({"prediction":prediction })
       
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    
def info():
    try:
        params,test,train = model_info()
        return jsonify({"Parametros del modelo": params, "Test_accuracy": test, "Train_accuracy": train})
    except Exception as e:
        return jsonify({'error':f'Error al obtener la información-{str(e)}'}), 400