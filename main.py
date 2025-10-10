from app.training.index import workshop
from app.server.router import bp
from flask import Flask 
from flask_cors import CORS
import logging



app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

def main():
    selection = input("Welcome to Pqrs-Bot! Choose an option:\n1. Start the API server\n2. Exit\nEnter your choice (1 or 2): ")
    if selection == "1":
        main()
    elif selection == "2":
        workshop()
    elif selection == "3":
        exit()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        index()


def index():
    
    print("""
          
    =================================================================
    Attention!
     
    You are about to start the Pqrs-Bot API server.
    Make sure you have the trained model and vectorizer ready in the corresponding paths.
    Without these files, the system will not be able to process requests correctly.
    
    let's proceed with http request:
    Example:
    
    curl -X POST "http://localhost:5000/api/v1/chatbot" -H "Content-Type: application/json" -d "{\"message\": \"Your sample text here\"}"

    curl -X GET "http://localhost:5000/api/v1/info" -H "Content-Type: application/json"
    
    curl -X GET "http://localhost:5000/api/v1/version" -H "Content-Type: application/json"

    If you want to cancel, press Ctrl+C
    
    =================================================================
    """)
    input("Would you like to continue? Press Enter to proceed or Ctrl+C to cancel...")
    
    
    logging.info("initializing app...")
    
    app.register_blueprint(bp, url_prefix='/api/v1')
        
    CORS(app)

    logging.info("App initialized in port 5000. get in http://localhost:5000/api/v1")
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
