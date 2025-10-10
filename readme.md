## Design, Analysis, and Presentation of the PQRS Chatbot

This project implements an end-to-end pipeline to build a PQRS (Petitions, Complaints, Claims, Suggestions) classification model and publish it as an HTTP service powered by Flask. It covers everything from corpus preparation and model training to exposing endpoints for inference and metric inspection.

---

## Full exploratory analysis available in the Jupyter notebook `Chatbot_Eda.ipynb`

## Model training workflow

1. **Data source** – `app/model/raw.py` stores a domain-curated dictionary with sample phrases per category.
2. **Pre-processing and vectorization** – `app/training/vectorizer.py`:
	- Uses spaCy (`en_core_web_sm`) to tokenize, normalize, lemmatize, and remove stopwords.
	- Builds a *bag-of-words* model with `CountVectorizer` and persists the vectorizer in `app/model/vectorizer.pkl`.
3. **Dataset creation** – Generates a `DataFrame` where each row represents a processed phrase and its original class.
4. **Train/test split** – `split_data` in `app/training/train.py` splits the dataset (80/20, `train_test_split`, `random_state=1`).
5. **Training** – `train_model` in `app/training/train.py`:
	- Runs a `GridSearchCV` over `LogisticRegression` using the search space defined in `app/config/configs.py` (solver `liblinear`, penalties `l1`/`l2`, multiple `C` and `max_iter` values).
	- Computes train/test accuracy, the classification report, and the confusion matrix.
	- Saves the optimized model as `app/model/model_v1.pkl` and a summary report in `app/model/info.py`.

`app/training/index.py` orchestrates this flow via the `workshop` function, while `app/training/main.py` provides a CLI with explanatory prompts and safeguards to prevent accidental runs.

---

## How to run the training stage

1. Create and activate a virtual environment (PowerShell):

	```powershell
	py -3 -m venv venv
	.\venv\Scripts\Activate.ps1
	```

2. Install dependencies and download the spaCy model:

	```powershell
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	```

3. Launch the project assistant and follow the on-screen instructions (option 1):

	```powershell
	python main.py
	```

	You can also invoke it directly:

	```powershell
	python -m app.training.main
	```

When the process finishes, check `app/model/` to find the model, vectorizer, and the `info.py` file containing the session parameters and metrics.

---

## Inference service (Flask API)

`main.py` also exposes the HTTP server (option 2). The API loads the persisted model and vectorizer, uses spaCy to pre-process inputs, and delegates prediction to `BotService`.

### Exposed endpoints (`/api/v1`)

| Method | Route            | Description                                                                  | Sample payload / Response |
|--------|------------------|-------------------------------------------------------------------------------|---------------------------|
| `GET`  | `/version`       | Health check that returns the deployed version.                              | `"API v1.0"`
| `GET`  | `/info`          | Retrieves best hyperparameters, train/test accuracy, report, and confusion matrix. | `{ "Model Parameters": "{'C': 1, ...}", "Test Accuracy": "0.78", ... }`
| `POST` | `/chatbot`       | Predicts the category of a phrase. Receives JSON with a `message` field.      | Request: `{ "message": "Need tech support" }`<br>Response: `{ "Prediction": "Technical support" }`

**Common errors**: If `message` is missing the API returns `400`; any loading or inference failures return `500` with a descriptive message.

### PowerShell examples

```powershell
Invoke-WebRequest -Uri "http://localhost:5000/api/v1/version" -Method GET

Invoke-WebRequest -Uri "http://localhost:5000/api/v1/info" -Method GET -Headers @{ "Content-Type" = "application/json" }

Invoke-WebRequest -Uri "http://localhost:5000/api/v1/chatbot" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{ "message": "Hey, thanks for your amazing service." }'
```

---

## Deployment with Docker (optional)

The repository includes a `Dockerfile` and a `docker-compose.yml` that build the image from the codebase, install dependencies, and expose the service on port 5000:

```powershell
docker-compose up --build
```

Remember to prepare the model artifacts (`model_v1.pkl`, `vectorizer.pkl`, `info.py`) beforehand so the container can serve predictions.

---

## Key aspects to keep in mind

- **Path consistency**: `app/config/configs.py` centralizes the names and paths of artifacts generated during training.
- **spaCy is mandatory**: Both pre-processing and inference depend on `en_core_web_sm`; the pipeline fails without it.
- **Readable persistence**: `info.py` acts as a snapshot of the hyperparameters and metrics from the last training session.
- **Lightweight validation**: The API checks for the `message` key and logs key events. Extend validation as needed for production scenarios.

This workflow lets you iterate over the dataset, retrain, and deploy new PQRS model versions in a controlled, reproducible way.

---

## Conclusion and projection

The project shows how a well-structured pipeline enables NLP solutions for real PQRS scenarios with an accessible stack (spaCy, scikit-learn, Flask). Its academic value lies in the fact that every stage (corpus curation, pre-processing, vectorization, hyperparameter tuning, artifact persistence, API publishing) is decoupled and documented, simplifying analysis, metric evaluation, and experimentation in learning environments.

Thanks to the modular design, the architecture can scale and adapt quickly to other domains:

- **New vocabulary**: replace or expand the dictionary in `app/model/raw.py`.
- **Different models**: adjust `train_model` to try alternative classifiers (SVM, RandomForest, lightweight Transformers) without breaking the service contract.
- **Additional integrations**: plug the Flask API into existing pipelines (e.g., FastAPI gateways, messaging queues, dashboards) by exposing new routes or middleware.
- **Heterogeneous deployments**: Docker support and centralized paths allow replication across labs, academic clouds, or edge devices.

In short, this codebase can serve as an extensible lab for NLP, MLOps, and microservice development, enabling rapid iteration and scalability toward more complex projects.


