# ❤️ Heart Disease Prediction using Machine Learning & MLOps

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-success)

---

# 📌 Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting the likelihood of heart disease using the UCI Cleveland Heart Disease dataset.

The objective is not only to build an accurate predictive model but also to demonstrate industry-standard MLOps practices such as experiment tracking, automated testing, containerization, Kubernetes deployment, monitoring, and CI/CD.

---

# 🎯 Objectives

- Predict heart disease using Machine Learning
- Compare multiple machine learning algorithms
- Track experiments using MLflow
- Deploy the model using FastAPI
- Containerize the application using Docker
- Deploy the application on Kubernetes
- Monitor the API using Prometheus metrics
- Automate testing using GitHub Actions

---

# 📂 Project Structure

```
heart-disease-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── artifacts/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── notebooks/
│
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── mlflow.db
```

---

# 📊 Dataset

**Dataset:** UCI Cleveland Heart Disease Dataset

Target Variable

- **0** → No Heart Disease
- **1** → Heart Disease

Dataset contains clinical attributes such as:

- Age
- Sex
- Chest Pain Type
- Cholesterol
- Blood Pressure
- ECG Results
- Maximum Heart Rate
- Exercise-induced Angina
- ST Depression
- Number of Major Vessels
- Thalassemia

---

# 📈 Exploratory Data Analysis

Performed:

- Missing value analysis
- Feature distributions
- Correlation heatmap
- Target class balance
- Chest pain vs target analysis
- Numerical feature visualization

---

# ⚙️ Data Preprocessing

- Missing value handling
- One-hot encoding
- Feature scaling using StandardScaler
- Train-Test Split
- Feature engineering
- Saved preprocessing artifacts using Joblib

---

# 🤖 Machine Learning Models

Models implemented:

- Logistic Regression
- Random Forest Classifier

Hyperparameter tuning performed using **GridSearchCV**.

Best model selected based on ROC-AUC score.

---

# 📊 Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross Validation
- Confusion Matrix

---

# 📦 Model Packaging

Saved artifacts:

- best_model.pkl
- scaler.pkl
- feature_names.pkl

---

# 📈 MLflow Experiment Tracking

MLflow was used to:

- Track model parameters
- Track evaluation metrics
- Compare experiments
- Store trained models
- Log artifacts

---

# 🚀 FastAPI REST API

Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | / | Health Check |
| POST | /predict | Heart Disease Prediction |
| GET | /metrics | Prometheus Metrics |

Swagger Documentation

```
http://localhost:8000/docs
```

---

# 🐳 Docker

Build Image

```bash
docker build -t heart-disease-api .
```

Run Container

```bash
docker run -p 8000:8000 heart-disease-api
```

---

# ☸ Kubernetes

Deploy

```bash
kubectl apply -f k8s/
```

Check Pods

```bash
kubectl get pods
```

Check Services

```bash
kubectl get services
```

---

# 📊 Monitoring

Monitoring implemented using Prometheus.

Available metrics:

- prediction_requests_total
- prediction_latency_seconds

Metrics Endpoint

```
http://localhost:8000/metrics
```

---

# 🔄 CI/CD

GitHub Actions pipeline performs:

- Install dependencies
- Execute Pytest
- Validate code before merge

---

# 🧪 Unit Testing

Tests written using Pytest.

Run tests:

```bash
pytest -v
```

---

# 💻 Installation

Clone repository

```bash
git clone https://github.com/akshaygund144/heart-disease-mlops.git
```

Navigate

```bash
cd heart-disease-mlops
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run API

```bash
uvicorn api.main:app --reload
```

---

# 📸 Project Screenshots

Include screenshots of:

- EDA
- MLflow
- FastAPI Swagger
- Docker
- Kubernetes Pods
- GitHub Actions
- Prometheus Metrics

---

# 🔮 Future Improvements

- Model Registry
- Continuous Deployment
- Grafana Dashboard
- Cloud Deployment (AWS/Azure/GCP)
- Authentication & Authorization
- Automated Retraining Pipeline

---

# 👨‍💻 Author

**Akshay Gund**

M.Tech Artificial Intelligence & Machine Learning

BITS Pilani (WILP)

---

# ⭐ Acknowledgements

- UCI Machine Learning Repository
- Scikit-learn
- FastAPI
- MLflow
- Docker
- Kubernetes
- Prometheus


## 🔄 System Workflow

![System Workflow](architecture.png)