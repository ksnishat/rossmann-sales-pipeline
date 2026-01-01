# Rossmann Sales Prediction Pipeline

An end-to-end Machine Learning solution to predict daily sales for Rossmann drug stores. This project implements a complete MLOps pipeline, including data extraction, experiment tracking, model deployment via REST API, and a user-friendly dashboard.

## Features
* **Machine Learning:** Random Forest Regressor trained on store, promotion, and competitor data.
* **Experiment Tracking:** Uses **MLflow** to track model metrics and parameters.
* **Backend API:** Served using **FastAPI** for real-time predictions.
* **Frontend Dashboard:** Interactive UI built with **Streamlit**.
* **Containerization:** Fully Dockerized for easy deployment (Docker Compose).
* **Database:** PostgreSQL integration for raw data storage.

## Tech Stack
* **Language:** Python 3.10
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **Web Frameworks:** FastAPI (Backend), Streamlit (Frontend)
* **DevOps:** Docker, Docker Compose
* **Tools:** MLflow, Uvicorn

## Project Structure
```text
├── models/                  # Serialized models (.pkl)
├── notebooks/               # Jupyter notebooks for EDA
├── src/
│   ├── app/
│   │   ├── api.py           # FastAPI Backend
│   │   └── dashboard.py     # Streamlit Frontend
│   └── scripts/
│       ├── train_model.py   # Training script
│       └── ...
├── docker-compose.yml       # Container orchestration
├── Dockerfile               # Image definition
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation