# 🏪 Rossmann Sales Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0171B9?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Containerization-0db7ed?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

An **end-to-end Machine Learning solution** to predict **daily sales** for Rossmann drug stores.
This project implements a complete **MLOps pipeline**, including data extraction, experiment tracking, model deployment via REST API, and a user-friendly dashboard.

---

## ✨ Features

- **Machine Learning:** Random Forest Regressor trained on store, promotion, and competitor data.
- **Experiment Tracking:** Uses **MLflow** to track model metrics and parameters.
- **Backend API:** Served using **FastAPI** for real-time predictions.
- **Frontend Dashboard:** Interactive UI built with **Streamlit**.
- **Containerization:** Fully Dockerized for easy deployment with **Docker Compose**.
- **Database:** **PostgreSQL** integration for raw data storage.

---

## 🏗️ System Architecture

```mermaid
graph LR
    A[User] -->|Interacts| B(Streamlit Dashboard)
    B -->|Sends JSON| C{FastAPI Backend}
    C -->|Loads| D[ML Model .pkl]
    C -->|Logs| E[MLflow Tracking]
    C -->|Queries| F[(PostgreSQL DB)]
    style B fill:#FF4B4B,stroke:#333,stroke-width:2px,color:white
    style C fill:#009688,stroke:#333,stroke-width:2px,color:white
    style F fill:#336791,stroke:#333,stroke-width:2px,color:white

```

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
| --- | --- |
| **Language** | Python 3.10 |
| **ML Libraries** | Scikit-Learn, Pandas, NumPy |
| **Web Frameworks** | FastAPI (Backend), Streamlit (Frontend) |
| **DevOps** | Docker, Docker Compose |
| **Tools** | MLflow, Uvicorn |

---

## 📂 Project Structure

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

```

---

## ⚡ Quick Start (Docker)

The easiest way to run the application is using Docker Compose.

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd rossmann-sales-pipeline

```

**2. Build and Launch**

```bash
docker compose up --build

```

**3. Access the Services**

* **Dashboard:** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
* **API Documentation:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

---

## 🧪 Manual Installation (Optional)

If you prefer running without Docker:

**1. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Run the API**

```bash
uvicorn src.app.api:app --reload

```

**4. Run the Dashboard**

```bash
streamlit run src/app/dashboard.py

```

---

## 📊 Usage

**Predicting Sales**

1. Open the Dashboard at [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501).
2. Enter the **Store ID** (e.g., 1115), **Day of Week**, and **Promotion** status.
3. Click **Predict Sales**.
4. The model will return the estimated sales in Euros (€).

---

## 🔮 Future Improvements

* Add Cloud Deployment (AWS / Azure).
* Implement Drift Detection.
* Add Unit Tests with PyTest.

---

## 🧑‍💻 Author

**Khaled Saifullah**

* 🔗 [GitHub Profile](https://github.com/ksnishat)
* 📧 [Email Me](mailto:ksnishat@gmail.com)

## 🪪 License

This project is licensed under the MIT License.

```