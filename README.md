# 🚀 Phishing URL Detection using XGBoost and SageMaker

A production-ready machine learning pipeline for detecting phishing URLs based on custom-engineered features — built with MLOps best practices including CI/CD, testing, containerization, and cloud deployment on AWS SageMaker.

> 📄 For the full technical writeup and detailed report, see [`docs/project_report.md`](docs/project_report.md)

---

## 🧠 Project Overview

Phishing attacks via malicious URLs are a common cybersecurity threat. This project builds an end-to-end ML pipeline that:

- Extracts custom features from URLs
- Trains an XGBoost classifier
- Deploys the model to a live SageMaker endpoint
- Offers inference via a **Lambda + API Gateway** endpoint
- Demonstrates inference via a **Web UI**
- Integrates CI for linting and unit testing

The project is designed for clarity, reproducibility, and fast deployment.

---

## 📁 Project Structure

```
.
├── data/
│   └── processed_data/             # Processed train/val/test splits
├── notebooks/                      # Archived EDA/training notebooks
├── src/
│   ├── helpers/
│   │   ├── feature_engineering.py  # Feature computation functions
│   │   └── params/
│   │       ├── preprocessing_params.py
│   │       └── training_params.py
│   ├── preprocessing.py            # Loads S3 data, computes features, saves splits
│   └── train_and_deploy.py         # Trains model, deploys endpoint
├── ml_pipelines/
│   └── run_pipeline.sh             # Master script to run preprocessing + training
├── lambda_function/
│   └── lambda_function.py          # Lambda function for real-time inference
├── api_gateway/
│   └── index.html                  # Web UI that calls the API Gateway
├── tests/
│   └── test_feature_engineering.py # Unit tests for feature engineering
├── docs/
│   └── project_report.md           # Full technical write-up and report
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── requirements.txt
└── README.md
```

---

## 🖼️ Architecture Diagram

## 🧭 Architecture Overview

```text
        ┌────────────────────┐
        │    S3 Bucket       │◄──────┐
        │(Raw CSV Dataset)   │       │
        └────────┬───────────┘       │
                 │                   │
        ┌────────▼─────────┐         │
        │ Preprocessing    │         │
        │(Feature Eng. +   │         │
        │ Split & Upload)  │         │
        └────────┬─────────┘         │
                 │                   │
        ┌────────▼──────────┐        │
        │  SageMaker Train  │        │
        │ (XGBoost Model)   │        │
        └────────┬──────────┘        │
                 │                   │
        ┌────────▼──────────────┐    │
        │   SageMaker Endpoint  │────┘
        └────────┬──────────────┘
                 │
        ┌────────▼────────┐
        │ Lambda Function │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  API Gateway    │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  Web UI (S3)    │
        │  index.html     │
        └─────────────────┘

---

## ⚙️ Features Engineered

| Feature | Description |
|--------|-------------|
| `URLLength` | Total length of the URL |
| `DomainLength` | Length of the domain |
| `NoOfSubDomain` | Number of subdomains |
| `LetterRatioInURL` | Ratio of alphabetic characters in the URL |
| `NoOfAmpersandInURL` | Count of '&' in the URL |
| `SpacialCharRatioInURL` | Ratio of special (non-alphanumeric) characters |
| `IsHTTPS` | 1 if URL uses HTTPS, else 0 |
| `CharContinuationRate` | Frequency of repeated adjacent characters |

---

## 📦 Requirements

Clone this repo into a SageMaker Studio instance or your local environment, then install:

```bash
pip install -r requirements.txt
```

Includes:
- `boto3`
- `sagemaker`
- `scikit-learn`
- `pandas`, `numpy`, `pytest`, `flake8`

---

## 🧾 How to Run Locally

### 1. Run preprocessing, training, and deployment
```bash
bash ml_pipelines/run_pipeline.sh
```

### 2. Deploy Lambda + API Gateway (manual steps)
- Package and upload `lambda_function/` to AWS Lambda
- Create a new REST API in API Gateway and connect it to the Lambda
- Enable CORS and deploy the API
- Upload `index.html` to S3 and enable static website hosting

---

## 🧪 Run Unit Tests

```bash
python -m unittest discover -s tests
```

> Includes tests for feature extraction logic.

---

## 🔁 CI/CD with GitHub Actions

CI pipeline includes:
- `flake8` linting
- `pytest`-based unit testing
- Runs on every push and PR to any branch

Config: `.github/workflows/ci.yml`

---

## ☁️ AWS SageMaker Integration

- **Data** is loaded from and saved to S3 buckets
- **Model** is trained on `ml.m5.large` instances using SageMaker's XGBoost container
- **Endpoint** is deployed automatically for real-time inference

---

## 🛠️ Inference Architecture

After training, the model is used for inference via:

### 🔹 Lambda Function (`lambda_function/lambda_function.py`)
- Deploy to AWS Lambda manually
- Should be configured with execution role that can access SageMaker endpoint
- This Lambda function performs real-time inference

### 🔹 API Gateway
- Set up to trigger the Lambda function
- Enable **CORS** so it can be accessed from web browsers
- This exposes the model via a public HTTP endpoint

### 🔹 Web UI (`api_gateway/index.html`)
- Simple HTML form that calls the API Gateway endpoint
- To use publicly:
  1. Upload to an S3 bucket
  2. Enable **static website hosting**
  3. Make the bucket **publicly readable**

---



## ✅ Status

- [x] Feature Engineering
- [x] Unit Testing
- [x] CI/CD with GitHub Actions
- [x] SageMaker Deployment
- [x] Lambda Inference
- [x] API Gateway
- [x] Web UI Demo

---

## ✍️ Author

Built by Harish Kumaravel for MLE portfolio, with emphasis on real-world deployment in the cloud.

---

## 🚀 License

MIT — feel free to fork and build upon.