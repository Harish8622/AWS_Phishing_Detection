

# Phishing Detection

### Setup and Data

This project uses several key Python libraries for data processing and model training, including `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, and `boto3` for AWS interactions.

To get started:

1. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib boto3
2.	Download the dataset:
The dataset can be obtained from the UCI Machine Learning Repository - PhiUSIIL Phishing URL dataset.
3.	Upload to S3:
After downloading, upload the dataset to an S3 bucket (e.g., s3://your-bucket/initial_dataset/). This ensures that it is accessible for your SageMaker training jobs.


### Repo Overview

The following Jupyter notebooks were used in this project:

- **eda_notebook.ipynb**: Conducts initial data exploration and cleaning, providing insights into feature distributions and identifying any anomalies or patterns.
- **Benchmark-model-notebook.ipynb**: Implements a simple logistic regression model using only the `URLLength` feature to establish a performance benchmark.
- **training_notebook_v1.ipynb**: Performs feature engineering and trains an initial XGBoost model using all engineered features, analysing its performance metrics and identifying opportunities for improvement.
- **training_notebook_v2.ipynb**: Refines the model by removing non-contributing features (as identified in earlier analysis), re-trains the XGBoost model, and documents the final performance.

Each notebook was critical in building the final deployed solution, ensuring a rigorous approach to feature engineering, model training, and evaluation.


The **lambda function** code, which is responsible for real-time inference by invoking the deployed SageMaker endpoint, is located in the `lambda` folder. This function processes incoming URLs, computes the required features, and calls the model endpoint to get the classification result (phishing or legitimate).

The **data** for this project (such as the cleaned datasets and any supporting files) should be saved locally in the `data` folder. This ensures consistent data management during development, testing, and further experimentation.
# Domain Background

The dataset for this project was sourced from the [UCI Machine Learning Repository - PhiUSIIL Phishing URL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset). It was downloaded into an S3 bucket under the `initial_dataset` folder. This needs to be downloaded before starting

For the classification task, the label column is structured as follows:
- **1** indicates a legitimate URL
- **0** indicates a phishing URL

The dataset is considered mostly balanced, with a slight majority of legitimate URLs:
- **Legitimate (1)**: 57.19%
- **Phishing (0)**: 42.81%

According to [Google’s ML Crash Course on imbalanced datasets](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets), a dataset is considered “mildly imbalanced” if the minority class comprises 20-40% of the total. Since the minority class in this dataset exceeds this threshold, we treat the dataset as balanced and no further resampling or balancing techniques are required.

## Feature Engineering & Preprocessing

The next step involved selecting features that can be efficiently computed during real-time inference. Additionally, an **entropy feature** was engineered using domain knowledge, under the assumption that URLs with higher randomness (higher entropy) are more likely to be phishing.

A correlation analysis was performed to identify redundant or collinear features. Specifically, we removed features representing raw counts of letters or other characters, opting to keep ratio-based features instead, as these are more dynamically tied to the length of the URL.

The entropy feature showed an inverse correlation with the label, aligning with the expectation that higher entropy is more indicative of phishing URLs.

Finally, the cleaned and processed dataset was saved in the S3 bucket under the `cleaned_dataset` folder. It was split into training, validation, and test sets.

## Benchmark Model

A benchmark model was created using **logistic regression**, implemented in the `benchmark-model-notebook`. This benchmark used only the `URLLength` feature to provide a simple but measurable benchmark.

### Benchmark Results

The logistic regression baseline achieved the following performance:
- **Accuracy**: 0.7425
- **Precision**: 0.7201
- **Recall**: 0.8994
- **F1 Score**: 0.7998

### XGBoost

The final model was trained using Amazon SageMaker’s built-in XGBoost algorithm, eliminating the need for a custom training script. To improve parallelism, training was run across two instances (note: this is optional but serves as a good demonstration of distributed training). The data was fully replicated, although a sharded approach could be taken.

For compatibility with the built-in XGBoost container (version 1.7.1), it was necessary to:
- Remove any header rows from the dataset.
- Ensure that the label column is the first column in the dataset.
- 
#### Initial Model Performance

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| **0** (phishing) | 1.00 | 0.99 | 0.99 | 15142 |
| **1** (legitimate) | 0.99 | 1.00 | 1.00 | 20227 |

- **Accuracy:** 0.9953066244451356  
- **Macro Average:** Precision=1.00, Recall=0.99, F1-score=1.00  
- **Weighted Average:** Precision=1.00, Recall=1.00, F1-score=1.00  

**Confusion Matrix:**

|                | Predicted: Phishing (0) | Predicted: Legitimate (1) |
|----------------|-------------------------|---------------------------|
| **Actual: Phishing (0)**    | 15008                  | 134                       |
| **Actual: Legitimate (1)**  | 32                     | 20195                     |

This matrix highlights how the model maintains very low false positives and false negatives.


The log loss plot can be seen below, showing that 300 rounds was sufficient for the model to converge.

![XGBoost Log-Loss Curve](plots/xgboost-training-1.png)

#### Feature Importance

After this initial training run, feature importance was studied to identify features that were not contributing. The following plot was produced:

![Feature Importance Plot](plots/feature_importance.png)

Based on this, the following features were dropped as they weren't contributing and removing these could help speed up inference:
- `IsDomainIP` (Index 3)
- `NoOfEqualsInURL` (Index 6)
- `NoOfQMarkInURL` (Index 7)
- `URLEntropy` (Index 12)

The remaining columns are:
- `label`
- `URLLength`
- `DomainLength`
- `NoOfSubDomain`
- `LetterRatioInURL`
- `NoOfAmpersandInURL`
- `SpacialCharRatioInURL`
- `IsHTTPS`
- `CharContinuationRate`

#### Final Model Performance

The model was retrained using this reduced set of features and achieved the following:

**Classification Report (Final Model):**

| Class             | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| **Phishing (0)**  | 1.00      | 0.99   | 0.99     | 15142   |
| **Legitimate (1)**| 0.99      | 1.00   | 1.00     | 20228   |

| **Accuracy**      |           |        | 1.00     | 35370   |
| **Macro Avg**     | 1.00      | 0.99   | 1.00     | 35370   |
| **Weighted Avg**  | 1.00      | 1.00   | 1.00     | 35370   |

**Accuracy:** 0.9953915747808878

**Confusion Matrix:**

|                | Predicted: Phishing (0) | Predicted: Legitimate (1) |
|----------------|-------------------------|---------------------------|
| **Actual: Phishing (0)**    | 15012                  | 130                       |
| **Actual: Legitimate (1)**  | 33                     | 20195                     |

The final model achieves very high accuracy and low error rates, maintaining strong performance even after feature reduction.

#### Hyperparameters Used

The final hyperparameters for the XGBoost model were:

- `eta`: 0.2
- `gamma`: 4
- `max_depth`: 5
- `min_child_weight`: 6
- `num_round`: 300
- `objective`: binary:logistic
- `subsample`: 0.7




 ### Summary of Model Performance
| Model Variant                            | Accuracy | Final Validation Log-Loss | Notes                                   |
|------------------------------------------|----------|---------------------------|-----------------------------------------|
| Logistic Regression (benchmark)          | 0.7425   | N/A                       | Simple baseline, only URLLength         |
| XGBoost (300 rounds, full features)      | 0.9953   | 0.01907                   | Full convergence, full feature set      |
| XGBoost (300 rounds, reduced features)   | 0.9953   | 0.01876                  | Final model, reduced feature set        |




## Final Deployment & Inference

Finally, the model was deployed to a real-time inference endpoint using an **ml.m5.large** instance to ensure low-latency predictions. To make this accessible and easy to use, a Lambda function was created. The Lambda function performs the following:

- Extracts and preprocesses features from the incoming URL.
- Calls the SageMaker endpoint with the processed features.
- Returns a prediction label (`phishing` or `legitimate`) and the numeric prediction score.

It’s important to note that the Lambda function requires **SageMaker Full Access** permissions to perform the invoke operation.

### Example Lambda Function Input (Phishing)
```json
{
  "body": "{\"url\": \"http://secure-bank.com.login-now.co\"}"
}
```

This correctly returns phishing 

```json
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"features\": \"35,28,2,0.771,0,0.229,0,0.058824\", \"prediction_score\": 1.4314126417502848e-07, \"prediction_label\": \"phishing\"}"
}
```


### Example Lambda Function Input (Legitimate)


```json
{
  "body": "{\"url\": \"https://www.google.com\"}"
}
```
This correctly returns legitimate


```json
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"features\": \"22,14,1,0.773,0,0.227,1,0.238095\", \"prediction_score\": 0.9979040622711182, \"prediction_label\": \"legitimate\"}"
}
```

### Future Steps

- **Deploy API Gateway**: Currently, the Lambda function can be invoked directly, but I plan to create an API Gateway endpoint to securely expose it as a RESTful API. This will allow external systems to easily integrate and make predictions.

- **Logging and Monitoring**: Implement better logging and monitoring for the endpoint to catch any issues and to gather performance metrics over time.

- **Security Enhancements**: Consider integrating API keys or authentication tokens to ensure only authorised users can access the prediction service.

- **Model Updates**: In future iterations, I will explore ways to retrain and deploy updated models as new phishing patterns emerge to ensure ongoing protection.

- **User-facing Tools**: Build a simple web interface or browser extension to check URLs for phishing in real-time using the deployed model.

