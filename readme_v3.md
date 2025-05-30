

# phishing_detection

The dataset for this project was sourced from the [UCI Machine Learning Repository - PhiUSIIL Phishing URL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset). It was downloaded into an S3 bucket under the `initial_dataset` folder.

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

The final model was trained using Amazon SageMaker’s built-in XGBoost algorithm, eliminating the need for a custom training script. To improve parallelism, training was run across two instances (note: this is optional but serves as a good demonstration of distributed training) The data is fully replicated although a sharded approach could be taken.

For compatibility with the built-in XGBoost container (version 1.7.1), it was necessary to:
- Remove any header rows from the dataset.
- Ensure that the label column is the first column in the dataset.



Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.99      0.99     15142
           1       0.99      1.00      1.00     20227

    accuracy                           1.00     35369
   macro avg       1.00      0.99      1.00     35369
weighted avg       1.00      1.00      1.00     35369

Accuracy: 0.9953066244451356
Confusion Matrix:
 [[15008   134]
 [   32 20195]]



 link to plot


 after dropping features


 Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.99      0.99     15142
           1       0.99      1.00      1.00     20228

    accuracy                           1.00     35370
   macro avg       1.00      0.99      1.00     35370
weighted avg       1.00      1.00      1.00     35370

Accuracy: 0.9953915747808878
Confusion Matrix:
 [[15012   130]
 [   33 20195]]


 removes unneccesary features not contributing

 improves efficiency


 ### Summary of Model Performance
| Model Variant                            | Accuracy | Final Validation Log-Loss | Notes                                   |
|------------------------------------------|----------|---------------------------|-----------------------------------------|
| Logistic Regression (benchmark)          | 0.7425   | N/A                       | Simple baseline, only URLLength         |
| XGBoost (300 rounds, full features)      | 0.9953   | 0.01907                   | Full convergence, full feature set      |
| XGBoost (300 rounds, reduced features)   | 0.9953   | 0.01876                  | Final model, reduced feature set        |

 created unit test to verify extract features works properly

 run pip install pytest