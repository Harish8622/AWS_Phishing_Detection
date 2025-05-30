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

Initially, training was conducted with **50 boosting rounds**, but the loss curves indicated that the model did not fully converge. as shown in following graph



| Round | Train Log Loss | Validation Log Loss |
|-------|----------------|---------------------|
| 1     | 0.40733        | 0.40781             |
| 2     | 0.32492        | 0.32552             |
| 3     | 0.26293        | 0.26366             |
| 4     | 0.21601        | 0.21676             |
| 5     | 0.17856        | 0.17948             |
| 6     | 0.14947        | 0.15052             |
| 7     | 0.12560        | 0.12660             |
| 8     | 0.10676        | 0.10774             |
| 9     | 0.09169        | 0.09283             |
| 10    | 0.07900        | 0.08010             |
| 11    | 0.06820        | 0.06913             |
| 12    | 0.05941        | 0.06020             |
| 13    | 0.05244        | 0.05310             |
| 14    | 0.04671        | 0.04730             |
| 15    | 0.04223        | 0.04279             |
| 16    | 0.03848        | 0.03894             |
| 17    | 0.03548        | 0.03593             |
| 18    | 0.03301        | 0.03342             |
| 19    | 0.03102        | 0.03150             |
| 20    | 0.02922        | 0.02974             |
| 21    | 0.02776        | 0.02827             |
| 22    | 0.02657        | 0.02707             |
| 23    | 0.02564        | 0.02616             |
| 24    | 0.02465        | 0.02520             |
| 25    | 0.02379        | 0.02426             |
| 26    | 0.02319        | 0.02369             |
| 27    | 0.02254        | 0.02300             |
| 28    | 0.02195        | 0.02244             |
| 29    | 0.02153        | 0.02200             |
| 30    | 0.02120        | 0.02171             |
| 31    | 0.02076        | 0.02134             |
| 32    | 0.02053        | 0.02106             |
| 33    | 0.02012        | 0.02067             |
| 34    | 0.01990        | 0.02046             |
| 35    | 0.01968        | 0.02024             |
| 36    | 0.01938        | 0.01997             |
| 37    | 0.01928        | 0.01987             |
| 38    | 0.01879        | 0.01943             |
| 39    | 0.01844        | 0.01915             |
| 40    | 0.01830        | 0.01902             |
| 41    | 0.01788        | 0.01859             |
| 42    | 0.01751        | 0.01824             |
| 43    | 0.01719        | 0.01792             |
| 44    | 0.01704        | 0.01780             |
| 45    | 0.01686        | 0.01764             |
| 46    | 0.01676        | 0.01754             |
| 47    | 0.01657        | 0.01742             |
| 48    | 0.01643        | 0.01728             |
| 49    | 0.01630        | 0.01724             |


Based on these insights, the final model was re-trained with 300 rounds to ensure full convergence and improved performance. In future work, adding an early stopping criterion could optimize this process further.


### Increasing number of boosting rounds

After increasing the number of boosting rounds to **300**, the model achieved impressive convergence:

- **Final training log loss:** 0.01202  
- **Validation log loss:** 0.01438

This demonstrates the model’s ability to fit the data well while maintaining generalisation. 

#### Training and Validation Log Loss Plot
![XGBoost Log-Loss Curve](plots/train-plot-2.png)

#### Evaluation Metrics on the Test Set

- **Classification Report:**
           precision    recall  f1-score   support

       0       1.00      0.99      1.00     15142
       1       1.00      1.00      1.00     20228

accuracy                           1.00     35370

- **Accuracy:** 0.9969
- **Confusion Matrix:**
[[15046    96]
[   13 20215]]


### Feature Importance and Model Refinement

The initial feature set for the model included:

- `URLLength`
- `DomainLength`
- `IsDomainIP`
- `NoOfSubDomain`
- `LetterRatioInURL`
- `DegitRatioInURL`
- `NoOfEqualsInURL`
- `NoOfQMarkInURL`
- `NoOfAmpersandInURL`
- `SpacialCharRatioInURL`
- `IsHTTPS`
- `CharContinuationRate`
- `URLEntropy`

After analysing the **feature importance scores** and practical considerations, we decided to refine the feature set. This refinement not only improved model performance by removing less impactful features, but also **simplified the inference logic and improved runtime efficiency** by dropping features that were computationally intensive or challenging to calculate in real time.

#### Feature Importance Summary
| Feature                | Importance |
|-------------------------|------------|
| LetterRatioInURL        | 378.0      |
| DomainLength            | 367.0      |
| URLEntropy              | 286.0      |
| SpacialCharRatioInURL   | 226.0      |
| URLLength               | 211.0      |
| DegitRatioInURL         | 168.0      |
| CharContinuationRate    | 140.0      |
| NoOfSubDomain           | 92.0       |
| IsHTTPS                 | 46.0       |

#### Dropped Features
The following features were identified as either:
✅ Having **no significant contribution** to the model’s predictive performance  
✅ Being **challenging to calculate accurately** in real-time (e.g., complex edge cases)  
✅ Or **adding unnecessary computational overhead** during inference

Hence, they were **removed** to create a simpler, faster, and still highly effective model:

- `IsDomainIP`
- `NoOfEqualsInURL`
- `NoOfQMarkInURL`
- `NoOfAmpersandInURL`
- `DegitRatioInURL`
- `SpacialCharRatioInURL`
- `CharContinuationRate`
- `NoOfSubDomain`



#### Final Selected Features
The refined model will use the following core features:
- `DomainLength`
- `URLEntropy`
- `IsHTTPS`
- `LetterRatioInURL`
- `URLLength`
this strikes a balance of keeping the most important features while minimising edge cases in real world from edge case url's
This leaner set of features maintains strong predictive accuracy while **streamlining the overall system** for better scalability and lower latency during inference.

### Retraining with Reduced Feature Set (Train_v2 Notebook)

After identifying and dropping non-contributing features, we retrained the model using the built-in Amazon SageMaker XGBoost algorithm (version 1.7.1). This retraining was done with **300 boosting rounds** to ensure thorough convergence.

#### Model Log-Loss over 300 Rounds
- **Final Training Log-Loss**: 0.01202
- **Final Validation Log-Loss**: 0.01438


#### Final Performance Metrics
| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 0.9969 |
| Precision (0)      | 1.00   |
| Recall (0)         | 0.99   |
| F1-Score (0)       | 1.00   |
| Precision (1)      | 1.00   |
| Recall (1)         | 1.00   |
| F1-Score (1)       | 1.00   |
| Support (0)        | 15,142 |
| Support (1)        | 20,228 |
| Total Support      | 35,370 |
| Confusion Matrix   | `[[15058, 84], [23, 20205]]` |

This final retraining confirms that the model remains highly performant even after reducing the feature set, with virtually perfect classification metrics.

### Summary of Model Performance
| Model Variant                            | Accuracy | Final Validation Log-Loss | Notes                                   |
|------------------------------------------|----------|---------------------------|-----------------------------------------|
| Logistic Regression (benchmark)          | 0.7425   | N/A                       | Simple baseline, only URLLength         |
| XGBoost (300 rounds, full features)      | 0.9969   | 0.01438                   | Full convergence, full feature set      |
| XGBoost (300 rounds, reduced features)   | 0.9969   | 0.01536                  | Final model, reduced feature set        |

despite dropping 8 features, performance is comparable to previous, but here we can simplofy inference pipeline and reduce computational time

## Deploy Endpoint

- now we can see our final model is significantly better than the first iteration, we can deploy an endpoint


we will use a lambda function to retrieve user url process it and invoke endpoint

we add sagemaekr full access to allow lambda to invoke endpoint

here we tested using test payload
{
  "body": "{\"url\": \"https://www.phishing-1244>?\"}"
}

and returns
![XGBoost Log-Loss Curve](plots/lambda-test-1.png)


and with 

{
  "body": "{\"url\": \"https://www.aap.org\"}"
}

and returns
![XGBoost Log-Loss Curve](plots/lambda-test-2.png)


we will use api gateway for this

In future iterations, incorporating early stopping criteria using a custom training script could help prevent potential overfitting and further fine-tune performance.