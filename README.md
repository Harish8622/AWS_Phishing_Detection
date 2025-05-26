# phishing_detection

first have downloaded the csv from `https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset` into s3 bucket
download into a bucket in folder `initial_dataset'
for label, 1 is legitimate and 0 is phishing

dataset is mostly balanced, slight legitiamte majority, percentages are 1    57.189508
0    42.810492,

according to `https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets` if the minority classs percantage is 20-40% it is mildly imbalanced, here the minority class is above this so we will conclude it can be treated as balanced and no further action will need to be taken

we will first select all features that are easy to compute at run time for inference

then perform cor analysis

we dropped no letters due to collinearity kept others due to domain knowledge

then saved to `cleaned_dataset` folder in s3 buckett


baseline model is logistic regression trained just with url length
this is in benchmark-model-notebook
the only feature here was urllength

these are benchmark results
Accuracy: 0.7425093831506181
Precision: 0.7200569918964648
Recall: 0.8994438264738599
F1 Score: 0.7998153612819202
