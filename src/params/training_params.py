class TrainingParams:
    def __init__(self):
        self.bucket = "your-bucket-name"
        self.model_output_key = "models/xgboost"
        self.model_output_path = f"s3://{self.bucket}/{self.model_output_key}"
        self.xgboost_version = "1.7-1"
        self.instance_type = "ml.m5.large"
        self.volume_size = 5
        self.instance_count = 2
        self.content_type = "text/csv"
        self.hyperparameters = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "subsample": "0.7",
            "objective": "binary:logistic",
            "num_round": "300",
        }
