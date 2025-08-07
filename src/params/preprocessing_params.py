class PreprocessingParams:
    def __init__(self):
        self.bucket = "your-bucket-name"
        self.raw_s3_key = "initial_dataset/PhiUSIIL_Phishing_URL_Dataset 3.csv"
        self.raw_s3_uri = f"s3://{self.bucket}/{self.raw_s3_key}"
        self.processed_local_dir = "data/processed_data"
        self.processed_s3_prefix = "data/processed_data"
        self.test_size = 0.15
        self.val_size = 0.176
        self.random_state = 42
