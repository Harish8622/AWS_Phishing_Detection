from sagemaker import Session, get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker import image_uris

from src.helpers.params.training_params import TrainingParams

training_params = TrainingParams()


def train_model():
    role = get_execution_role()
    session = Session()
    region = session.boto_region_name

    container = image_uris.retrieve(
        "xgboost", region, version=training_params.xgboost_version
    )

    estimator = Estimator(
        image_uri=container,
        role=role,
        instance_count=training_params.instance_count,
        instance_type=training_params.instance_type,
        output_path=training_params.model_output_path,
        volume_size=training_params.volume_size,
        hyperparameters=training_params.hyperparameters,
    )

    train_input = TrainingInput(
        f"s3://{training_params.bucket}/data/processed_data/train.csv",
        content_type=training_params.content_type,
    )
    val_input = TrainingInput(
        f"s3://{training_params.bucket}/data/processed_data/validation.csv",
        content_type=training_params.content_type,
    )

    estimator.fit(
        {"train": train_input, "validation": val_input}, wait=True, logs="All"
    )
    print("Training complete.")
    return estimator.model_data  # Return model artifact path


def deploy_model(model_data_path: str):
    role = get_execution_role()
    session = Session()
    region = session.boto_region_name

    container = image_uris.retrieve(
        "xgboost", region, version=training_params.xgboost_version
    )

    model = Model(
        image_uri=container,
        model_data=model_data_path,
        role=role,
        sagemaker_session=session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=training_params.instance_type,
        serializer=CSVSerializer(),
        deserializer=JSONDeserializer(),
    )
    print("Endpoint deployed.")
    return predictor


if __name__ == "__main__":
    model_path = train_model()
    deploy_model(model_path)
