import json

import warnings

warnings.simplefilter(action="ignore")

import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="AmazonSageMaker-ExecutionRole-20220224T111993")[
        "Role"
    ]["Arn"]
    print(role)

model_id = "tiiuae/falcon-40b-instruct"
num_gpus = 4
hub = {"HF_MODEL_ID": model_id, "SM_NUM_GPUS": json.dumps(num_gpus)}

print("Creating hf model...")
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="1.1.0"),
    env=hub,
    role=role,
)

print("Deploying hf model...")

instance_type = "ml.g5.12xlarge"
endpoint_name = "falcon-40b-instruct"

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    container_startup_health_check_timeout=300,
    endpoint_name=endpoint_name,
)


# send request
sequence = predictor.predict(
    {
        "inputs": "Hey Falcon! Any recommendations for my holidays in Abu Dhabi?",
    }
)

print(sequence)
