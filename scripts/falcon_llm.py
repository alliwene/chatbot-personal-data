import json
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

# Hub Model configuration. https://huggingface.co/models
hub = {"HF_MODEL_ID": "tiiuae/falcon-7b-instruct", "SM_NUM_GPUS": json.dumps(1)}

print("Creating hf model...")
# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="1.0.3"),
    env=hub,
    role=role,
)

print("Deploying hf model...")
# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
    endpoint_name="falcon-7b-instruct",
)


# send request
sequence = predictor.predict(
    {
        "inputs": "Hey Falcon! Any recommendations for my holidays in Abu Dhabi?",
    }
)

print(sequence)
