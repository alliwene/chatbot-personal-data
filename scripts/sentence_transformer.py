import sagemaker
import boto3

from sagemaker.huggingface.model import HuggingFaceModel


sess = sagemaker.Session()

sagemaker_session_bucket = "sentence-transformers-sagemaker-hf-150523"

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="AmazonSageMaker-ExecutionRole-20220224T111993")[
        "Role"
    ]["Arn"]

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

repository = "sentence-transformers/all-mpnet-base-v2"

model_id = repository.split("/")[-1]
s3_location = f"s3://{sess.default_bucket()}/custom_inference/{model_id}/model.tar.gz"


# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    model_data=s3_location,
    role=role,
    transformers_version="4.28",
    pytorch_version="2.0",
    py_version="py310",
)

# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name="sentence-transformers-all-mpnet-base-v2",
)

# "ml.g4dn.xlarge"
# ml.g5.2xlarge
