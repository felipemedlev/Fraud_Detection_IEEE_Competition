#!/usr/bin/env python3
"""
Update SageMaker endpoint with fixed inference code.
This script creates a new model tarball and updates the endpoint.
"""
import boto3
import tarfile
import os
import time
from pathlib import Path

REGION = "eu-north-1"
ENDPOINT_NAME = "fraud-detection-endpoint"
MODEL_DATA_S3 = "s3://sagemaker-eu-north-1-702135187995/fraud-detection-model/model.tar.gz"

def create_code_tarball():
    """Create tarball with updated inference.py"""
    print("Creating source code tarball...")

    with tarfile.open('sourcedir.tar.gz', 'w:gz') as tar:
        tar.add('code/inference.py', arcname='inference.py')
        tar.add('code/preprocessing.py', arcname='preprocessing.py')
        tar.add('code/requirements.txt', arcname='requirements.txt')

    print("✅ Created sourcedir.tar.gz")

def upload_to_s3(local_file, s3_path):
    """Upload file to S3"""
    s3 = boto3.client('s3', region_name=REGION)

    # Parse S3 path
    parts = s3_path.replace('s3://', '').split('/', 1)
    bucket = parts[0]
    key = parts[1]

    print(f"Uploading {local_file} to {s3_path}...")
    s3.upload_file(local_file, bucket, key)
    print(f"✅ Uploaded to S3")

def update_endpoint():
    """Update the SageMaker endpoint with new code"""
    sm = boto3.client('sagemaker', region_name=REGION)

    print(f"\n{'='*60}")
    print("Updating SageMaker Endpoint")
    print(f"{'='*60}\n")

    # Step 1: Create new tarball with updated code
    create_code_tarball()

    # Step 2: Upload to S3
    code_s3_path = "s3://sagemaker-eu-north-1-702135187995/fraud-detection-model/sourcedir.tar.gz"
    upload_to_s3('sourcedir.tar.gz', code_s3_path)

    # Step 3: Get current endpoint config
    endpoint = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    current_config = endpoint['EndpointConfigName']
    print(f"\nCurrent endpoint config: {current_config}")

    # Step 4: Get the current model
    config = sm.describe_endpoint_config(EndpointConfigName=current_config)
    current_model_name = config['ProductionVariants'][0]['ModelName']
    print(f"Current model: {current_model_name}")

    # Step 5: Create new model with updated code
    timestamp = int(time.time())
    new_model_name = f"fraud-detection-model-{timestamp}"

    # Get model details
    model_desc = sm.describe_model(ModelName=current_model_name)

    # Handle both PrimaryContainer and Containers
    if 'PrimaryContainer' in model_desc:
        container = model_desc['PrimaryContainer']
    elif 'Containers' in model_desc and len(model_desc['Containers']) > 0:
        container = model_desc['Containers'][0]
    else:
        raise ValueError("Could not find container configuration in model")

    print(f"\nCreating new model: {new_model_name}")
    sm.create_model(
        ModelName=new_model_name,
        PrimaryContainer={
            'Image': container['Image'],
            'ModelDataUrl': MODEL_DATA_S3,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': code_s3_path,
                'SAGEMAKER_REGION': REGION
            }
        },
        ExecutionRoleArn=model_desc['ExecutionRoleArn']
    )
    print(f"✅ Created model: {new_model_name}")

    # Step 6: Create new endpoint config
    new_config_name = f"fraud-detection-config-{timestamp}"
    print(f"\nCreating new endpoint config: {new_config_name}")

    sm.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': new_model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.xlarge'
        }]
    )
    print(f"✅ Created endpoint config: {new_config_name}")

    # Step 7: Update endpoint
    print(f"\nUpdating endpoint: {ENDPOINT_NAME}")
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=new_config_name
    )

    print(f"\n{'='*60}")
    print("✅ Endpoint update initiated!")
    print(f"{'='*60}")
    print("\nThe endpoint is now updating. This will take several minutes.")
    print("You can monitor the progress with:")
    print(f"  aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME} --region {REGION}")
    print("\nOnce status is 'InService', test with:")
    print("  python test_endpoint_fixed.py")

if __name__ == "__main__":
    update_endpoint()
