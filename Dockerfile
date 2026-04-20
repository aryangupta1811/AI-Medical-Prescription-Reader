# Use the official AWS Lambda Python Base Image
FROM public.ecr.aws/lambda/python:3.11

# Copy constraints list
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Mathematically install CPU-only PyTorch and all dependencies
# We use the extra index URL to ensure Linux grabs the lightweight CPU tensors, preventing massive layer sizes
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the entire workspace into the Lambda Root Container
COPY src ${LAMBDA_TASK_ROOT}/src
COPY data/processed/final_knowledge_base.json ${LAMBDA_TASK_ROOT}/data/processed/final_knowledge_base.json
COPY model_augmented.pth ${LAMBDA_TASK_ROOT}/model_augmented.pth
COPY app.py ${LAMBDA_TASK_ROOT}/app.py

# Set the CMD mathematically to the Mangum AWS Lambda Handler
CMD [ "app.handler" ]
