# Step-by-Step AWS Integration Guide 
*(Serverless Docker Architecture)*

This guide outlines exactly how a Python PyTorch Docker model is translated from a local computer into a scalable, serverless AWS Cloud environment.

## Phase 1: Local Docker Setup & ECR Registry Building
Because our OCR heavily relies on large PyTorch neural weights and OpenCV visual logic, the standard "zip file" upload method for AWS Lambda will fail natively due to size limits. Instead, we bundle the entire system into a secure Docker image and push it to **Amazon Elastic Container Registry (ECR)**.

1. **Install Prerequisites**: Ensure you have `Docker Desktop` and the `AWS CLI` installed locally on your Windows machine, and you are logged into your IAM profile via `aws configure`.
2. **Create ECR Repository**: Within the AWS Console, you must mathematically spin up a private repository named `medical-ocr-api` inside ECR.
3. **Log Docker into AWS**: You must execute the ECR authentication login string within your local terminal securely passing your tokens to the local Docker engine.
4. **Build Image**: Run `docker build -t medical-ocr-api .` structurally in the project folder to construct the Linux container.
5. **Tag & Push**: Tag the local images aligning specifically with the ECR sequence string, and push the massive gigabyte blobs statically into the AWS cloud. 

---

## Phase 2: Lambda Architecture Implementation
AWS Lambda natively supports executing these massively heavy Docker containers exactly like it would a standard lambda function. 

1. **Create Lambda Instance**: You will deploy a completely new Lambda function, but critically select **"Container Image"** instead of structural blueprints.
2. **Image Selection**: Bind the exact ECR Image URL you just generated to the new Lambda setup parameters.
3. **Modify Limitations (CRITICAL)**: By default, Lambda strictly allows only 128MB of memory and completely kills execution natively after 3 seconds. Since you are booting an entire PyTorch CNN matrix sequence:
    * You **must upgrade the Memory parameter logically to at least 1024 MB (or 2048 MB)**. 
    * You **must violently extend the Timeout parameter organically to 30 or 60 seconds**.
4. **Deploy Environment**: Finalize deployment. AWS parses the container architecture securely onto its cloud backbone grids.

---

## Phase 3: Web Bridging (Function URL or API Gateway)
At this stage, your neural mathematical backend operates flawlessly in the cloud, but you physically cannot reach it from the outside web browser!

1. **Enable Public Endpoint**: The fastest and cheapest structural deployment is enabling the native **"Function URL"** inside your Lambda configuration panel, enabling raw Unauthenticated `CORS` access.
2. **Test Natively**: Open **Postman**, set up a `POST` network bridge targeting the URL you generated, and append the physical `test_prescription.jpg` as binary data into a "form-data" param mapped structurally to your FastAPI.  
3. **Response Monitoring**: Wait approximately 5 seconds on cold-boot. AWS automatically reads your image over the web, invokes inference sequentially, and pumps out exact JSON coordinates matching your local terminal test exactly!
