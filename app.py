from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import shutil
import uuid
import os
import boto3

# Initialize lightweight AWS S3 Cloud Sync parameter natively
try:
    s3_client = boto3.client('s3')
except Exception:
    s3_client = None

# Import the native pipeline
from src.pipeline.hybrid_pipeline import run_pipeline

app = FastAPI(title="Medical Prescription OCR API")

# Dynamically enable CORS natively to ensure HTML files can mathematically hit the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all local HTML files and test domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "active", "message": "OCR Deep Learning Engine is securely running on AWS Lambda"}

@app.post("/scan")
async def scan_prescription(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})

    # Securely save the uploaded blob temporarily avoiding Linux specific /tmp/
    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename.replace(' ', '_')}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Execute Neural Architecture Pipeline
        results = run_pipeline(temp_filename)
        
        # 🔥 AWS MICRO-INTEGRATION: S3 Cloud Dataset Sync
        # Automatically pushes physical prescription uploads directly into Amazon's cloud bucket securely!
        try:
            if s3_client:
                s3_client.upload_file(temp_filename, "my-medical-ocr-backup-bucket", f"dataset_sync/{file.filename}")
                print(f"[AWS] Successfully synced {file.filename} natively to Amazon S3!")
        except Exception as aws_error:
            # Fails completely silently if you haven't explicitly set up the AWS keys locally via 'aws configure' yet!
            print(f"[AWS WARNING] S3 Bucket Sync bypassed natively: {aws_error}")
        
        # Cleanup AWS RAM overhead
        os.remove(temp_filename)
        
        return {
            "status": "success",
            "extracted_medicines": results
        }
        
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mangum acts to magically bridge FastAPI directly onto AWS Lambda's architecture
handler = Mangum(app)
