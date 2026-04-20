import sys
from src.pipeline.hybrid_pipeline import run_pipeline

image_path = r"C:\Users\18ary\.gemini\antigravity\brain\1bea6b0a-0d79-49c6-bd61-ca970886de8b\sample_prescription_1776583011100.png"

try:
    print("Testing Pipeline with generated image...")
    results = run_pipeline(image_path)
    print("Pipeline Success. Results:")
    print(results)
except Exception as e:
    import traceback
    traceback.print_exc()
