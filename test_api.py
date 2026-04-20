import urllib.request
import urllib.parse
import json

url = "http://127.0.0.1:8000/scan"

# For file upload, urllib is a bit annoying so we will manually craft multipart/form-data
import mimetypes
import uuid

boundary = uuid.uuid4().hex
headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}

with open('test_prescription.jpg', 'rb') as f:
    file_content = f.read()

payload = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test_prescription.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + file_content + f"\r\n--{boundary}--\r\n".encode('utf-8')

req = urllib.request.Request(url, data=payload, headers=headers)
try:
    with urllib.request.urlopen(req) as f:
        print("Status Code:", f.status)
        print("Response JSON:", f.read().decode('utf-8'))
except urllib.error.URLError as e:
    print("Exception occurred:", e)
    if hasattr(e, 'read'):
        print("Error content:", e.read().decode('utf-8'))
