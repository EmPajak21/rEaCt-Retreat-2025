from google.cloud import storage
import os

BUCKET_NAME = "ddo_hackathon"
SUBMISSIONS_DIR = "admin/submissions"
#GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Ensure the directory exists
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

def download_submissions():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()

    for blob in blobs:
        file_path = os.path.join(SUBMISSIONS_DIR, blob.name)
        with open(file_path, "wb") as f:
            blob.download_to_file(f)
        print(f"Downloaded {blob.name} to {file_path}")

if __name__ == "__main__":
    download_submissions()
