import os
import sys
import importlib.util
from google.cloud import storage
from google.oauth2 import service_account
import streamlit as st
import tempfile
from typing import Callable, List

# Update with the actual path to your Google Cloud service account JSON file.
CREDENTIALS_FILE = "day_1_ddo\ddo_hackathon\intense-pixel-446617-e2-9a9d3fd50dd4.json"


def load_student_algorithms(
    bucket_name: str = "ddo_hackathon",
    gcloud_path: str = "",
    func_name: str = "genetic_algorithm",
) -> List[Callable]:
    """
    Load Python submissions from a specified folder in a GCS bucket, dynamically import
    the module, and extract the function named `func_name`.

    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        gcloud_path (str): Folder path within the bucket (e.g., "day1/t1").
        func_name (str): Name of the function to search for in each file.

    Returns:
        List[Callable]: A list of functions loaded from the submissions.
    """
    algorithms: List[Callable] = []  # List to store loaded functions

    # Initialize the Google Cloud Storage client using the provided credentials.
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE
    )
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    # Ensure the folder path ends with '/' for proper prefix matching.
    if gcloud_path and not gcloud_path.endswith("/"):
        gcloud_path += "/"

    # List all blobs (files) in the specified folder within the bucket.
    blobs = bucket.list_blobs(prefix=gcloud_path)

    # Create a temporary directory to store the downloaded Python files.
    with tempfile.TemporaryDirectory() as temp_dir:
        for blob in blobs:
            # Process only Python files.
            if blob.name.endswith(".py"):
                # Build the local file path to save the downloaded blob.
                local_path = os.path.join(temp_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
                print(f"\n\nDownloaded {blob.name} to {local_path}")

                # Derive a module name from the file name (by stripping the '.py' extension).
                module_name = os.path.basename(blob.name)[:-3]
                try:
                    # Dynamically import the module from the downloaded file.
                    spec = importlib.util.spec_from_file_location(
                        module_name, local_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Check if the module contains the desired function.
                    if hasattr(module, func_name):
                        func = getattr(module, func_name)
                        # Rename the function to the module name for easier identification.
                        func.__name__ = module_name
                        algorithms.append(func)
                        print(
                            f"Loaded and renamed {func_name} to {module_name} from {blob.name}"
                        )
                    else:
                        # Warn if the expected function is not found.
                        print(
                            f"Warning: {blob.name} does not contain a {func_name} function"
                        )
                except Exception as e:
                    # Print any errors encountered during the import process.
                    print(f"Error loading {blob.name}: {e}")

    # Return the list of successfully loaded functions.
    return algorithms


def upload_to_bucket(
    html_leaderboard, bucket_name="ddo_hackathon", file_name="leaderboard.html"
):
    """
    Uploads the provided HTML string to a Cloud Storage bucket as the leaderboard file.

    Args:
        html_leaderboard (str): The HTML content of the leaderboard.
        bucket_name (str): The name of the Cloud Storage bucket.
        file_name (str): The destination file name (including folder path if required).
    """
    # Load credentials from the provided file path
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE
    )

    # Initialize the Storage client
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(html_leaderboard, content_type="text/html")
    print(f"Uploaded leaderboard to gs://{bucket_name}/{file_name}")


if __name__ == "__main__":
    algorithms = load_student_algorithms(
        bucket_name="ddo_hackathon",
        gcloud_path="day2/t1",
        func_name="genetic_algorithm",
    )
