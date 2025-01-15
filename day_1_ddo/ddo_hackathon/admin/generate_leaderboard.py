import os
import sys
import importlib.util
from google.cloud import storage
from google.oauth2 import service_account
import streamlit as st
import tempfile

# Add bm_routine (the parent directory) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports benchmarking routine and problem
from benchmarking import *

SUBMISSIONS_DIR = "day_1_ddo/ddo_hackathon/admin/submissions/ddo_hackathon"
RESULTS_DIR = "day_1_ddo/ddo_hackathon/admin/results"

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Function to process submissions and run your_alg from each
def load_student_algorithms(bucket_name="ddo_hackathon", prefix=""):
    """
    Load all Python submissions from a Google Cloud Storage bucket, dynamically import
    the `your_alg` function from each file, and rename it based on the filename.
    
    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        prefix (str): Optional prefix to filter files in the bucket.

    Returns:
        list: A list of `your_alg` functions from each submission file, renamed to the file name.
    """
    algorithms = []
    
    # Initialize Google Cloud Storage client
    credentials = service_account.Credentials.from_service_account_file("day_1_ddo\ddo_hackathon\intense-pixel-446617-e2-9a9d3fd50dd4.json")
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    # Create a temporary directory to store downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for blob in blobs:
            if blob.name.endswith(".py"):  # Only process Python files
                local_path = os.path.join(temp_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
                print(f"Downloaded {blob.name} to {local_path}")
                
                module_name = os.path.basename(blob.name)[:-3]  # Remove the .py extension
                
                try:
                    # Dynamically import the submission as a module
                    spec = importlib.util.spec_from_file_location(module_name, local_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Check for the presence of the "your_alg" function
                    if hasattr(module, "your_alg"):
                        your_alg_func = module.your_alg
                        # Rename the function to the file name
                        your_alg_func.__name__ = module_name
                        algorithms.append(your_alg_func)
                        print(f"Loaded and renamed your_alg to {module_name} from {blob.name}")
                    else:
                        print(f"Warning: {blob.name} does not contain a your_alg function")
                except Exception as e:
                    print(f"Error loading {blob.name}: {e}")

    return algorithms

def run_benchmark():
    # Load all student algorithms
    algorithms_test = load_student_algorithms()
    print(algorithms_test)
    if not algorithms_test:
        print("No valid your_alg functions found. Exiting.")
        return
    
    # Define additional parameters for ML4CE_uncon_eval
    home_dir = ""
    # N_x_l = [2,5,7]
    # f_eval_l = [50,60,70]

    N_x_l = [2]
    f_eval_l = [50]
    functions_test = ["Rosenbrock_f","Ackley_f"]
    reps = 3
    
    # Run the benchmark
    try:
        info, trajectories, timestamp = ML4CE_uncon_eval(
            N_x_l=N_x_l,
            f_eval_l=f_eval_l,
            functions_test=functions_test,
            algorithms_test=algorithms_test,
            reps=reps,
            home_dir=home_dir,
            SafeData=False
        )
        ML4CE_uncon_graph_abs(
            trajectories,
            algorithms_test,
            functions_test,
            N_x_l,
            home_dir,
            timestamp,
            SafeFig=False,
        )
        print("Benchmark completed successfully.")
    except Exception as e:
        print(f"Error running benchmark: {e}")
    return trajectories


def get_leaderboard():
    traj = run_benchmark()
    html_leaderboard = ML4CE_uncon_leaderboard(traj, as_html=True)
    return html_leaderboard


def upload_to_bucket(html_leaderboard, bucket_name="ddo_hackathon", file_name="leaderboard.html"):
    """
    Uploads the provided HTML string to a Cloud Storage bucket as the leaderboard file.
    Loads credentials from the environment variable 'GOOGLE_APPLICATION_CREDENTIALS'.
    """
    # Load credentials from the provided file path
    credentials = service_account.Credentials.from_service_account_file("day_1_ddo\ddo_hackathon\intense-pixel-446617-e2-9a9d3fd50dd4.json")
    
    # Initialize the Storage client (ADC is used here automatically after providing the credentials)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(html_leaderboard, content_type="text/html")
    print(f"Uploaded leaderboard to gs://{bucket_name}/{file_name}")


if __name__ == "__main__":
    # Generate leaderboard HTML
    print("Starting leaderboard generation...")
    leaderboard_html = get_leaderboard()

    # Upload the generated HTML to Cloud Storage
    upload_to_bucket(leaderboard_html)
    print("Leaderboard update complete.")
