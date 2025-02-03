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

# Update the credentials path if needed (use forward slashes for cross-platform compatibility)
CREDENTIALS_FILE = "day_1_ddo/ddo_hackathon/intense-pixel-446617-e2-9a9d3fd50dd4.json"

def load_student_algorithms(bucket_name="ddo_hackathon", prefix=""):
    """
    Load all Python submissions from a Google Cloud Storage bucket, dynamically import
    the `your_alg` function from each file, and rename it based on the filename.
    
    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        prefix (str): Folder prefix to filter files in the bucket (e.g., "day1/t1" or "day1/t2").
    
    Returns:
        list: A list of `your_alg` functions from each submission file, renamed to the file name.
    """
    algorithms = []
    
    # Initialize Google Cloud Storage client
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
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

def run_benchmark(prefix=""):
    """
    Loads student algorithms using the provided prefix and runs the benchmark routine.
    
    Args:
        prefix (str): Folder prefix to filter student submissions (e.g., "day1/t1" or "day1/t2").
    
    Returns:
        trajectories: The trajectories produced by the benchmark routine.
    """
    # Load all student algorithms filtered by the prefix (track folder)
    algorithms_test = load_student_algorithms(prefix=prefix)
    print(f"Algorithms loaded for prefix '{prefix}':", algorithms_test)
    if not algorithms_test:
        print("No valid your_alg functions found. Exiting benchmark for this track.")
        return None
    
    # Define additional parameters for ML4CE_uncon_eval
    home_dir = ""
    N_x_l = [2]
    f_eval_l = [50]
    functions_test = ["Rosenbrock_f", "Ackley_f"]
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
        # Optionally, generate graphs (this may create files in your results directory)
        ML4CE_uncon_graph_abs(
            trajectories,
            algorithms_test,
            functions_test,
            N_x_l,
            home_dir,
            timestamp,
            SafeFig=False,
        )
        print("Benchmark completed successfully for prefix", prefix)
    except Exception as e:
        print(f"Error running benchmark for prefix {prefix}: {e}")
        return None
    return trajectories

def get_leaderboard(prefix=""):
    """
    Runs benchmarking for a given track (prefix), takes algorithm trajectories and 
    generates the leaderboard as an HTML string.
    
    Args:
        prefix (str): Folder prefix corresponding to the track.
    
    Returns:
        str: An HTML representation of the leaderboard.
    """
    traj = run_benchmark(prefix=prefix)
    if traj is None:
        return "<h1 style='color: red;'>No algorithms uploaded and benchmarked yet</h1>"
    html_leaderboard = ML4CE_uncon_leaderboard(traj, as_html=True)
    return html_leaderboard

def upload_to_bucket(html_leaderboard, bucket_name="ddo_hackathon", file_name="leaderboard.html"):
    """
    Uploads the provided HTML string to a Cloud Storage bucket as the leaderboard file.
    
    Args:
        html_leaderboard (str): The HTML content of the leaderboard.
        bucket_name (str): The name of the Cloud Storage bucket.
        file_name (str): The destination file name (including folder path if required).
    """
    # Load credentials from the provided file path
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    
    # Initialize the Storage client
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(html_leaderboard, content_type="text/html")
    print(f"Uploaded leaderboard to gs://{bucket_name}/{file_name}")

if __name__ == "__main__":
    # Generate and upload leaderboard for Track 1
    print("Starting leaderboard generation for Track 1...")
    leaderboard_html_t1 = get_leaderboard(prefix="day1/t1")
    upload_to_bucket(leaderboard_html_t1, file_name="day1/leaderboard_t1.html")
    print("Track 1 leaderboard update complete.")

    # Generate and upload leaderboard for Track 2
    print("Starting leaderboard generation for Track 2...")
    leaderboard_html_t2 = get_leaderboard(prefix="day1/t2")
    upload_to_bucket(leaderboard_html_t2, file_name="day1/leaderboard_t2.html")
    print("Track 2 leaderboard update complete.")
