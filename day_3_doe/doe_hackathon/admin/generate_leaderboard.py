import os
import sys
import ast
from typing import Dict, Any, List, Tuple

# google cloud storage and streamlit imports
from google.cloud import storage
from google.oauth2 import service_account

# Add parent directories to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bio_model import (
    evaluate_student_solution,
    generate_test_data,
    true_model_day3,
)

from hackathon_utils import upload_to_bucket

CREDENTIALS_FILE: str = (
    "day_1_ddo/ddo_hackathon/intense-pixel-446617-e2-9a9d3fd50dd4.json"
)


def load_json_submissions(
    prefix: str, bucket_name: str = "ddo_hackathon"
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Loads all JSON submissions found in the specified GCS prefix (folder).

    Args:
        prefix (str): The folder path in the GCS bucket (e.g. "day2/t1").
        bucket_name (str): The name of the Google Cloud Storage bucket.

    Returns:
        A list of tuples: [(filename, submission_dict), ...]
            where submission_dict has keys "mask" and "params".
    """
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE
    )
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    # List all objects under the given prefix
    blobs = bucket.list_blobs(prefix=prefix)

    submissions = []
    for blob in blobs:
        # Only process .json files
        if blob.name.endswith(".json"):
            try:
                json_str = blob.download_as_text()
                # Convert the JSON content to a dict
                submission_dict = ast.literal_eval(
                    json_str
                )  # or use `json.loads(json_str)`

                # Ensure it has the keys we need
                if "mask" in submission_dict and "params" in submission_dict:
                    submissions.append((blob.name, submission_dict))
                else:
                    print(f"Skipping {blob.name} - missing 'mask' or 'params'.")
            except Exception as e:
                print(f"Error parsing {blob.name}: {e}")

    return submissions


def get_test_data() -> List[Dict[str, Any]]:
    """
    Generate test data for evaluating student solutions.
    """
    ic_test: List[List[float]] = [
        [0.05, 15.0, 0.0, 2.5],
        [0.4, 3.0, 0.0, 0.2],
    ]
    print("Generating testing data...")
    test_data = generate_test_data(test_conditions=ic_test, true_model=true_model_day3)
    return test_data


def evaluate_all_student_algorithms(
    submissions: List[Tuple[str, Dict[str, Any]]],
    test_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each JSON submission, extract 'mask' and 'params', compute RMSE on test_data,
    and return a list of dicts with {"algorithm": <filename>, "rmse": <float>}.
    """
    overall_results: List[Dict[str, Any]] = []

    for file_name, submission_data in submissions:
        mask = submission_data["mask"]
        params = submission_data["params"]

        # Evaluate the submission
        rmse, variable_rmse, test_predictions = evaluate_student_solution(
            mask, params, test_data
        )

        # For the "algorithm" name on the leaderboard, we'll use the filename
        overall_results.append({"algorithm": file_name, "rmse": rmse})

    # Sort results by RMSE
    overall_results.sort(key=lambda x: x["rmse"])

    print("\nLeaderboard (Algorithm Name : Overall RMSE):")
    for result in overall_results:
        print(f"{result['algorithm']} : {result['rmse']:.4f}")

    return overall_results


def generate_leaderboard_html(overall_results: List[Dict[str, Any]]) -> str:
    """
    Generates an HTML string representing the leaderboard in a nicely formatted table.
    """
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Leaderboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h2 { text-align: center; }
        table { border-collapse: collapse; margin: auto; width: 80%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even){background-color: #f9f9f9;}
    </style>
</head>
<body>
    <h2>Leaderboard (Filename : Overall RMSE)</h2>
    <table>
        <tr>
            <th>Submission</th>
            <th>Overall RMSE</th>
        </tr>
"""
    for result in overall_results:
        html += f"        <tr><td>{result['algorithm']}</td><td>{result['rmse']:.4f}</td></tr>\n"
    html += """    </table>
</body>
</html>
"""
    return html


def run_benchmark(prefix: str = "", file_name: str = "leaderboard.html") -> None:
    """
    Loads all .json submissions from the GCS prefix, evaluates them on test data,
    generates and uploads a leaderboard HTML to GCS.
    """
    # 1. Load all JSON submissions from GCS
    submissions = load_json_submissions(prefix, bucket_name="ddo_hackathon")
    print(f"Submissions loaded for prefix '{prefix}': {[s[0] for s in submissions]}")

    if not submissions:
        print("No JSON submissions found. Exiting benchmark for this track.")
        return

    # 2. Get test data
    test_data = get_test_data()

    # 3. Evaluate each submission
    overall_results = evaluate_all_student_algorithms(submissions, test_data)

    # 4. Generate and upload leaderboard
    if overall_results:
        html_content = generate_leaderboard_html(overall_results)
    else:
        html_content = "No results to display in the leaderboard."

    upload_to_bucket(html_content, file_name=f"{prefix}/{file_name}.html")


if __name__ == "__main__":
    # Example usage for Track 1
    print("\nStarting leaderboard generation for Track 1...")
    run_benchmark(prefix="day3/t1", file_name="leaderboard_t1")
    print("Track 1 leaderboard update complete.")

    # Example usage for Track 2
    print("\nStarting leaderboard generation for Track 2...")
    run_benchmark(prefix="day3/t2", file_name="leaderboard_t2")
    print("Track 2 leaderboard update complete.")
