import os
import sys

# Add bm_routine and hackathon_utils (the parent directory) to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports (after sys.path is modified).
from hackathon_utils import load_student_algorithms, upload_to_bucket
from benchmarking import *

# Update the credentials path if needed (use forward slashes for cross-platform compatibility)
CREDENTIALS_FILE = "day_1_ddo/ddo_hackathon/intense-pixel-446617-e2-9a9d3fd50dd4.json"

# Ensure the results directory exists
RESULTS_DIR = "day_1_ddo/ddo_hackathon/admin/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_benchmark(prefix=""):
    # Load all student algorithms
    algorithms_test = load_student_algorithms(
        bucket_name="ddo_hackathon", gcloud_path=prefix, func_name="your_alg"
    )
    print(algorithms_test)
    if not algorithms_test:
        print("No valid your_alg functions found. Exiting benchmark for this track.")
        return None

    # Define additional parameters for ML4CE_uncon_eval
    home_dir = ""
    N_x_l = [10]
    functions_test = ["Rosenbrock_f"]
    reps = 10

    # Track-specific functione evaluation budget
    if prefix.endswith("1"):
        f_eval_l = [200]
    elif prefix.endswith("2"):
        f_eval_l = [100]

    # Run the benchmark
    try:
        info, trajectories, timestamp = ML4CE_uncon_eval(
            N_x_l=N_x_l,
            f_eval_l=f_eval_l,
            functions_test=functions_test,
            algorithms_test=algorithms_test,
            reps=reps,
            home_dir=home_dir,
            SafeData=False,
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
        print(f"Benchmark completed successfully for prefix {prefix}")
    except Exception as e:
        print(f"Error running benchmark for prefix {prefix}: {e}")
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
