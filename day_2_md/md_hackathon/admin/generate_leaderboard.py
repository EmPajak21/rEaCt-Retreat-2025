import os
import sys
import time
import multiprocessing
from typing import Dict, Any, List, Callable, Optional, Tuple
import pickle
import numpy as np

# Required imports for multiprocessing with dill serialization.
import dill
import multiprocessing.reduction as reduction

reduction.ForkingPickler.dumps = dill.dumps

# Add parent directories to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hackathon_utils import load_student_algorithms, upload_to_bucket

from bio_model import (
    evaluate_student_solution,
    generate_test_data,
    true_model_day2,
    fitness_function,
    candidate_models,
)


RESULTS_DIR: str = "day_2_md/md_hackathon/admin/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CREDENTIALS_FILE: str = (
    "day_1_ddo/ddo_hackathon/intense-pixel-446617-e2-9a9d3fd50dd4.json"
)


def get_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate or load training and test data.
    """
    print("Generating/loading training data...")
    with open("day_2_md/training_data_day2.pickle", "rb") as handle:
        training_data = pickle.load(handle)

    ic_test: List[List[float]] = [
        [0.05, 15.0, 0.0, 2.5],
        [0.4, 3.0, 0.0, 0.2],
    ]
    test_data = generate_test_data(test_conditions=ic_test, true_model=true_model_day2)
    return training_data, test_data


def generate_leaderboard_html(overall_results: List[Dict[str, Any]]) -> str:
    """
    Generates an HTML string representing the leaderboard in a nicely formatted table.
    """
    if not overall_results:
        return "<h2>No results to display in the leaderboard.</h2>"

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
    <h2>Leaderboard (Algorithm Name : Overall RMSE)</h2>
    <table>
        <tr>
            <th>Algorithm</th>
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


def run_student_algorithm_t2(
    serialized_alg: bytes,
    training_data: List[Dict[str, Any]],
    best_container: Dict[str, Any],
    candidate_models: Callable,
    fitness_function: Callable,
) -> None:
    """
    Child process wrapper that deserializes and runs the student algorithm.
    Used in T2 for timeouts.
    """
    import dill  # Ensure dill is imported in the child process.
    import sys

    original_stdout = sys.stdout
    student_alg: Callable = dill.loads(serialized_alg)
    print("[Child] Student algorithm started.")
    # Run the student algorithm as is.
    student_alg(
        training_data=training_data,
        candidate_models=candidate_models,
        basic_fitness_function=fitness_function,
        best_container=best_container,
    )
    print("[Child] Student algorithm finished.")
    sys.stdout = original_stdout


def run_with_timeout_t2(
    training_data: List[Dict[str, Any]], serialized_alg: bytes, timeout: int
) -> Optional[Dict[str, Any]]:
    """
    Runs the student algorithm in a separate process with a timeout (for T2).
    """
    manager = multiprocessing.Manager()
    best_container: Dict[str, Any] = manager.dict()

    process = multiprocessing.Process(
        target=run_student_algorithm_t2,
        kwargs={
            "serialized_alg": serialized_alg,
            "training_data": training_data,
            "candidate_models": candidate_models,
            "fitness_function": fitness_function,
            "best_container": best_container,
        },
    )

    process.start()
    start_time: float = time.time()
    last_best: Optional[Dict[str, Any]] = None

    # Poll the shared dict until timeout.
    while time.time() - start_time < timeout:
        if "best" in best_container:
            last_best = best_container["best"]
        if not process.is_alive():
            break
        time.sleep(0.1)

    if process.is_alive():
        print("[Parent] Timeout reached: terminating the process.")
        process.terminate()
        process.join()

    print("[Parent] Retrieved best_container (from shared dict):", last_best)
    return last_best


def test_t2(
    algorithms_test: List[Callable],
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    timeout: int = 5,
) -> List[Dict[str, Any]]:
    """
    T2 approach: Evaluate student algorithms with multiprocessing + timeouts.
    """
    import dill

    overall_results: List[Dict[str, Any]] = []
    for idx, student_alg in enumerate(algorithms_test):
        try:
            serialized_alg: bytes = dill.dumps(student_alg)
        except Exception as e:
            print(f"Error serializing algorithm {idx}: {e}")
            continue

        best_individual = run_with_timeout_t2(
            training_data=training_data,
            serialized_alg=serialized_alg,
            timeout=timeout,
        )

        if best_individual is None:
            print(f"No best individual found for algorithm {idx} (timeout or error).")
            continue

        rmse, variable_rmse, test_predictions = evaluate_student_solution(
            best_individual.get("mask"), best_individual["params"], test_data
        )
        overall_results.append({"algorithm": student_alg.__name__, "rmse": rmse})

    # Sort and print
    overall_results.sort(key=lambda x: x["rmse"])
    print("\nT2 Leaderboard (Algorithm Name : Overall RMSE):")
    for result in overall_results:
        print(f"{result['algorithm']} : {result['rmse']:.4f}")

    return overall_results


def test_t1(
    algorithms_test: List[Callable],
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    T1 approach: Evaluate student algorithms without multiprocessing/timeouts.
    Here, each `student_alg` is expected to be a function that follows
    the structure of local_search_optimize(params_start, masks_start, training_data, ...).
    """
    overall_results: List[Dict[str, Any]] = []

    for idx, student_alg in enumerate(algorithms_test):
        try:
            # Create random initial guess for continuous parameters and binary masks.
            params_start = np.random.uniform(-1, 1, size=5)
            masks_start = np.random.randint(0, 2, size=9)

            # Call the student's local search algorithm
            best_obj_value, best_params, best_masks = student_alg(
                params_start,
                masks_start,
                training_data,
                iterations=1000,  # You can tweak these hyperparameters
                samples=5,
            )

            # Evaluate the final solution with your test data
            rmse, variable_rmse, test_predictions = evaluate_student_solution(
                best_masks, best_params, test_data
            )

            overall_results.append(
                {
                    "algorithm": student_alg.__name__,
                    "rmse": rmse,
                }
            )

            print(
                f"[T1] Completed {student_alg.__name__}: RMSE={rmse:.4f}, "
                f"best_obj={best_obj_value:.4f}"
            )

        except Exception as e:
            print(f"Error running algorithm {idx} directly (T1): {e}")

    # Sort by RMSE (ascending)
    overall_results.sort(key=lambda x: x["rmse"])

    # Print out a simple console leaderboard
    print("\nT1 Leaderboard (Algorithm Name : Overall RMSE):")
    for result in overall_results:
        print(f"{result['algorithm']} : {result['rmse']:.4f}")

    return overall_results


def run_benchmark(track: str, prefix: str, file_name: str) -> None:
    """
    High-level function that loads algorithms, loads data, and
    delegates to either T1 or T2 approach based on `track`.
    """
    # Choose function name based on track.
    if track.lower() == "t1":
        func_name = "local_search_optimize"
    elif track.lower() == "t2":
        func_name = "genetic_algorithm"
    else:
        print(f"Unknown track '{track}'. Exiting benchmark.")
        return

    # Load student algorithms with the selected function name.
    algorithms_test = load_student_algorithms(
        bucket_name="ddo_hackathon", gcloud_path=prefix, func_name=func_name
    )
    print(f"Algorithms loaded for prefix '{prefix}':", algorithms_test)
    if not algorithms_test:
        print("No valid student algorithms found. Exiting benchmark for this track.")
        return

    training_data, test_data = get_data()

    if track.lower() == "t1":
        overall_results = test_t1(algorithms_test, training_data, test_data)
    elif track.lower() == "t2":
        overall_results = test_t2(algorithms_test, training_data, test_data, timeout=5)

    html_content = generate_leaderboard_html(overall_results)
    upload_to_bucket(html_content, file_name=f"{prefix}/{file_name}.html")


if __name__ == "__main__":
    # # Example: run T1
    print("Starting leaderboard generation for Track 1...")
    run_benchmark(track="t1", prefix="day2/t1", file_name="leaderboard_t1")
    print("Track 1 leaderboard update complete.")

    # Example: run T2
    # print("\nStarting leaderboard generation for Track 2...")
    # run_benchmark(track="t2", prefix="day2/t2", file_name="leaderboard_t2")
    # print("Track 2 leaderboard update complete.")
