import os
import sys
import time
import multiprocessing
from typing import Dict, Any, List, Callable, Optional, Tuple

# Required imports for multiprocessing with dill serialization.
import dill
import multiprocessing.reduction as reduction

reduction.ForkingPickler.dumps = dill.dumps

# Add parent directories to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hackathon_utils import load_student_algorithms, upload_to_bucket

from bio_model import (
    evaluate_student_solution,
    generate_training_data,
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


def run_student_algorithm(
    serialized_alg: bytes,
    training_data: List[Dict[str, Any]],
    best_container: Dict[str, Any],
    candidate_models: Callable = candidate_models,
    fitness_function: Callable = fitness_function,
) -> None:
    """
    Child process wrapper that deserializes and runs the student algorithm.
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


def run_genetic_algorithm_with_timeout(
    training_data: List[Dict[str, Any]],
    serialized_alg: bytes,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Runs the student algorithm in a separate process with a timeout.

    Polls both a shared dictionary to capture the latest "best" update.
    When the timeout is reached, the child is terminated,
    and the most recent update is returned.
    """
    manager = multiprocessing.Manager()
    best_container: Dict[str, Any] = manager.dict()

    process = multiprocessing.Process(
        target=run_student_algorithm,
        kwargs={
            "serialized_alg": serialized_alg,
            "training_data": training_data,
            "fitness_function": fitness_function,
            "candidate_models": candidate_models,
            "best_container": best_container,
        },
    )

    process.start()
    start_time: float = time.time()
    last_best: Optional[Dict[str, Any]] = None

    # Poll output_queue and best_container until timeout.
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

    print(
        "[Parent] Retrieved best_container (from shared dict):",
        last_best,
    )
    return last_best


def get_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate training and test data.
    """
    ic_train: List[List[float]] = [
        [0.1, 10.0, 0.0, 1.0],
        [0.2, 8.0, 0.0, 2.0],
        [0.3, 6.0, 0.0, 0.5],
        [0.3, 8.0, 0.2, 1.0],
    ]
    print("Generating training data...")
    training_data, training_df = generate_training_data(
        initial_conditions=ic_train, true_model=true_model_day2
    )
    ic_test: List[List[float]] = [
        [0.05, 15.0, 0.0, 2.5],
        [0.4, 3.0, 0.0, 0.2],
    ]
    test_data = generate_test_data(test_conditions=ic_test, true_model=true_model_day2)
    return training_data, test_data


def evaluate_all_student_algorithms(
    algorithms_test: List[Callable],
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each student algorithm, serialize it, run it with a timeout,
    evaluate its solution, and print the overall results.
    Returns a list of dictionaries containing algorithm names and their RMSE.
    """
    overall_results: List[Dict[str, Any]] = []
    for idx, student_alg in enumerate(algorithms_test):
        try:
            serialized_alg: bytes = dill.dumps(student_alg)
        except Exception as e:
            print(f"Error serializing algorithm {idx}: {e}")
            continue

        best_individual: Optional[Dict[str, Any]] = run_genetic_algorithm_with_timeout(
            training_data=training_data,
            serialized_alg=serialized_alg,
            timeout=5,  # Adjust timeout as needed (e.g., 5 seconds)
        )

        if best_individual is None:
            print(f"No best individual found for algorithm {idx} (timeout or error).")
            continue

        rmse, variable_rmse, test_predictions = evaluate_student_solution(
            best_individual.get("mask"), best_individual["params"], test_data
        )

        overall_results.append({"algorithm": student_alg.__name__, "rmse": rmse})

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


def run_benchmark(prefix: str = "", file_name: str = "leaderboard.html") -> None:
    algorithms_test = load_student_algorithms(
        bucket_name="ddo_hackathon", gcloud_path=prefix, func_name="genetic_algorithm"
    )
    print(f"Algorithms loaded for prefix '{prefix}':", algorithms_test)
    if not algorithms_test:
        print("No valid student algorithms found. Exiting benchmark for this track.")
        return

    training_data, test_data = get_data()
    overall_results = evaluate_all_student_algorithms(
        algorithms_test, training_data, test_data
    )

    if overall_results:
        html_content = generate_leaderboard_html(overall_results)
    else:
        html_content = "No results to display in the leaderboard."

    upload_to_bucket(html_content, file_name=f"{prefix}/{file_name}.html")


if __name__ == "__main__":
    # # Generate and upload leaderboard for Track 1
    # print("Starting leaderboard generation for Track 1...")
    # run_benchmark(prefix="day2/t1", file_name="leaderboard_t1")
    # print("Track 1 leaderboard update complete.")

    # Generate and upload leaderboard for Track 2
    print("\n\n Starting leaderboard generation for Track 2...")
    run_benchmark(prefix="day2/t1", file_name="leaderboard_t1")
    print("Track 2 leaderboard update complete.")
