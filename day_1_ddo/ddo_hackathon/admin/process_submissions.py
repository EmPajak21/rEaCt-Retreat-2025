import os
import sys
import importlib.util

# Add bm_routine (the parent directory) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports benchmarking routine and problem
from benchmarking import *

SUBMISSIONS_DIR = "admin/submissions/ddo_hackathon"
RESULTS_DIR = "admin/results"

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Function to process submissions and run your_alg from each
def load_student_algorithms(submissions_dir):
    algorithms = []
    for file_name in os.listdir(submissions_dir):
        if file_name.endswith(".py"):
            file_path = os.path.join(submissions_dir, file_name)
            module_name = file_name[:-3]  # Remove the .py extension
            
            try:
                # Dynamically import the submission as a module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for the presence of the "your_alg" function
                if hasattr(module, "your_alg"):
                    algorithms.append(module.your_alg)
                    print(f"Loaded your_alg from {file_name}")
                else:
                    print(f"Warning: {file_name} does not contain a your_alg function")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    return algorithms

def run_benchmark():
    # Load all student algorithms
    algorithms_test = load_student_algorithms(SUBMISSIONS_DIR)
    print(algorithms_test)
    if not algorithms_test:
        print("No valid your_alg functions found. Exiting.")
        return
    
    # Define additional parameters for ML4CE_uncon_eval
    home_dir = "01_hackathon/admin/results"
    N_x_l = [4]
    f_eval_l = [100]
    functions_test = ["Rosenbrock_f"]
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

if __name__ == "__main__":
    run_benchmark()

