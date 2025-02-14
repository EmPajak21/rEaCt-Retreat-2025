import os
import sys
import unittest
import inspect
from typing import List, Dict, Any, Callable

# Add parent directories to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from bio_model import candidate_models, fitness_function


class TestFrameworkT1(unittest.TestCase):
    def test_local_search_function_exists(self):
        """Check if the student submission has 'local_search_optimize'."""
        try:
            from student_submission import local_search_optimize
        except ImportError:
            self.fail("Function `local_search_optimize` not found in your submission!")


    # def test_local_search_basic_run(self):
    #     """
    #     Test a basic run of local_search_optimize on a small dummy problem
    #     to ensure it returns the correct types and shapes.
    #     """
    #     # Some minimal dummy data you might pass to local_search_optimize
    #     # Adjust keys/structure to match what fitness_function expects
    #     training_data = [
    #         {
    #             "IC": [0.0, 0.0],  # Example initial condition
    #             "time": [0, 1, 2],  # Example time steps
    #             "observations": [0.0, 0.1, 0.15],  # Dummy data
    #         }
    #     ]

    #     # Small 2D parameter space and a 2D mask for demonstration
    #     params_start = [0.5, -0.5]
    #     masks_start = [1, 0]

    #     # Retrieve the student's local_search_optimize
    #     local_search = student_submission.local_search_optimize

    #     # Call local_search_optimize on a small iteration count to not hang
    #     best_obj_value, best_params, best_masks = local_search(
    #         params_start=params_start,
    #         masks_start=masks_start,
    #         training_data=training_data,
    #         iterations=5,  # drastically reduced for fast test
    #         samples=2,  # smaller sample
    #     )

    #     # Basic type checks
    #     self.assertIsInstance(
    #         best_obj_value, float, "Expected best_obj_value to be a float."
    #     )
    #     self.assertIsInstance(
    #         best_params, np.ndarray, "Expected best_params to be a numpy.ndarray."
    #     )
    #     self.assertIsInstance(
    #         best_masks, np.ndarray, "Expected best_masks to be a numpy.ndarray."
    #     )

    #     # Check dimension
    #     self.assertEqual(
    #         best_params.shape, (2,), "Expected best_params to be shape (2,)."
    #     )
    #     self.assertEqual(
    #         best_masks.shape, (2,), "Expected best_masks to be shape (2,)."
    #     )


if __name__ == "__main__":
    unittest.main()
