import os
import sys
import unittest
import inspect

# Add parent directories to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)


class TestFrameworkT1(unittest.TestCase):
    def test_local_search_function_exists(self):
        """Check if the student submission has 'local_search_optimize'."""
        try:
            from student_submission import local_search_optimize
        except ImportError:
            self.fail("Function `local_search_optimize` not found in your submission!")

    def test_local_search_optimize_parameter_names(self):
        """Check if the function has the correct parameter names (ignoring type annotations)."""
        from student_submission import local_search_optimize
        sig = inspect.signature(local_search_optimize)
        param_names = list(sig.parameters.keys())
        expected_param_names = [
            "params_start",
            "masks_start",
            "training_data",
            "iterations",
            "samples"
        ]
        self.assertEqual(
            param_names,
            expected_param_names,
            f"Expected parameters {expected_param_names}, but got {param_names}"
        )


if __name__ == "__main__":
    unittest.main()