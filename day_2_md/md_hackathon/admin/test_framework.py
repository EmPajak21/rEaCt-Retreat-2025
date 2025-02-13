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


class TestGeneticAlgorithm(unittest.TestCase):
    def test_genetic_algorithm_signature(self):
        """
        Test that the function `genetic_algorithm` is defined with the expected signature:

            def genetic_algorithm(
                training_data: List[Dict[str, Any]],
                population_size: int = 20,
                candidate_models: Callable = candidate_models,
                basic_fitness_function: Callable = fitness_function,
                best_container: Dict[str, Any] = None
            ) -> Dict[str, Any]:
        """
        try:
            from student_submission import genetic_algorithm
        except ImportError:
            self.fail("Function `genetic_algorithm` not found in your submission!")

        sig = inspect.signature(genetic_algorithm)
        params = sig.parameters

        # Check required parameter: training_data
        self.assertIn("training_data", params, "Missing parameter 'training_data'")

        # Check required parameter: population_size
        self.assertIn("population_size", params, "Missing parameter 'population_size'")

        # Verify default value of population_size
        population_size_default = params["population_size"].default
        self.assertIsInstance(
            population_size_default,
            int,
            "Default for 'population_size' must be an integer!",
        )
        self.assertLessEqual(
            population_size_default, 20, "Default for 'population_size' must be <= 20!"
        )

        # Check that candidate_models, basic_fitness_function, and best_container have the expected defaults.
        expected_defaults = {
            "candidate_models": candidate_models,
            "basic_fitness_function": fitness_function,
            "best_container": None,
        }

        for name, default_val in expected_defaults.items():
            self.assertIn(name, params, f"Missing parameter '{name}'")
            param = params[name]
            self.assertEqual(
                param.default,
                default_val,
                f"Default value for '{name}' must be {default_val}",
            )

        # Optionally, verify the return annotation if provided.
        expected_return = Dict[str, Any]
        if sig.return_annotation is not inspect.Signature.empty:
            self.assertEqual(
                sig.return_annotation,
                expected_return,
                "Return annotation should be Dict[str, Any]",
            )


if __name__ == "__main__":
    unittest.main()
