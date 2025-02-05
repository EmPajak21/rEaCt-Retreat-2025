import os
import sys
import unittest
import inspect
from typing import List, Dict, Any

# Add the parent directory to sys.path so the student_submission module can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestGeneticAlgorithm(unittest.TestCase):
    def test_genetic_algorithm_signature(self):
        """
        Test that the function `genetic_algorithm` is defined with the required signature:
        
            def genetic_algorithm(
                training_data: List[Dict[str, Any]],
                *,
                generations: int = 10,
                population_size: int = 30,
                mutation_rate: float = 0.1,
                best_container: Dict[str, Any] = None
            ) -> Dict[str, Any]:
        """
        try:
            from student_submission import genetic_algorithm
        except ImportError:
            self.fail("Function `genetic_algorithm` not found in your submission!")
        
        sig = inspect.signature(genetic_algorithm)
        params = sig.parameters

        # Verify that the first parameter 'training_data' exists.
        self.assertIn("training_data", params, "Missing parameter 'training_data'")
        
        # Expected keyword-only parameters and their default values.
        expected_kwargs = {
            "generations": 10,
            "population_size": 30,
            "mutation_rate": 0.1,
            "best_container": None,
        }
        for name, default in expected_kwargs.items():
            self.assertIn(name, params, f"Missing parameter '{name}'")
            param = params[name]
            # Ensure the parameter is keyword-only.
            self.assertEqual(
                param.kind,
                inspect.Parameter.KEYWORD_ONLY,
                f"Parameter '{name}' should be keyword-only"
            )
            # Check that the default value is as expected.
            self.assertEqual(
                param.default,
                default,
                f"Default value for '{name}' must be {default}"
            )
        
        # Optionally check the return annotation if provided.
        expected_return = Dict[str, Any]
        if sig.return_annotation is not inspect.Signature.empty:
            self.assertEqual(
                sig.return_annotation,
                expected_return,
                "Return annotation should be Dict[str, Any]"
            )
    
    def test_genetic_algorithm_output(self):
        """
        Test that `genetic_algorithm` returns a dictionary.
        """
        from student_submission import genetic_algorithm
        
        # Create a dummy training_data list (adjust as needed for a minimal valid structure).
        dummy_training_data: List[Dict[str, Any]] = []
        # Create a dummy best_container dictionary.
        dummy_best_container: Dict[str, Any] = {}
        
        result = genetic_algorithm(
            dummy_training_data,
            generations=10,
            population_size=30,
            mutation_rate=0.1,
            best_container=dummy_best_container,
        )
        self.assertIsInstance(result, dict, "genetic_algorithm should return a dictionary")

if __name__ == "__main__":
    unittest.main()
