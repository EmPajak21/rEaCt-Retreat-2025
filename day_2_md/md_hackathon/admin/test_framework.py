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

# Now you can do:
import bio_model
from bio_model import candidate_models, fitness_function

class TestGeneticAlgorithm(unittest.TestCase):
    def test_genetic_algorithm_signature(self):
        """
        Test that the function `genetic_algorithm` is defined with the expected signature:
        
            def genetic_algorithm(
                training_data: List[Dict[str, Any]],
                candidate_models: Callable = candidate_models,
                basic_fitness_function: Callable = fitness_function,
                best_container: Dict[str, Any] = None
            ) -> Dict[str, Any]:
        
        Also, ensure that `population_size` is not an input argument.
        """
        try:
            from student_submission import genetic_algorithm
        except ImportError:
            self.fail("Function `genetic_algorithm` not found in your submission!")
        
        sig = inspect.signature(genetic_algorithm)
        params = sig.parameters
        
        # Verify that 'training_data' exists.
        self.assertIn("training_data", params, "Missing parameter 'training_data'")
        
        # Expected parameters and their default values.
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
                f"Default value for '{name}' must be {default_val}"
            )
        

        # Optionally, verify the return annotation if provided.
        expected_return = Dict[str, Any]
        if sig.return_annotation is not inspect.Signature.empty:
            self.assertEqual(
                sig.return_annotation,
                expected_return,
                "Return annotation should be Dict[str, Any]"
            )
    
    def test_global_population_size(self):
        """
        Test that there is a global variable named POPULATION_SIZE defined in the student's submission,
        and that it is an integer with a value less than or equal to 20.
        """
        import student_submission
        
        self.assertTrue(hasattr(student_submission, "POPULATION_SIZE"),
                        "Global variable 'POPULATION_SIZE' is not defined in your submission!")
        pop_size = getattr(student_submission, "POPULATION_SIZE")
        self.assertIsInstance(pop_size, int, "POPULATION_SIZE should be an integer!")
        self.assertLessEqual(pop_size, 20, "POPULATION_SIZE must be less than or equal to 20!")
    

if __name__ == "__main__":
    unittest.main()
