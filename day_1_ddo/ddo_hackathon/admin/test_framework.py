import os
import sys
import unittest
import numpy as np

# Add the parent directory (bm_routine) to the Python path so the module can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the benchmarking routine and problem definitions.
from benchmarking import Test_function


class TestStudentSubmission(unittest.TestCase):
    """
    Unit test suite for validating the student's submission.

    The student's code must define a function `your_alg` with the following signature:

        your_alg(f, x_dim, bounds, iter_tot)

    The function should return:
      - opt_x: a NumPy array,
      - opt_fun: a number (int or float),
      - team_name: a list containing exactly one team name,
      - names: a list containing at least one name.
    """

    def setUp(self):
        """
        Set up a test environment with a sample problem.
        """
        self.x_dim = 2
        # Generate a random shift array with values in the interval [-3, 3] for the given dimensions.
        randShift_l = np.random.uniform(-3, 3, (1, self.x_dim))
        # Define bounds for each dimension.
        self.bounds = np.array([(0, 2), (0, 2)])
        # Initialize the test function from the benchmarking module.
        self.f = Test_function(
            "Rosenbrock_f", self.x_dim, True, randShift_l, self.bounds
        )
        # Define the total number of iterations for the optimization process.
        self.iter_tot = 50

    def test_function_signature(self):
        """
        Test that the function `your_alg` is defined in the student's submission.
        """
        try:
            from student_submission import your_alg  # noqa: F401
        except ImportError:
            self.fail("Function `your_alg` not found in your submission!")

    def test_function_output(self):
        """
        Test the output types and constraints from `your_alg`.

        Expects:
            - opt_x: a numpy.ndarray,
            - opt_fun: an int or float,
            - team_name: a list with exactly one element,
            - names: a list with at least one element.
        """
        from student_submission import your_alg

        opt_x, opt_fun, team_name, names = your_alg(
            self.f, self.x_dim, self.bounds, self.iter_tot
        )

        # Check that opt_x is a numpy array.
        self.assertIsInstance(opt_x, np.ndarray, "opt_x must be a numpy array!")
        # Check that opt_fun is a number.
        self.assertIsInstance(opt_fun, (int, float), "opt_fun must be a number!")
        # Check that team_name is a list with exactly one element.
        self.assertEqual(len(team_name), 1, "You must include exactly one team name!")
        # Check that names is a list with at least one element.
        self.assertGreaterEqual(len(names), 1, "Include at least one name!")


if __name__ == "__main__":
    unittest.main()
