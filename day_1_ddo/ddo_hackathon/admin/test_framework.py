import unittest
import numpy as np
import sys
import os

# Add bm_routine (the parent directory) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports benchmarking routine and problem
from benchmarking import *

class TestStudentSubmission(unittest.TestCase):
    def setUp(self):
        self.x_dim = 2
        randShift_l = np.random.uniform(-3, 3, (1, self.x_dim))
        self.bounds = np.array([(0, 2), (0, 2)])

        self.f = Test_function("Rosenbrock_f", self.x_dim, True, randShift_l, self.bounds)
        self.iter_tot = 50

    def test_function_signature(self):
        try:
            from student_submission import your_alg
        except ImportError:
            self.fail("Function `your_alg` not found in your submission!")

    def test_function_output(self):
        from student_submission import your_alg
        opt_x, opt_fun, team_name, names = your_alg(self.f, self.x_dim, self.bounds, self.iter_tot)

        self.assertIsInstance(opt_x, np.ndarray, "opt_x must be a numpy array!")
        self.assertIsInstance(opt_fun, (int, float), "opt_fun must be a number!")
        self.assertEqual(len(team_name), 1, "You must include one team name!")
        self.assertGreaterEqual(len(names), 1, "Include at least one name!")
