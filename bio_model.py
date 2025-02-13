import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from typing import List, Dict, Tuple, Any, Union, Optional, Callable
import random 

# Constants
TERM_NAMES = [
    "Monod Growth",
    "Hill Kinetics Growth",
    "Substrate Inhibition Factor",
    "Product Inhibition Factor (Competitive)",
    "Non-Competitive Product Inhibition",
    "Competitive Inhibition Factor",
    "Double Substrate Limited Factor",
    "Substrate Threshold Activation",
    "Inhibitor Saturation",
]

DEFAULT_PARAMS = {
    "mu_max": 0.5,
    "Ks": 2.0,
    "Kp": 1.0,
    "Yxs": 0.4,
    "Ki": np.nan,
}


DAY3_PARAMS = {
    "mu_max": 0.8,  # Changed from 0.5
    "Ks": 3.0,  # Changed from 2.0
    "Kp": 2.0,  # Changed from 1.0
    "Yxs": 0.5,  # Changed from 0.4
    "Ki": 1.5,  # New parameter
}


def true_model_day2(
    y: Union[List[float], np.ndarray],
    t: Union[float, np.ndarray],
) -> List[float]:
    """
    True model using candidate_models with fixed parameters.

    Parameters
    ----------
    y : Union[List[float], np.ndarray]
        State variables [X, S, P, I] representing biomass, substrate,
        product, and inhibitor concentrations.
    t : Union[float, np.ndarray]
        Time point(s) at which to evaluate the model.

    Returns
    -------
    List[float]
        Derivatives [dX/dt, dS/dt, dP/dt, dI/dt] at the given time point.
    """
    mask = [1, 0, 0, 1, 0, 0, 0, 0, 0]
    params = list(DEFAULT_PARAMS.values())
    dX, dS, dP, dI = candidate_models(y=y, t=t, params=params, mask=mask)
    return [dX, dS, dP, dI]


def true_model_day3(y, t):
    """
    True model using candidate_models with fixed parameters.

    Parameters
    ----------
    y : Union[List[float], np.ndarray]
        State variables [X, S, P, I] representing biomass, substrate,
        product, and inhibitor concentrations.
    t : Union[float, np.ndarray]
        Time point(s) at which to evaluate the model.

    Returns
    -------
    List[float]
        Derivatives [dX/dt, dS/dt, dP/dt, dI/dt] at the given time point.
    """
    mask = [0, 1, 1, 0, 1, 0, 0, 0, 0]

    # Modified true mechanism: Hill kinetics with substrate inhibition and non-competitive product inhibition
    params = list(DAY3_PARAMS.values())
    dX, dS, dP, dI = candidate_models(y=y, t=t, params=params, mask=mask)
    return [dX, dS, dP, dI]


def candidate_models(
    y: Union[List[float], np.ndarray],
    t: Union[float, np.ndarray],
    params: List[float],
    mask: List[int],
) -> List[float]:
    """
    Calculate derivatives based on selected growth mechanisms.

    Parameters
    ----------
    y : Union[List[float], np.ndarray]
        State variables [X, S, P, I] representing biomass, substrate,
        product, and inhibitor concentrations.
    t : Union[float, np.ndarray]
        Time point(s) at which to evaluate the model.
    params : List[float]
        Model parameters [mu_max, Ks, Ki, Yxs, Kp].
    mask : List[int]
        Binary mask indicating which growth terms to include.

    Returns
    -------
    List[float]
        Derivatives [dX/dt, dS/dt, dP/dt, dI/dt] at the given time point.
    """
    X, S, P, I = y
    mu_max, Ks, Kp, Yxs, Ki = params

    growth = mu_max * X
    # Growth terms
    # Monod Growth
    if mask[0]:
        growth *= S / (Ks + S)

    # Hill Kinetics Growth
    if mask[1]:
        n = 2
        growth *= S**n / (Ks**n + S**n)

    # Substrate Inhibition Factor
    if mask[2]:
        growth *= 1 / (1 + S / Ki)

    # Product Inhibition Factor (Competitive)
    if mask[3]:
        growth *= (Ks + S) / (S + Ks + (Ks * P / Kp)) # This term is modified so that is can be combined with mask 0 or 1 for a correct inhibition

    # Non-Competitive Product Inhibition
    if mask[4]:
        growth *= 1 / (1 + P / Kp)

    # Competitive Inhibition Factor
    if mask[5]:
        growth *= 1 / (1 + I / Ki)

    # Double Substrate Limited Factor (Inhibitor is a second substrate in case of an inhibitor)
    if mask[6]:
        growth *= I / (Ki + I)

    # Substrate Threshold Activation
    if mask[7]:
        S_threshold = 0.5
        growth *= (S - S_threshold) / (Ks + (S - S_threshold)) if S > S_threshold else 0

    # Inhibitor Saturation
    if mask[8]:
        growth *= 1 / (1 + (P / (P + Ki)))

    # Calculate derivatives
    dX = growth
    dS = -(growth / Yxs)
    dP = 0.3 * growth
    dI = -0.1 * I

    return [dX, dS, dP, dI]


def generate_training_data(
    initial_conditions: Union[List[float], List[List[float]]],
    true_model: callable = true_model_day2,
    n_timepoints: int = 20,
    noise_level: float = 0.05,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Generate training data with noise.

    Parameters
    ----------
    initial_conditions : Union[List[float], List[List[float]]]
        Initial values for state variables [X, S, P, I].
    true_model : callable, optional
        Model function to generate data, by default true_model_day2.
    n_timepoints : int, optional
        Number of time points to generate, by default 20.
    noise_level : float, optional
        Standard deviation of noise relative to signal, by default 0.05.

    Returns
    -------
    List[Dict[str, Any]]
        Training data as list of dictionaries containing time points and noisy data

    Raises
    ------
    TypeError
        If initial_conditions is not a list.
    """
    if not isinstance(initial_conditions, list):
        raise TypeError("initial_conditions must be a list")

    if isinstance(initial_conditions[0], (int, float)):
        initial_conditions = [initial_conditions]

    t = np.linspace(0, 10, n_timepoints)
    training_data = []

    for i, y0 in enumerate(initial_conditions):
        # Generate clean data
        solution = odeint(true_model, y0, t)

        # Add noise
        noise = np.random.normal(0, noise_level, solution.shape)
        noisy_data = solution + noise * solution

        # Store data
        training_data.append({"t": t, "data": noisy_data, "initial_conditions": y0})

    return training_data


def generate_test_data(
    test_conditions: Union[List[float], List[List[float]]],
    true_model: callable = true_model_day2,
) -> List[Dict[str, Any]]:
    """
    Generate test data with specific conditions.

    Parameters
    ----------
    true_model : callable, optional
        Model function to generate data, by default true_model_day2.
    test_conditions : Union[List[float], List[List[float]]]
        Initial values for state variables [X, S, P, I].

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing time points and test data.
    """
    t = np.linspace(0, 15, 30)  # Longer time period
    test_data = []

    for y0 in test_conditions:
        solution = odeint(true_model, y0, t)
        test_data.append({"t": t, "data": solution, "initial_conditions": y0})

    return test_data


def fitness_function(
    mask: List[int],
    params: List[float],
    training_data: List[Dict[str, Any]],
) -> Tuple[float, Optional[List[float]]]:
    """
    Calculate fitness of a solution.

    Parameters
    ----------
    mask : List[int]
        Binary mask indicating which growth terms to include.
    params : List[float]
        Model parameters [mu_max, Ks, Ki, Yxs, Kp].
    training_data : List[Dict[str, Any]]
        Training data to evaluate fitness against.

    Returns
    -------
    Tuple[float, Optional[List[float]]]
        Tuple containing:
        - Negative total error (including complexity penalty)
        - List of errors for each experiment (or None if error occurred)
    """
    total_error = 0
    n_active_terms = sum(mask)

    errors_by_experiment = []
    for experiment in training_data:
        t = experiment["t"]
        y_true = experiment["data"]
        y0 = experiment["initial_conditions"]

        try:
            y_pred = odeint(
                lambda y, t: candidate_models(y, t, params, mask),
                y0,
                t,
            )
            error = np.mean((y_true - y_pred) ** 2)
            total_error += error
            errors_by_experiment.append(error)
        except:
            return float("-inf"), None

    complexity_penalty = 0.05 * n_active_terms

    return -(total_error + complexity_penalty), errors_by_experiment


def genetic_algorithm(
    training_data: List[Dict[str, Any]],
    *,
    generations: int = 10,
    population_size: int = 30,
    mutation_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Genetic algorithm that performs model selection and parameter estimation.

    Parameters
    ----------
    training_data : List[Dict[str, Any]]
        Training data to fit the model to.
    candidate_models : Callable
        Calculate derivatives based on selected growth mechanisms.
    basic_fitness_function : Callable
        Evaluates fitness of a candidate solution based on model performance.
    best_container : dict
        IMPORTANT! Stores the best candidate found.

    Returns
    -------
    Dict[str, Any]
        Best individual found (mask and parameters).
    """
    generations = 1000
    mutation_rate = 0.1

    if population_size % 2 != 0:
        raise ValueError("Population size must be even")

    n_params = 5

    # Initialize population
    population = [
        {
            "mask": np.random.randint(0, 2, size=9),
            "params": [
                np.random.uniform(0.1, 1.0),  # mu_max
                np.random.uniform(0.5, 5.0),  # Ks
                np.random.uniform(0.5, 5.0),  # Ki
                np.random.uniform(0.2, 0.8),  # Yxs
                np.random.uniform(0.5, 5.0),  # Kp
            ],
        }
        for _ in range(population_size)
    ]

    best_fitness = float("-inf")
    best_individual = None
    best_errors = None

    fitness_history = []
    param_history = []

    for gen in range(generations):
        # Evaluate fitness
        fitness_and_errors = [
            fitness_function(ind["mask"], ind["params"], training_data)
            for ind in population
        ]
        fitness = [f[0] for f in fitness_and_errors]
        errors = [f[1] for f in fitness_and_errors]

        # Update best individual
        current_best = max(fitness)
        current_best_idx = np.argmax(fitness)

        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[current_best_idx].copy()
            best_errors = errors[current_best_idx]

            print(f"\nGeneration {gen}")
            print(f"New best fitness: {best_fitness}")
            print("Parameters:", best_individual["params"])
            print(
                "Active terms:",
                [f"Term {i}" for i, m in enumerate(best_individual["mask"]) if m],
            )
            if best_errors:
                print("Errors by experiment:", best_errors)
            
            # Update the shared container, if provided
            if best_container is not None:
                best_container["best"] = best_individual
                
        # Track history
        fitness_history.append(best_fitness)
        param_history.append(
            best_individual["params"].copy() if best_individual else None
        )

        # Selection - keep top 50%
        sorted_indices = np.argsort(fitness)[::-1]
        population = [population[i] for i in sorted_indices[: population_size // 2]]

        # Create new population
        new_population = []
        for _ in range(population_size // 2):
            # Select parents
            parent1, parent2 = np.random.choice(population, 2, replace=False)

            # Create child
            child_mask = np.array(
                [
                    np.random.choice([parent1["mask"][j], parent2["mask"][j]])
                    for j in range(9)
                ]
            )

            # Parameter crossover
            child_params = []
            for p1, p2 in zip(parent1["params"], parent2["params"]):
                param = np.random.uniform(min(p1, p2), max(p1, p2))
                child_params.append(param)

            # Mutation
            if np.random.rand() < mutation_rate:
                child_mask[np.random.randint(0, 9)] ^= 1
                param_idx = np.random.randint(0, n_params)
                child_params[param_idx] *= np.random.uniform(0.8, 1.2)

            new_population.append({"mask": child_mask, "params": child_params})

        population.extend(new_population)

        if gen % 10 == 0:
            print(f"\nGeneration {gen} summary:")
            print(f"Best fitness: {best_fitness}")
            if best_individual:
                print("Current best parameters:", best_individual["params"])
                current_errors = fitness_function(
                    best_individual["mask"], best_individual["params"], training_data
                )[1]
                if current_errors:
                    print("Current errors by experiment:", current_errors)
        
        # Catch all masks zero i.e., no active leaves
        if sum(best_individual['mask']) == 0:
            random_idx = random.randint(0, len(best_individual['mask']) - 1)
            best_individual['mask'][random_idx] = 1

    return best_individual

def plot_results(
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    best_individual: Dict[str, Any],
    test_predictions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Plot results comparing training data, test data, and predictions with custom styling.

    Parameters
    ----------
    training_data : List[Dict[str, Any]]
        List of dictionaries containing training data with keys:
        't', 'data', and 'initial_conditions'.
    test_data : List[Dict[str, Any]]
        List of dictionaries containing test data with keys:
        't', 'data', and 'initial_conditions'.
    best_individual : Dict[str, Any]
        Dictionary containing the best solution with keys:
        'mask' and 'params'.
    test_predictions : Optional[List[Dict[str, Any]]], optional
        List of dictionaries containing model predictions for test data,
        by default None.
    """
    plt.figure(figsize=(15, 10))
    plt.suptitle("Model Performance on Training and Test Data", fontsize=16)

    variables = ["Biomass", "Substrate", "Product", "Inhibitor"]
    colors_training = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#e377c2"]
    colors_test = ["#9467bd", "#8c564b", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, var in enumerate(variables):
        plt.subplot(2, 2, idx + 1)

        # Plot training data and predictions
        for i, exp in enumerate(training_data):
            color = colors_training[i]
            # Training data (circles)
            plt.scatter(
                exp["t"],
                exp["data"][:, idx],
                color=color,
                marker="o",
                alpha=0.6,
                label=f"Training Set {i+1}",
            )

            # Training predictions (dashed line)
            t = exp["t"]
            y0 = exp["initial_conditions"]
            y_pred = odeint(
                lambda y, t: candidate_models(
                    y, t, best_individual["params"], best_individual["mask"]
                ),
                y0,
                t,
            )
            plt.plot(
                t,
                y_pred[:, idx],
                color=color,
                linestyle="--",
                alpha=0.8,
                label=f"Training Prediction {i+1}",
            )

        # Plot test data and predictions
        if test_predictions:
            for i, (exp, pred) in enumerate(zip(test_data, test_predictions)):
                color = colors_test[i]
                # Test data (squares)
                plt.scatter(
                    exp["t"],
                    exp["data"][:, idx],
                    color=color,
                    marker="s",
                    alpha=0.6,
                    label=f"Test Set {i+1}",
                )
                # Test predictions (dashed line)
                plt.plot(
                    pred["t"],
                    pred["prediction"][:, idx],
                    color=color,
                    linestyle="--",
                    alpha=0.8,
                    label=f"Test Prediction {i+1}",
                )

        plt.xlabel("Time")
        plt.ylabel(var)
        if idx == 0:  # Only show legend for first subplot
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
    return


def _plot_evolution(
    fitness_history: List[float],
    param_history: List[List[float]],
) -> None:
    """
    Plot the evolution of fitness and parameters.

    Parameters
    ----------
    fitness_history : List[float]
        History of best fitness values.
    param_history : List[List[float]]
        History of parameter values.
    """
    # Plot fitness history
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Evolution")
    plt.grid(True)
    plt.show()

    # Plot parameter evolution
    param_history = np.array(param_history)
    plt.figure(figsize=(12, 8))
    param_names = ["mu_max", "Ks", "Ki", "Yxs", "Kp"]
    for i in range(len(param_names)):
        plt.plot(param_history[:, i], label=param_names[i])
    plt.xlabel("Generation")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_student_solution(
    mask: List[int],
    params: List[float],
    test_data: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], Optional[List[Dict[str, Any]]]]:
    """
    Evaluate a student's solution on the test dataset.

    Parameters
    ----------
    mask : List[int]
        Binary mask indicating which growth terms to include.
    params : List[float]
        Model parameters [mu_max, Ks, Ki, Yxs, Kp].
    test_data : List[Dict[str, Any]]
        List of dictionaries containing test data with keys:
        't', 'data', and 'initial_conditions'.

    Returns
    -------
    Tuple[float, Dict[str, float], Optional[List[Dict[str, Any]]]]
        Tuple containing:
        - Overall RMSE across all variables
        - Dictionary of RMSE values for each variable
        - List of prediction dictionaries (or None if error occurred)
    """
    total_squared_error = 0
    n_points = 0
    errors_by_variable = {
        "biomass": [],
        "substrate": [],
        "product": [],
        "inhibitor": [],
    }

    predictions = []

    for exp in test_data:
        t = exp["t"]
        y_true = exp["data"]
        y0 = exp["initial_conditions"]

        try:
            y_pred = odeint(
                lambda y, t: candidate_models(y, t, params, mask),
                y0,
                t,
            )
            predictions.append(
                {
                    "t": t,
                    "prediction": y_pred,
                    "initial_conditions": y0,
                }
            )

            for i, var in enumerate(["biomass", "substrate", "product", "inhibitor"]):
                mse = np.mean((y_true[:, i] - y_pred[:, i]) ** 2)
                rmse = np.sqrt(mse)
                errors_by_variable[var].append(rmse)

            total_squared_error += np.sum((y_true - y_pred) ** 2)
            n_points += y_true.size

        except:
            return float("inf"), None, None

    overall_rmse = np.sqrt(total_squared_error / n_points)

    # Calculate average RMSE for each variable
    variable_rmse = {var: np.mean(errors) for var, errors in errors_by_variable.items()}

    return overall_rmse, variable_rmse, predictions
