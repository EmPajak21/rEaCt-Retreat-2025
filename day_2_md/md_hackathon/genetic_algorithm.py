import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from typing import List, Dict, Tuple, Any, Union, Optional

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
def genetic_algorithm(
    training_data: List[Dict[str, Any]],
    *,
    generations: int = 10,
    population_size: int = 30,
    mutation_rate: float = 0.1,
    best_container= None
) -> Dict[str, Any]:
    """
    Genetic algorithm for model selection.

    Parameters
    ----------
    training_data : List[Dict[str, Any]]
        Training data to fit the model to.
    generations : int, optional
        Number of generations to run, by default 10.
    population_size : int, optional
        Size of the population, by default 30.
    mutation_rate : float, optional
        Probability of mutation, by default 0.1.

    Returns
    -------
    Dict[str, Any]
        Best individual found (mask and parameters).

    Raises
    ------
    ValueError
        If population_size is not even.
    """
    if population_size % 2 != 0:
        raise ValueError("Population size must be even")

    n_params = 5

    # Initialize population
    population = [
        {
            "mask": np.random.randint(0, 2, size=len(TERM_NAMES)),
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
                [TERM_NAMES[i] for i, m in enumerate(best_individual["mask"]) if m],
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
                    for j in range(len(TERM_NAMES))
                ]
            )

            # Parameter crossover
            child_params = []
            for p1, p2 in zip(parent1["params"], parent2["params"]):
                param = np.random.uniform(min(p1, p2), max(p1, p2))
                child_params.append(param)

            # Mutation
            if np.random.rand() < mutation_rate:
                child_mask[np.random.randint(0, len(TERM_NAMES))] ^= 1
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

#    _plot_evolution(fitness_history, param_history)

    return best_individual

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