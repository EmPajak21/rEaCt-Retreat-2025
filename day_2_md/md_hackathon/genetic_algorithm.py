import numpy as np
from typing import List, Dict, Any, Callable
from bio_model import candidate_models, fitness_function


def genetic_algorithm(
    training_data: List[Dict[str, Any]],
    candidate_models: Callable = candidate_models,
    basic_fitness_function: Callable = fitness_function,
    best_container=None,
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
    POPULATION_SIZE = 10

    if POPULATION_SIZE % 2 != 0:
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
        for _ in range(POPULATION_SIZE)
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
        population = [population[i] for i in sorted_indices[: POPULATION_SIZE // 2]]

        # Create new population
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
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

    return best_individual
