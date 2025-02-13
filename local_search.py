import numpy as np
from bio_model import fitness_function
import pickle

with open("training_data_day2.pickle", "rb") as handle:
    training_data = pickle.load(handle)


def local_search_optimize(
    params_start,
    masks_start,
    training_data,
    iterations=10000,
    samples=10,
):
    """
    Local search optimizer for continuous and binary variables, minimizing obj_fn.

    Parameters
    ----------
    params_start : array-like
        Initial continuous parameters.
    masks_start : array-like
        Initial binary mask.
    training_data : List[Dict[str, Any]]
        Training data to fit the model to.
    iterations : int, optional
        Maximum number of outer iterations.
    samples : int, optional
        Number of samples tested per iteration.


    Returns
    -------
    best_obj_value : float
        Minimum objective value found.
    best_params : np.ndarray
        Continuous parameters at the best found solution.
    best_masks : np.ndarray
        Binary mask at the best found solution.
    """
    radius=1.0
    flip_prob=0.5

    best_params = np.array(params_start, dtype=float)
    best_masks = np.array(masks_start, dtype=int)

    best_obj_value = -fitness_function(
        mask=best_masks, params=best_params, training_data=training_data
    )[0]

    for iter_idx in range(iterations):
        for _ in range(samples):
            # Continuous parameter perturbation
            perturbation = np.random.normal(0, 1, size=len(best_params))
            params_candidate = best_params + radius * (
                perturbation / np.sqrt(np.sum(perturbation**2))
            )

            # Binary flips
            masks_candidate = best_masks.copy()
            for i in range(len(masks_candidate)):
                if np.random.rand() < flip_prob:
                    masks_candidate[i] = 1 - masks_candidate[i]

            # Evaluate candidate#
            candidate_obj_value = -fitness_function(
                mask=masks_candidate,
                params=params_candidate,
                training_data=training_data,
            )[0]

            if candidate_obj_value < best_obj_value:
                best_obj_value = candidate_obj_value
                best_params = params_candidate
                best_masks = masks_candidate
                break
        else:
            # Reduce radius if no improvement
            radius *= 0.9

        print(
            f"Iter {iter_idx:4d} | Best obj: {best_obj_value:.6f} | "
            f"params: {best_params} | masks: {best_masks}"
        )

    return best_obj_value, best_params, best_masks


if __name__ == "__main__":
    # Initial guesses
    params_init = [0.5, 1.0, 0.1, 0.9, 0.01]  # 5 continuous parameters
    masks_init = [1, 0, 1, 0, 1, 0, 1, 0, 0]  # 9 binary masks

    best_val, best_p, best_m = local_search_optimize(
        params_start=params_init,
        masks_start=masks_init,
        training_data=training_data,
        iterations=100,
        samples=10,
    )

    print("\nFinal Results:")
    print(f"Best objective value: {best_val}")
    print(f"Best params: {best_p}")
    print(f"Best masks: {best_m}")
