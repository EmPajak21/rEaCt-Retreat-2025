import os
import sys
from typing import List, Union
import pickle

# Add parent directories to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


from bio_model import generate_training_data, true_model_day3


def generate_after_submission(
    initial_conditions: Union[List[float], List[List[float]]] = [[0.1, 0.2, 0.05, 0.6]],
    team_name="",
):
    """
    Generate training data with noise for day 3.

    Parameters
    ----------
    initial_conditions : Union[List[float], List[List[float]]]
        Initial values for state variables [X, S, P, I].

    Returns
    -------
    List[Dict[str, Any]]
        Training data as list of dictionaries containing time points and noisy data
    """
    data = generate_training_data(
        initial_conditions=initial_conditions, true_model=true_model_day3
    )

    if team_name:
        pickle.dump(
            data,
            open(
                f"day_3_doe/doe_hackathon/admin/data/{team_name}.pkl",
                "wb",
            ),
        )
        return data
    else:
        return data


if __name__ == "__main__":
    data = generate_after_submission(
        initial_conditions=[[0.1, 0.2, 0.05, 0.6]],
        team_name="team1",
    )

