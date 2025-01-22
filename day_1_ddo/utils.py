import numpy as np
import matplotlib.pyplot as plt


# Constants for the ranges
VMAX_MIN = 5
VMAX_MAX = 25
KM_MIN = 0.5
KM_MAX = 8


# Generate synthetic data
def generate_data(Vmax=15, Km=1.8, noise_level=0.1, num_points=50):
    """
    Generate synthetic data for the Michaelis-Menten model with added noise.

    Parameters:
    Vmax (float): Maximum reaction rate. Default is 15.
    Km (float): Michaelis constant (substrate concentration at half Vmax). Default is 1.8.
    noise_level (float): Standard deviation of the noise added to the data. Default is 0.1.
    num_points (int): Number of substrate concentration data points to generate. Default is 50.

    Returns:
    tuple: A tuple containing:
        - S (numpy.ndarray): Substrate concentration data points.
        - v_noisy (numpy.ndarray): Noisy reaction rate data points.
        - v (numpy.ndarray): True reaction rate data points (without noise).
    """
    S = np.linspace(0.1, 10, num_points)  # Substrate concentrations
    v = (Vmax * S) / (Km + S)  # Michaelis-Menten equation
    noise = np.random.normal(0, noise_level, size=S.shape)  # Add noise
    v_noisy = v + noise
    return S, v_noisy, v


# Calculate the loss (Mean Squared Error)
def calculate_loss(Vmax, Km, S, v_data):
    """
    Calculate the root mean squared error (RMSE) between model predictions and observed data.

    Parameters:
    Vmax (float): Maximum reaction rate.
    Km (float): Michaelis constant.
    S (numpy.ndarray): Substrate concentration data points.
    v_data (numpy.ndarray): Observed reaction rate data points.

    Returns:
    float: The root mean squared error (RMSE) value.
    """
    v_model = (Vmax * S) / (Km + S)
    rmse = np.sqrt(np.mean((v_model - v_data) ** 2))
    return rmse


# Plotting function with 2D Michaelis-Menten and 3D loss surface
def plot_model_and_loss(Vmax=15, Km=1.8):
    """
    Plot the Michaelis-Menten model and a 3D loss surface.

    Parameters:
    Vmax (float): Maximum reaction rate. Default is 15.
    Km (float): Michaelis constant. Default is 1.8.

    Returns:
    None: The function generates plots but does not return any values.
    """
    # Round input
    Vmax = np.round(Vmax, decimals=3)
    Km = np.round(Km, decimals=3)
    # Generate data
    S, v_data, v_true = generate_data()

    # Calculate the reaction rate based on current parameters
    v_model = (Vmax * S) / (Km + S)

    # Calculate loss (Mean Squared Error)
    rmse = calculate_loss(Vmax, Km, S, v_data)

    # Create subplots for 2D model plot and 3D loss surface plot
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, projection="3d", computed_zorder=False)

    # Left plot: Michaelis-Menten Model (2D)
    ax0.scatter(S, v_data, label="Synthetic Data (noisy)", color="red")
    ax0.plot(S, v_model, label=f"Model (Vmax={Vmax}, Km={Km})", color="blue")
    ax0.set_xlabel("Substrate Concentration [S]")
    ax0.set_ylabel("Reaction Rate v")
    ax0.set_title("Michaelis-Menten Model")
    ax0.legend()
    ax0.grid()

    # Right plot: Loss Surface (3D)
    Vmax_range = (VMAX_MIN, VMAX_MAX)
    Km_range = (KM_MIN, KM_MAX)

    Vmax_values = np.linspace(*Vmax_range, 50)
    Km_values = np.linspace(*Km_range, 50)
    Vmax_grid, Km_grid = np.meshgrid(Vmax_values, Km_values)

    # Compute the loss for each pair of (Vmax, Km)
    loss_grid = np.array(
        [
            calculate_loss(Vmax, Km, S, v_data)
            for Vmax, Km in zip(Vmax_grid.ravel(), Km_grid.ravel())
        ]
    ).reshape(Vmax_grid.shape)

    # Plot the 3D surface
    surf = ax1.plot_surface(
        Vmax_grid,
        Km_grid,
        loss_grid,
        cmap="viridis",
        edgecolor="none",
        alpha=0.7,  # Reduce transparency of the surface
        zorder=1,
    )

    # Mark the current point with an 'x', ensuring it's plotted above the surface
    ax1.scatter(
        Vmax,
        Km,
        rmse,
        color="orange",
        s=200,
        label="Current Point",
        marker="x",
        zorder=100,  # Increase the zorder for the marker
    )

    ax1.set_xlabel("$V_{max}$")
    ax1.set_ylabel("$K_m$")
    ax1.set_zlabel("Loss (RMSE)")
    ax1.set_title("Loss Surface")
    ax1.view_init(elev=15, azim=290)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # Show the plot
    plt.tight_layout()
    plt.show()
    return


def visualize_ofat(func, initial_guess=[1, 1, 1, 1]):
    """
    Visualises the One-Factor-At-a-Time (OFAT) optimisation process by iteratively updating
    one parameter at a time while keeping others fixed at their initial values. Each plot shows
    the function's response to changes in one parameter.

    Parameters:
    func (callable): The function to apply OFAT to.
    initial_guess (list): List of initial guesses for the function parameters.

    Returns:
    None: The function generates plots but does not return any values.
    """
    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    param_names = ["p1", "p2", "p3", "p4"]
    fixed_params = initial_guess
    print("We start with an initial guess of")
    for name, param in zip(param_names, fixed_params):
        print(f"{name}: {param}")

    for i, ax in enumerate(axs.flatten()):
        param_values = np.linspace(
            -10, 10, 100
        )  # Range of values for the current parameter
        function_values = []

        # For each value in param_values, update the ith parameter and compute the function value
        for val in param_values:
            updated_params = fixed_params[:]
            updated_params[i] = val  # Vary the ith parameter
            function_values.append(func(updated_params))

        # Plot the function values with respect to the varying parameter
        ax.plot(param_values, function_values, label=f"Varying {param_names[i]}")

        # Find the minimum of the function and highlight it
        min_value = min(function_values)
        min_index = function_values.index(min_value)
        min_param_value = param_values[min_index]
        ax.axvline(
            min_param_value, color="r", linestyle="--", label=f"Min of {param_names[i]}"
        )

        ax.set_title(f"OFAT for {param_names[i]}")
        ax.legend()
        ax.grid()

        # Update the fixed parameters for the next plot
        fixed_params[i] = min_param_value

    plt.tight_layout()
    plt.show()
    print("The final parameters after OFAT are")
    for name, param in zip(param_names, fixed_params):
        print(f"{name}: {np.round(param, decimals=2)}")
    print(f"The score is {func(fixed_params)}")
    return
