# utils.py
import numpy as np
import matplotlib.pyplot as plt

def plot_opt_landscapes(rastrigin_f, ackley_f, x_range=(-5, 5), y_range=(-5, 5), num_points=400):
    """
    Plot 3D surface and contour plots for Rastrigin and Ackley functions.

    Parameters:
    - rastrigin_f: Callable, Rastrigin function to evaluate.
    - ackley_f: Callable, Ackley function to evaluate.
    - x_range: Tuple, range of x values (min, max). Default is (-5, 5).
    - y_range: Tuple, range of y values (min, max). Default is (-5, 5).
    - num_points: Int, number of points in each dimension. Default is 400.
    """
    # Grid of points
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Evaluate functions
    rastrigin_values = np.array([rastrigin_f(point) for point in grid_points]).reshape(X.shape)
    ackley_values = np.array([ackley_f(point) for point in grid_points]).reshape(X.shape)

    # Create 2x2 plot
    fig = plt.figure(figsize=(12, 8))

    # 1. Rastrigin - 3D Surface Plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, rastrigin_values, cmap='viridis', linewidth=0, antialiased=False)
    ax1.set_title('Rastrigin Function - 3D Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # 2. Ackley - 3D Surface Plot
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, ackley_values, cmap='plasma', linewidth=0, antialiased=False)
    ax2.set_title('Ackley Function - 3D Surface')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(X, Y)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    # 3. Rastrigin - Contour Plot
    ax3 = fig.add_subplot(2, 2, 3)
    contour1 = ax3.contour(X, Y, rastrigin_values, levels=50, cmap='viridis')
    ax3.set_title('Rastrigin Function - Contour')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    fig.colorbar(contour1, ax=ax3)

    # 4. Ackley - Contour Plot
    ax4 = fig.add_subplot(2, 2, 4)
    contour2 = ax4.contour(X, Y, ackley_values, levels=50, cmap='plasma')
    ax4.set_title('Ackley Function - Contour')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    fig.colorbar(contour2, ax=ax4)

    plt.tight_layout()
    plt.show()


def plot_rosenbrock_contour(rosenbrock_f, x_list, x_range=(-3, 3), y_range=(-3, 3), grid_points=400, plot_every_n=100):
    """
    Plot a contour of the 2D Rosenbrock function with a gradient descent trajectory.

    Parameters:
    - rosenbrock_f: Callable, the Rosenbrock function to evaluate.
    - x_list: np.ndarray, array of gradient descent trajectory points.
    - x_range: Tuple, range of x values (min, max). Default is (-3, 3).
    - y_range: Tuple, range of y values (min, max). Default is (-3, 3).
    - grid_points: Int, number of points in each dimension for the grid. Default is 400.
    """
    # Generate a grid for the contour plot
    x = np.linspace(x_range[0], x_range[1], grid_points)
    y = np.linspace(y_range[0], y_range[1], grid_points)
    X, Y = np.meshgrid(x, y)
    x_list = np.array(x_list)

    # Flatten the grid and compute the function values
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = rosenbrock_f(grid_points).reshape(X.shape)

    # Plot the contour and the trajectory
    plt.figure(figsize=(8, 4))
    contour = plt.contour(X, Y, Z, levels=100, cmap="viridis")
    plt.colorbar(contour, label="Function Value")
    plt.plot(x_list[::plot_every_n, 0], x_list[::plot_every_n, 1], 'r-o', label='Gradient Descent Trajectory', markersize=5)
    plt.plot(1, 1, 'bo', label='True Optimum')  # Plot the point (1, 1) in blue
    plt.title("2D Rosenbrock Function with Gradient Descent Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()
