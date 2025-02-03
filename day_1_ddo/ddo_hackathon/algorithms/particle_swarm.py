import numpy as np

def particle_swarm(f, x_dim, bounds, iter_tot=100):
    """
    Particle Swarm Optimization (PSO) algorithm following Algorithm 16.1 (gbest PSO).
    
    Parameters:
    f : function
        Objective function to minimize.
    x_dim : int
        Dimensionality of the search space.
    bounds : np.ndarray
        Array with shape (x_dim, 2) specifying lower and upper bounds for each dimension.
    iter_tot : int, optional
        Total number of function evaluations (default is 100).
    
    Returns:
    tuple
        Best found position, best fitness value, team name, and names (for logging purposes).
    """
    # Step 1: Initialize swarm
    swarm_size = 5  # Number of particles in the swarm
    
    # Initialize particle positions randomly within bounds
    positions = np.random.uniform(0, 1, (swarm_size, x_dim)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    
    # Initialize velocities to zero
    velocities = np.zeros_like(positions)
    
    # Set personal best positions to initial positions
    personal_best_positions = np.copy(positions)
    
    # Evaluate personal best fitness
    personal_best_fitness = np.array([f.fun_test(pos) for pos in positions])
    
    # Determine global best position from initial personal bests
    global_best_idx = np.argmin(personal_best_fitness)  # Index of best particle
    global_best_position = personal_best_positions[global_best_idx]  # Best position found so far
    global_best_fitness = personal_best_fitness[global_best_idx]  # Best fitness value found so far
    
    # Step 2: Compute maximum iterations based on evaluation budget
    max_iterations = iter_tot // swarm_size
    
    # Optimization loop
    for iter_num in range(max_iterations):
        # Step 3: Update inertia weight (dynamically decreases over iterations)
        w = 0.7 - iter_num / max_iterations * 0.5  
        
        # Step 4: Generate random coefficients for velocity update
        r1, r2 = np.random.rand(swarm_size, x_dim), np.random.rand(swarm_size, x_dim)
        
        # Step 5: Update velocities using the PSO velocity update equation
        velocities = (w * velocities +
                      1.5 * r1 * (personal_best_positions - positions) +
                      1.5 * r2 * (global_best_position - positions))
        
        # Step 6: Update positions
        positions += velocities
        
        # Enforce boundary constraints to keep particles within the search space
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])  
        
        # Step 7: Evaluate fitness at new positions
        fitness = np.array([f.fun_test(pos) for pos in positions])
        
        # Step 8: Update personal bests (y_i)
        improved = fitness < personal_best_fitness  # Boolean mask for improvements
        personal_best_positions[improved] = positions[improved]  # Update positions
        personal_best_fitness[improved] = fitness[improved]  # Update fitness values
        
        # Step 9: Update global best position (ŷ)
        global_best_idx = np.argmin(personal_best_fitness)  # Index of best personal best
        global_best_position = personal_best_positions[global_best_idx]  # Update global best position
        global_best_fitness = personal_best_fitness[global_best_idx]  # Update global best fitness
    
    # Return best solution found
    team_name = ['8']  # Placeholder for logging
    names = ['01234567']  # Placeholder for logging
    return global_best_position, global_best_fitness, team_name, names
