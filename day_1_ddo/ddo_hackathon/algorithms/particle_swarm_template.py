import numpy as np

# Resource: https://web2.qatar.cmu.edu/~gdicaro/15382/additional/CompIntelligence-Engelbrecht-ch16.pdf
# First 4 pages give sufficient detail, equations, etc. to code PSO

def particle_swarm(f, x_dim, bounds, iter_tot=100):
    """
    Particle Swarm Optimization (PSO) algorithm template.
    
    Parameters:
    f (function): Objective function to minimize.
    x_dim (int): Dimensionality of the search space.
    bounds (np.ndarray): Array of shape (x_dim, 2) specifying bounds for each dimension.
    iter_tot (int): Budget of function evaluations.
    
    Returns (tuple): Best found position, best fitness value, team name, and names.
    """
    # Step 1: Initialize swarm
    # - Set swarm size
    # - Initialize particle positions within given bounds
    # - Initialize velocities
    # - Set personal best positions and evaluate fitness
    # - Determine the global best position
    
    
    # Step 2: Compute maximum iterations based on evaluation budget
    max_iterations = iter_tot // swarm_size
    
    
    # Optimization loop
    for iter_num in range(max_iterations):
        pass
        # Step 3: Update inertia weight
        
        
        # Step 4: Generate random coefficients for velocity update
        
        
        # Step 5: Update velocities using the PSO velocity update equation
        
        
        # Step 6: Update positions
        
        
        # Step 7: Enforce boundary constraints
        
        
        # Step 8: Evaluate fitness at new positions
        
        
        # Step 9: Update personal best positions
        
        
        # Step 10: Update global best position
        
    
    # Return best solution found
    return global_best_position, global_best_fitness
