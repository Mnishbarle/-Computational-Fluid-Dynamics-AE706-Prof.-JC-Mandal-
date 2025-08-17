import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def solve_laplace_stream_function(nx, ny, dx, dy, inlet_pos, outlet_pos, psi1, psi2, psi3, 
                                 max_iter=100000, tolerance=1e-4, initial_value=None):
    
    psi = np.zeros((ny, nx))# Initialize stream function matrix with zeros
    
    if initial_value is not None:# Set initial values throughout the domain if provided
        psi.fill(initial_value)
    
    psi[:, 0] = psi3# Left wall (x=0)
    psi[:, -1] = psi3# Right wall (x=3)
    for j in range(nx):# Bottom wall with inlet and outlet
        if j >= inlet_pos[0] and j <= inlet_pos[1]:
            psi[-1, j] = psi1  # Inlet position set to psi1
        elif j >= outlet_pos[0] and j <= outlet_pos[1]:
            psi[-1, j] = psi2  # Outlet position set to psi2
        else:
            psi[-1, j] = psi3  # Rest of bottom wall set to psi3
    psi[0, :] = psi3# Top wall (y=4)
    
    # Calculate the indices for the interior wall at x=1.5m from y=1 to y=2
    wall_x = int(1.5 / dx)  # x index for wall at x=1
    wall_y_start = ny - 1 - int(1.0 / dy)  # y index for bottom of wall (y=1)
    wall_y_end = ny - 1 - int(2.0 / dy)    # y index for top of wall (y=2)
    
    # Set values on the interior wall - varying linearly from psi1 to psi2
    for i in range(wall_y_end, wall_y_start+1):
        y_pos = 4.0 - i * dy  # Calculate physical y position (4-i*dy due to inverted y-axis)
        if y_pos <= 1.1:
            psi[i, wall_x] = psi1  # Below y=1.1m, use psi1
        elif y_pos >= 1.9:
            psi[i, wall_x] = psi2  # Above y=1.9m, use psi2
        else:
            ratio = (y_pos - 1.1) / 0.8 
            psi[i, wall_x] = psi1 + ratio * (psi2 - psi1)
    
    hold_boundary = np.zeros((ny, nx), dtype=bool)
    hold_boundary[0, :] = True  # Top boundary
    hold_boundary[-1, :] = True  # Bottom boundary
    hold_boundary[:, 0] = True  # Left boundary
    hold_boundary[:, -1] = True  # Right boundary
    for i in range(wall_y_end, wall_y_start+1):# Mark the interior wall as a boundary
        hold_boundary[i, wall_x] = True

    psi_new = psi.copy()
    
    iter_count = 0  # Initialize iteration counter
    error = float('inf')  # Initialize error to a large value
    
    # Continue iterating until error is below tolerance or max iterations reached
    while iter_count < max_iter and error > tolerance:
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not hold_boundary[i, j]:  
                    psi_new[i, j] = 0.25 * (psi[i+1, j] + psi[i-1, j] + 
                                           psi[i, j+1] + psi[i, j-1])
        error = np.max(np.abs(psi_new - psi))# Calculate maximum absolute difference between old and new values
        psi = psi_new.copy() # Update solution for next iteration
        iter_count += 1
        if iter_count % 1000 == 0: # Print progress every 1000 iterations
            print(f"Iteration {iter_count}, Error: {error:.8f}")
    
    if iter_count >= max_iter:# Print convergence information
        print(f"Warning: Maximum iterations ({max_iter}) reached before convergence")
    else:
        print(f"Converged after {iter_count} iterations with error {error:.8f}")
    
    return psi, iter_count, error

def plot_contour(psi, title, dx, dy, inlet_pos, outlet_pos, wall_pos):
    ny, nx = psi.shape  # Get dimensions of solution array
    x = np.linspace(0, (nx-1)*dx, nx)# x coordinates from 0 to 3m
    y = np.linspace(4.0, 0, ny)     # y coordinates from 4 to 0m (reversed for plotting)
    X, Y = np.meshgrid(x, y)        # Create 2D grid of coordinates
    plt.figure(figsize=(10, 8))# Create a new figure
    
    contour = plt.contour(X, Y, psi, 20, colors='black', linewidths=0.5)# Plot contour lines in black
    contourf = plt.contourf(X, Y, psi, 20, cmap=cm.viridis)# Fill contours with color gradient
    plt.colorbar(contourf, label='Stream Function ψ')# Add colour bar to show stream function values
    
    wall_x = int(1.5 / dx)  # x index for wall
    wall_y_start = ny - 1 - int(1.0 / dy)  # Bottom of wall (y index)
    wall_y_end = ny - 1 - int(2.0 / dy)    # Top of wall (y index)
    
    wall_y = np.linspace(1.0, 2.0, wall_y_start - wall_y_end + 1)  # y positions along wall
    wall_x_coords = np.ones_like(wall_y) * 1.5                      # constant x=1.5m
    
    plt.plot(wall_x_coords, wall_y, 'k-', linewidth=3)
 
    plt.plot([inlet_pos[0]*dx, inlet_pos[1]*dx], [0, 0], 'r-', linewidth=3)   # Plot inlet as red line
    plt.plot([outlet_pos[0]*dx, outlet_pos[1]*dx], [0, 0], 'g-', linewidth=3)# Plot outlet as green line
    
    # Add title and labels
    plt.title(f'{title}')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure plot has equal aspect ratio
    plt.axis('equal')

def run_test_cases():
    width = 3.0   # Domain width in meters
    height = 4.0  # Domain height in meters
    dx = dy = 0.1  # Grid spacing in meters
    
    nx = int(width / dx) + 1# Calculate number of grid points
    ny = int(height / dy) + 1
    inlet_start_x = 1.0
    inlet_width = 0.1
    inlet_pos = (int(inlet_start_x / dx), int((inlet_start_x + inlet_width) / dx))
    
    # Outlet setup (0.1m wide starting at x=1.9m)
    outlet_start_x = 1.9
    outlet_width = 0.1
    outlet_pos = (int(outlet_start_x / dx), int((outlet_start_x + outlet_width) / dx))
    
    # Interior wall position at x=1.5m
    wall_pos = int(1.5 / dx)
    
    test_cases = [
        {"name": "Test Case 1", "psi1": 100, "psi2": 150, "psi3": 300},  # Test case with smallest psi2-psi1 difference
        {"name": "Test Case 2", "psi1": 100, "psi2": 200, "psi3": 300},  # Test case with medium psi2-psi1 difference
        {"name": "Test Case 3", "psi1": 100, "psi2": 250, "psi3": 300}   # Test case with largest psi2-psi1 difference
    ]
    
    # Initial guesses to try for each test case
    initial_guesses = [100, 150, 200]
    for test in test_cases:
        for initial in initial_guesses:
            # Print test case information
            print(f"\n\n{'=' * 80}")
            print(f"Running {test['name']} with initial guess = {initial}")
            print(f"psi1={test['psi1']}, psi2={test['psi2']}, psi3={test['psi3']}")
            print('=' * 80)
            
            # Solve the Laplace equation using Jacobi iteration
            psi, iterations, error = solve_laplace_stream_function(
                nx, ny, dx, dy, inlet_pos, outlet_pos,
                test['psi1'], test['psi2'], test['psi3'],
                max_iter=50000, tolerance=1e-4, initial_value=initial
            )
            
            # Create title for plot with test case info
            plot_title = f"{test['name']} (ψ1={test['psi1']}, ψ2={test['psi2']}, ψ3={test['psi3']})\nInitial Guess = {initial}, Iterations = {iterations}"
            
            # Generate contour plot
            plot_contour(psi, plot_title, dx, dy, inlet_pos, outlet_pos, wall_pos)
    
    # Display all plots
    plt.show()

# ===== MAIN PROGRAM =====
if __name__ == "__main__":
    run_test_cases()  # Execute the test cases
