import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0          # Cavity size (m)
I = 31           # Grid points (31x31)
J = 31
Re = 100         # Reynolds number
nu = 1.0 / Re    # Kinematic viscosity
sig_c = 0.4    # Courant number (convective)
sig_d = 0.6    # Diffusion number
dx = L / 30 # Grid spacing
dy = dx
RMS_threshold = 1e-8  # Convergence threshold for velocities
max_steps = 10000   # Maximum time steps

# Initialize arrays
psi = np.full((I, J), 100.0)  # Stream function (ψ = 100 everywhere initially)
omega = np.zeros((I, J))      # Vorticity (ω = 0 initially)
u = np.zeros((I, J))          # x-velocity
v = np.zeros((I, J))          # y-velocity

# Function to apply boundary conditions
def apply_boundary_conditions(psi, omega, u, v):
    """
    Apply boundary conditions for vorticity, stream function, and velocity.
    Top wall: u = 1 m/s, v = 0, ψ = 100.
    Bottom, left, right walls: u = v = 0, ψ = 100.
    Vorticity computed using assignment formulas.
    """
    # Top wall (y = 1, j = J-1)
    psi[:, -1] = 100.0
    u[:, -1] = 1.0
    v[:, -1] = 0.0
    omega[:, -1] = -2 * (psi[:, -2] - psi[:, -1]) / dy**2 - 2 * u[:, -1] / dy

    # Bottom wall (y = 0, j = 0)
    psi[:, 0] = 100.0
    u[:, 0] = 0.0
    v[:, 0] = 0.0
    omega[:, 0] = -2 * (psi[:, 1] - psi[:, 0]) / dy**2

    # Left wall (x = 0, i = 0)
    psi[0, :] = 100.0
    u[0, :] = 0.0
    v[0, :] = 0.0
    omega[0, :] = -2 * (psi[1, :] - psi[0, :]) / dx**2

    # Right wall (x = 1, i = I-1)
    psi[-1, :] = 100.0
    u[-1, :] = 0.0
    v[-1, :] = 0.0
    omega[-1, :] = -2 * (psi[-2, :] - psi[-1, :]) / dx**2

# Function to compute time step based on stability conditions
def compute_time_step(u, v):
    """
    Compute time step Δt based on convective and diffusive stability conditions.
    Δt = min(Δt_c, Δt_d).
    """
    u_max = np.max(np.abs(u)) + 1e-10  # Avoid division by zero
    v_max = np.max(np.abs(v)) + 1e-10
    dt_c = sig_c * (dx * dy) / (u_max * dy + v_max * dx)
    dt_d = sig_d * (dx**2 * dy**2) / (2 * nu * (dx**2 + dy**2))
    return min(dt_c, dt_d)

# Function to compute second-order upwind derivative
def upwind_derivative(f, u, delta, axis):
    """
    Compute second-order upwind derivative for convective terms.
    f: field (e.g., vorticity), u: velocity component, delta: grid spacing.
    axis: 0 for x-direction, 1 for y-direction.
    """
    df = np.zeros_like(f)
    if axis == 0:  # x-direction
        i = 2
        while i < I-2:  # Avoid i+2 out of bounds
            j = 1
            while j < J-1:
                if u[i,j] >= 0:
                    df[i,j] = (3*f[i,j] - 4*f[i-1,j] + f[i-2,j]) / (2*delta)
                else:
                    df[i,j] = (-f[i+2,j] + 4*f[i+1,j] - 3*f[i,j]) / (2*delta)
                j += 1
            i += 1
        # Near boundaries (i=I-2, J-1), use first-order upwind
        for i in [I-2, J-1]:  # For loop kept as it's more clear for this specific case
            j = 1
            while j < J-1:
                if u[i,j] >= 0 and i >= 1:
                    df[i,j] = (f[i,j] - f[i-1,j]) / delta
                else:
                    df[i,j] = 0  # Safe fallback
                j += 1
    else:  # y-direction
        i = 1
        while i < J-1:
            j = 2
            while j < J-2:  # Avoid j+2 out of bounds
                if u[i,j] >= 0:
                    df[i,j] = (3*f[i,j] - 4*f[i,j-1] + f[i,j-2]) / (2*delta)
                else:
                    df[i,j] = (-f[i,j+2] + 4*f[i,j+1] - 3*f[i,j]) / (2*delta)
                j += 1
            i += 1
        # Near boundaries (j=J-2, J-1), use first-order upwind
        i = 1
        while i < J-1:
            for j in [I-2, J-1]:  # For loop kept as it's more clear for this specific case
                if u[i,j] >= 0 and j >= 1:
                    df[i,j] = (f[i,j] - f[i,j-1]) / delta
                else:
                    df[i,j] = 0  # Safe fallback
            i += 1
    return df

# Function to update vorticity using vorticity transport equation
def update_vorticity(psi, omega, u, v, dt):
    """
    Update vorticity using the vorticity transport equation
    Uses second-order upwind for convection, central differences for diffusion.
    """
    omega_new = omega.copy()
    # Compute derivatives
    domega_dx = upwind_derivative(omega, u, dx, axis=0)
    domega_dy = upwind_derivative(omega, v, dy, axis=1)
    d2omega_dx2 = (omega[2:, 1:-1] - 2*omega[1:-1, 1:-1] + omega[:-2, 1:-1]) / dx**2
    d2omega_dy2 = (omega[1:-1, 2:] - 2*omega[1:-1, 1:-1] + omega[1:-1, :-2]) / dy**2
    # Update vorticity (explicit Euler)
    convection = u[1:-1, 1:-1] * domega_dx[1:-1, 1:-1] + v[1:-1, 1:-1] * domega_dy[1:-1, 1:-1]
    diffusion = nu * (d2omega_dx2 + d2omega_dy2)
    omega_new[1:-1, 1:-1] = omega[1:-1, 1:-1] + dt * (-convection + diffusion)
    return omega_new

# Function to solve stream function equation using Gauss-Seidel
def solve_stream_function(psi, omega, tol=1e-2, max_iter=1000):
    """
    Solve Poisson equation using Gauss-Seidel iteration.
    Stop when RMS residual ≤ 10^-2.
    """
    psi_new = psi.copy()
    iteration = 0
    while iteration < max_iter:
        psi_old = psi_new.copy()
        # Gauss-Seidel update
        i = 1
        while i < I-1:
            j = 1
            while j < J-1:
                psi_new[i,j] = 0.25 * ( psi_new[i+1,j] + psi_new[i-1,j] + psi_new[i,j+1] + psi_new[i,j-1] + dx**2 * omega[i,j] )
                j += 1
            i += 1

        # Compute residual
        residual = (
            (psi_new[2:, 1:-1] - 2*psi_new[1:-1, 1:-1] + psi_new[:-2, 1:-1]) / dx**2 +
            (psi_new[1:-1, 2:] - 2*psi_new[1:-1, 1:-1] + psi_new[1:-1, :-2]) / dy**2 +
            omega[1:-1, 1:-1]
        )
        R2 = np.sqrt(np.mean(residual**2))
        if R2 < tol:
            break
        iteration += 1
    return psi_new

# Function to compute velocities from stream function
def compute_velocities(psi):
    """
    Compute velocity components from the definition of stream functions
    Uses central differences
    """
    u = np.zeros((I, J))
    v = np.zeros((I, J))
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)
    return u, v

# Main simulation loop
RMS_u_history = []
RMS_v_history = []
step = 0
while step < max_steps:
    # Apply boundary conditions at the start of each step
    apply_boundary_conditions(psi, omega, u, v)

    # Compute velocities (old)
    u_old, v_old = compute_velocities(psi)

    # Compute time step
    dt = compute_time_step(u_old, v_old)

    # Update vorticity
    omega = update_vorticity(psi, omega, u_old, v_old, dt)

    # Solve stream function
    psi = solve_stream_function(psi, omega)

    # Compute velocities (new)
    u_new, v_new = compute_velocities(psi)

    # Update boundary velocities explicitly (to ensure consistency)
    u_new[:, -1] = 1.0  # Top wall
    u_new[:, 0] = 0.0   # Bottom wall
    u_new[0, :] = 0.0   # Left wall
    u_new[-1, :] = 0.0  # Right wall
    v_new[:, -1] = 0.0  # Top wall
    v_new[:, 0] = 0.0   # Bottom wall
    v_new[0, :] = 0.0   # Left wall
    v_new[-1, :] = 0.0  # Right wall

    # Compute RMS residuals
    RMS_u = np.sqrt(np.mean((u_new[1:-1, 1:-1] - u_old[1:-1, 1:-1])**2))
    RMS_v = np.sqrt(np.mean((v_new[1:-1, 1:-1] - v_old[1:-1, 1:-1])**2))
    RMS_u_history.append(RMS_u)
    RMS_v_history.append(RMS_v)

    # Print progress
    if step % 1000 == 0:
        print(f"Step {step}, RMS_u: {RMS_u:.2e}, RMS_v: {RMS_v:.2e}, u_max: {np.max(np.abs(u_new)):.2e}, v_max: {np.max(np.abs(v_new)):.2e}")

    # Check convergence
    if RMS_u < RMS_threshold and RMS_v < RMS_threshold:
        print(f"Converged at step {step}")
        break
        
    step += 1

# Ghia et al. benchmark data from csv file
# Mid vertical line (x-velocity, u at x=0.5)
ghia_u_y = np.array([1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
                     0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0])
ghia_u = np.array([1, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, 
                   -0.20581, -0.2109, -0.15662, -0.1015, -0.06434, -0.04775, -0.04192, -0.03717, 0])

# Mid horizontal line (y-velocity, v at y=0.5)
ghia_v_x = np.array([1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
                     0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0])
ghia_v = np.array([0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 
                   0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.1089, 0.10091, 0.09233, 0])

# Generate plots
x = np.linspace(0, L, I)
y = np.linspace(0, L, J)
X, Y = np.meshgrid(x, y)

# 1. Stream function contours
plt.figure(figsize=(6, 6))
contour = plt.contourf(X, Y, psi.T, levels=20, cmap='viridis')
plt.colorbar(contour, label='Stream Function (ψ)')
plt.title("Stream Function Contours")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()

# 2. Streamlines
plt.figure(figsize=(6, 6))
plt.streamplot(X, Y, u_new.T, v_new.T, density=1.5, color='black')
plt.title("Streamlines")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.show()

# 3. u-velocity along vertical midline (x = 0.5)
plt.figure(figsize=(6, 6))
plt.plot(u_new[I//2, :], y, label="Computed u at x=0.5", color='blue')
plt.plot(ghia_u, ghia_u_y, 'ro', label="Ghia et al. u", markersize=5)
plt.xlabel("u (m/s)")
plt.ylabel("y (m)")
plt.title("u-velocity along x=0.5")
plt.legend()
plt.grid(True)
plt.show()

# 4. v-velocity along horizontal midline (y = 0.5)
plt.figure(figsize=(6, 6))
plt.plot(x, v_new[:, J//2], label="Computed v at y=0.5", color='blue')
plt.plot(ghia_v_x, ghia_v, 'ro', label="Ghia et al. v", markersize=5)
plt.xlabel("x (m)")
plt.ylabel("v (m/s)")
plt.title("v-velocity along y=0.5")
plt.legend()
plt.grid(True)
plt.show()

# 5. Convergence history
plt.figure(figsize=(6, 6))
plt.semilogy(RMS_u_history, label="RMS_u", color='blue')
plt.semilogy(RMS_v_history, label="RMS_v", color='red')
plt.xlabel("Iteration")
plt.ylabel("RMS Residual")
plt.title("Convergence History")
plt.legend()
plt.grid(True)
plt.show()
