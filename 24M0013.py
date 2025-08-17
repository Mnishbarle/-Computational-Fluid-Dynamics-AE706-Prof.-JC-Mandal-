# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
gamma = 1.4
R = 287.0
p0 = 1.0133e5
T0 = 300.0
pe_p0 = 0.585
CFL = 0.5
Imax = 101
tol = 1e-8

# Grid setup
x = np.linspace(0, 2, Imax)
dx = x[1] - x[0]
A = 1 + 2 * (x - 1)**2
shock_idx = np.argmin(np.abs(x - 1.7))  # Index closest to x = 1.7

# Initialize flow
def initialize_flow():
    rho0 = p0 / (R * T0)                                      # Calculating the density
    U = np.zeros((3, Imax))                                   # Creating a dummy variable U (velocity)
    U[0, :] = rho0 * A                                      
    U[1, :] = rho0 * 100 * A                                  # Initial velocity 100 m/s
    U[2, :] = p0 / (gamma - 1) * A + 0.5 * rho0 * 100**2 * A
    return U

# Primitive variables
def get_primitives(U):
    rho = U[0, :] / A
    u = U[1, :] / (rho * A + 1e-10)
    E = U[2, :] / A
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    p = np.maximum(p, 1e-10)
    T = p / (rho * R)
    a = np.sqrt(gamma * R * T)              # Speed of sound
    M = u / a                               # Mach number
    return rho, u, p, a, M

# Van Leer Flux Vector Splitting with while loop
def van_leer_flux(U):
    rho, u, p, a, M = get_primitives(U)
    F_plus = np.zeros((3, Imax))
    F_minus = np.zeros((3, Imax))
    F = np.zeros((3, Imax))
    
    F[0, :] = rho * u * A
    F[1, :] = (rho * u**2 + p) * A
    F[2, :] = u * (U[2, :] + p * A)
    
    i = 0  # Initialize index
    while i < Imax:
        if M[i] <= -1:
            F_plus[:, i] = 0
            F_minus[:, i] = F[:, i]
        elif M[i] >= 1:
            F_plus[:, i] = F[:, i]
            F_minus[:, i] = 0
        else:
            f_mass_plus = 0.25 * rho[i] * a[i] * (M[i] + 1)**2 * A[i]
            f_mass_minus = -0.25 * rho[i] * a[i] * (M[i] - 1)**2 * A[i]
            f_mom_plus = f_mass_plus * ((gamma - 1) * u[i] + 2 * a[i]) / gamma
            f_mom_minus = f_mass_minus * ((gamma - 1) * u[i] - 2 * a[i]) / gamma
            f_energy_plus = f_mass_plus * ((gamma - 1) * u[i] + 2 * a[i])**2 / (2 * (gamma**2 - 1))
            f_energy_minus = f_mass_minus * ((gamma - 1) * u[i] - 2 * a[i])**2 / (2 * (gamma**2 - 1))
            
            F_plus[0, i] = f_mass_plus
            F_plus[1, i] = f_mom_plus
            F_plus[2, i] = f_energy_plus
            F_minus[0, i] = f_mass_minus
            F_minus[1, i] = f_mom_minus
            F_minus[2, i] = f_energy_minus
        i += 1  # Increment index
    
    return F_plus, F_minus

# Source term
def source_term(U):
    _, _, p, _, _ = get_primitives(U)
    dA_dx = np.zeros(Imax)
    dA_dx[1:-1] = (A[2:] - A[:-2]) / (2 * dx)
    S = np.zeros((3, Imax))
    S[1, :] = p * dA_dx
    return S

# Time step
def compute_dt(U):
    _, u, _, a, _ = get_primitives(U)
    lambda_max = np.max(np.abs(u) + a)
    return CFL * dx / lambda_max

# Boundary conditions
def apply_boundary(U):
    rho, u, p, a, _ = get_primitives(U)
    u[0] = u[1]
    M_in = u[0] / a[0]
    p[0] = p0 * (1 + (gamma - 1) / 2 * M_in**2)**(-gamma / (gamma - 1))
    rho[0] = p[0] / (R * T0 * (p[0] / p0)**((gamma - 1) / gamma))
    p[-1] = pe_p0 * p0
    rho[-1] = rho[-2]
    u[-1] = u[-2]
    U[0, :] = rho * A
    U[1, :] = rho * u * A
    U[2, :] = (p / (gamma - 1) + 0.5 * rho * u**2) * A
    return U

# Exact solution with shock
def exact_solution():
    p_exact = np.zeros(Imax)
    M_exact = np.zeros(Imax)
    A_star = 1.0
    
    # Before shock (x <= 1.7)
    for i in range(Imax):
        if x[i] <= 1.7:
            def area_ratio(M):
                return (1 / M) * (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M**2))**((gamma + 1) / (2 * (gamma - 1))) - A[i] / A_star
            M_guess = 1.0 if x[i] == 1.0 else (0.5 if x[i] < 1.0 else 1.5)
            M_exact[i] = fsolve(area_ratio, M_guess)[0]
            p_exact[i] = p0 * (1 + (gamma - 1) / 2 * M_exact[i]**2)**(-gamma / (gamma - 1))
    
    # Shock at x = 1.7
    M1 = M_exact[shock_idx]
    p1 = p_exact[shock_idx]
    M2 = np.sqrt(((gamma - 1) * M1**2 + 2) / (2 * gamma * M1**2 - (gamma - 1)))
    p2 = p1 * (2 * gamma * M1**2 - (gamma - 1)) / (gamma + 1)
    
    # After shock (x > 1.7)
    for i in range(shock_idx + 1, Imax):
        def area_ratio_post(M):
            return (1 / M) * (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M**2))**((gamma + 1) / (2 * (gamma - 1))) - A[i] / A[shock_idx] * (1 / M2) * (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M2**2))**((gamma + 1) / (2 * (gamma - 1)))
        M_exact[i] = fsolve(area_ratio_post, 0.6)[0]
        p_exact[i] = p2 * (1 + (gamma - 1) / 2 * M2**2)**(gamma / (gamma - 1)) / (1 + (gamma - 1) / 2 * M_exact[i]**2)**(gamma / (gamma - 1))
    
    return p_exact / p0, M_exact

# Solver
U = initialize_flow()
max_iter = 20000
residual = 1.0
iter_count = 0

while residual > tol and iter_count < max_iter:
    U_old = U.copy()
    dt = compute_dt(U)
    F_plus, F_minus = van_leer_flux(U)
    S = source_term(U)
    U_new = U.copy()
    U_new[:, 1:-1] = U[:, 1:-1] - dt / dx * (F_plus[:, 1:-1] - F_plus[:, :-2]) - dt / dx * (F_minus[:, 2:] - F_minus[:, 1:-1]) + dt * S[:, 1:-1]
    U = apply_boundary(U_new)
    residual = np.sqrt(np.sum((U - U_old)**2) / (3 * Imax))
    iter_count += 1
    if iter_count % 2000 == 0:
        _, _, p_temp, _, M_temp = get_primitives(U)
        print(f"Iteration {iter_count}, Residual = {residual:.2e}, Exit p/p0 = {p_temp[-1]/p0:.3f}, Exit M = {M_temp[-1]:.3f}")

print(f"Converged after {iter_count} iterations with residual = {residual:.2e}")

# Results
_, _, p, _, M = get_primitives(U)
p_p0 = p / p0
p_exact, M_exact = exact_solution()

# Numerical shock location
M_diff = np.abs(np.diff(M))
num_shock_idx = shock_idx - 1 + np.argmax(M_diff[shock_idx-5:shock_idx+5])
M_pre_num = M[num_shock_idx]
M_post_num = M[num_shock_idx + 1]

# Theoretical shock values
M_pre_theory = M_exact[shock_idx]
M_post_theory = M_exact[shock_idx + 1]

# Exit values
p_exit_num = p[-1]
M_exit_num = M[-1]
p_exit_exact = p0 * pe_p0
M_exit_exact = M_exact[-1]
print(f"Numerical Exit Pressure: {p_exit_num:.2f} Pa, p_e/p_0: {p_exit_num/p0:.3f}")
print(f"Theoretical Exit Pressure: {p_exit_exact:.2f} Pa, p_e/p_0: {pe_p0:.3f}")
print(f"Numerical Exit Mach Number: {M_exit_num:.3f}")
print(f"Theoretical Exit Mach Number: {M_exit_exact:.3f}")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, p_p0, 'b', label='Numerical')
plt.plot(x, p_exact, 'r--', label='Exact')
plt.xlabel('x (in m)')
plt.ylabel('p/p0')
plt.legend()
plt.title('Non-Dimensional Pressure Ratio')

plt.subplot(1, 2, 2)
plt.plot(x, M, 'b-', label='Numerical')
plt.plot(x, M_exact, 'r--', label='Exact')
plt.xlabel('x(in m)')
plt.ylabel('Mach Number')
plt.legend()
plt.title('Mach Number')
plt.tight_layout()
plt.show()
