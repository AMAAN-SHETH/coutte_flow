import numpy as np
import matplotlib.pyplot as plt

# Parameters
Re = 5000      # Reynolds number
Ny = 21        # Number of grid points
E = 1          # Parameter related to time step stability
D = 1          # Distance between plates (normalized)

# Grid setup
dy = D / (Ny - 1)
dt = E * dy**2 * Re
y = np.linspace(0, D, Ny)

# Initialize velocity field
u = np.zeros(Ny)
u[-1] = 1.0    # Upper plate moving with velocity 1 (non-dimensional)

u_steps = [0, 2, 4, 20, 60, 100, 150, 250, 500, 1000] 
u_profile = {0 : u.copy()} 

def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system using the Thomas algorithm
    """
    n = len(d)
    # Make copies to avoid modifying original arrays
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    
    # Forward elimination
    for i in range(1, n):
        m = ac[i-1] / bc[i-1]
        bc[i] = bc[i] - m * cc[i-1]
        dc[i] = dc[i] - m * dc[i-1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    
    return x

# Coefficients for the finite difference scheme
A = -E / 2
B = 1 + E

# Time stepping
for n in range(np.max(u_steps) + 1):
    # Note: We're solving for interior points only (1 to Ny-2)
    a = np.zeros(Ny-2)  # Lower diagonal (subdiagonal)
    a.fill(A)
    
    b = np.zeros(Ny-2)  # Main diagonal
    b.fill(B)
    
    c = np.zeros(Ny-2)  # Upper diagonal (superdiagonal)
    c.fill(A)
    
    d = np.zeros(Ny-2)  # Right-hand side
    
    # Build right-hand side vector
    for i in range(1, Ny-1):  # Interior points
        d[i-1] = (1-E)*u[i] + E/2 * (u[i-1] + u[i+1])
    
    # Apply boundary conditions to RHS
    d[0] -= A * u[0]    # Incorporate u[0] = 0 boundary
    d[-1] -= A * u[-1]  # Incorporate u[-1] = 1 boundary
    
    # Solve the system
    u_interior = thomas_algorithm(a, b, c, d)
    
    # Update solution (keeping boundary conditions)
    u[1:-1] = u_interior

    # Store the velocity profile at specified steps
    if n in u_steps:
        u_profile[n] = u.copy()

plt.figure(figsize=(10, 6))
for step in u_steps:
    plt.plot(u_profile[step], y, label=f'Step {step}')
plt.plot([0, 1], [0, D], 'r--', linewidth=2, label='Exact Couette')
plt.title('Velocity Profile at Different Time Steps')
plt.xlabel('Velocity (u)')
plt.ylabel('y (normalized distance)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Print the final velocity profile
print("Final velocity profile:")
print(u)