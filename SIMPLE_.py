import numpy as np
import matplotlib.pyplot as plt

# Parameters from the book
L = 0.5      # ft (length of domain)
D = 0.01     # ft (height of domain)
u_e = 1.0    # ft/s (upper wall velocity)
rho = 0.002377  # slug/ft^3
Re = 63.6     # Reynolds number
mu = u_e * D * rho / Re   # slug/(ft·s) (calculated from Re = 63.6)
alpha_p = 0.1  # Under-relaxation factor

# Grid parameters (matches book exactly)
Nx_p = 21     # Number of pressure points in x (index i)
Ny_p = 11     # Number of pressure points in y (index j)
Nt = 301      # Number of iterations

u_steps = [0, 2, 20, 60, 100, 250, 300]
v_steps = [0, 2, 20, 60, 100, 250, 300]
u_profile = {0: np.zeros(Ny_p)}
v_profile = {0: np.zeros(Ny_p + 1)}

dx = L / (Nx_p - 1)
dy = D / (Ny_p - 1)
dt = 0.001    # Time step (matches book)

# Initialize fields with correct dimensions
p = np.zeros((Ny_p, Nx_p))      # Pressure
u = np.zeros((Ny_p, Nx_p + 1))  # x-velocity (staggered in x)
v = np.zeros((Ny_p + 1, Nx_p + 2))  # y-velocity (staggered in x and y)

# Set initial conditions (matches book)
u[-1, :] = u_e       # Upper wall moving at u_e
v[5, 15] = 0.5       # Velocity spike at (15,5) in book's indexing (1-based)

# Create coordinate arrays for plotting
y_u = np.linspace(0, D, Ny_p)      # y-coords for u-velocity
y_v = np.linspace(-dy/2, D+dy/2, Ny_p + 1)  # y-coords for v-velocity
x_p = np.linspace(0, L, Nx_p)       # x-coords for pressure
x_u = np.linspace(-dx/2, L+dx/2, Nx_p + 1)  # x-coords for u-velocity
x_v = np.linspace(-dx, L+dx, Nx_p + 2)      # x-coords for v-velocity

X_p, Y_p = np.meshgrid(x_p, y_u)
X_u, Y_u = np.meshgrid(x_u, y_u)
X_v, Y_v = np.meshgrid(x_v, y_v)

plt.figure(figsize=(12, 6))

plt.scatter(X_p, Y_p, c='red', s=50, label='Pressure (P)', marker='o')
plt.scatter(X_u, Y_u, c='blue', s=30, label='U-velocity', marker='>')
plt.scatter(X_v, Y_v, c='green', s=30, label='V-velocity', marker='x')

# Add grid lines for reference
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=D, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.axvline(x=L, color='k', linestyle='-')

plt.title('Staggered Grid Visualization', fontsize=14)
plt.xlabel('X direction', fontsize=12)
plt.ylabel('Y direction', fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

plt.gca().set_aspect('auto')
plt.tight_layout()


# Main iteration loop
for n in range(Nt):
    # Store old values
    u_old = u.copy()
    v_old = v.copy()
    p_old = p.copy()
    
    # Top wall (j = Ny_p-1 in 0-based)
    u[-1, :] = u_e
    v[-1, :] = 0
    p[-1, :] = 0
    
    # Bottom wall (j = 0)
    u[0, :] = 0
    v[0, :] = 0
    p[0, :] = 0
    
    # Inflow (left boundary, i = 0)
    p[:, 0] = 0
    v[:, 0] = 0
    
    # Outflow (right boundary, i = Nx_p-1)
    p[:, -1] = 0
    
    # Velocity spike (only at first iteration)
    if n == 0:
        v[5, 15] = 0.5
    
    # Solve x-momentum equation (Eq. 6.94 in book)
    for j in range(1, Ny_p-1):      # j = 1 to Ny_p-2 
        for i in range(1, Nx_p):    # i = 1 to Nx_p-1 
            # Convective terms
            # p_old has shape (Ny_p, Nx_p)
            # u has shape (Ny_p, Nx_p+1)
            # So u[i] corresponds to p[i-1] (since u is staggered)
            conv_x = -(rho*u[j,i+1]**2 - rho*u[j,i-1]**2)/(2*dx)
            
            # Interpolate v to u points
            v_bar = 0.5*(v[j+1,i+1] + v[j+1,i])
            v_bar_bar = 0.5*(v[j,i+1] + v[j,i])
            conv_y = -(rho*u[j+1,i]*v_bar - rho*u[j-1,i]*v_bar_bar)/(2*dy)
            
            # Viscous terms
            visc_x = (mu/rho)*(u[j,i+1] - 2*u[j,i] + u[j,i-1])/dx**2
            visc_y = (mu/rho)*(u[j+1,i] - 2*u[j,i] + u[j-1,i])/dy**2
            
            A_star = conv_x + conv_y + visc_x + visc_y
            
            # Update u
            u[j,i] = u_old[j,i] + A_star*dt - (dt/dx)*(p_old[j,i] - p_old[j,i-1])/rho
    
    # Solve y-momentum equation (Eq. 6.95 in book)
    for j in range(1, Ny_p):        # j = 1 to Ny_p-1 (0-based)
        for i in range(1, Nx_p+1):  # i = 1 to Nx_p (0-based)
            # Calculate B* term
            # Interpolate u to v points
            # p_old has shape (Ny_p, Nx_p)
            # v has shape (Ny_p+1, Nx_p+2)
            # So v[j,i] corresponds to p[j-1,i-1]
            u_bar = 0.5*(u[j,i-1] + u[j-1,i-1])
            u_bar_bar = 0.5*(u[j,i-2] + u[j-1,i-2])
            conv_x = -(rho*v[j,i+1]*u_bar - rho*v[j,i-1]*u_bar_bar)/(2*dx)
            
            conv_y = -(rho*v[j+1,i]**2 - rho*v[j-1,i]**2)/(2*dy)
            
            # Viscous terms (include 1/Re)
            visc_x = (mu/rho)*(v[j,i+1] - 2*v[j,i] + v[j,i-1])/dx**2
            visc_y = (mu/rho)*(v[j+1,i] - 2*v[j,i] + v[j-1,i])/dy**2
            
            B_star = conv_x + conv_y + visc_x + visc_y

            # Update v
            v[j,i] = v_old[j,i] + B_star*dt - (dt/dy)*(p_old[j,i-1] - p_old[j-1,i-1])/rho
    
    # Pressure correction 
    p_prime = np.zeros_like(p)
    for k in range(200):  # IRelaxation technique
        for j in range(1, Ny_p-1):
            for i in range(1, Nx_p-1):
                # Calculate coefficients
                a = 2*(dt/dx**2 + dt/dy**2)
                b = -dt/dx**2
                c = -dt/dy**2
                
                # Need to use the updated u and v values
                d = (rho*u[j,i+1] - rho*u[j,i])/dx + (rho*v[j+1,i+1] - rho*v[j,i+1])/dy
                
                # Update pressure correction
                p_prime[j,i] = -(b*p_prime[j,i+1] + b*p_prime[j,i-1] + 
                                c*p_prime[j+1,i] + c*p_prime[j-1,i] + d)/a
        
        # Apply boundary conditions for p_prime
        p_prime[:, 0] = 0    # Inflow
        p_prime[:, -1] = 0   # Outflow
        p_prime[0, :] = 0    # Bottom wall
        p_prime[-1, :] = 0   # Top wall
    
    # Update pressure (Eq. 6.106 in book)
    p = p_old + alpha_p * p_prime
    
    # Velocity correction
    for j in range(1, Ny_p-1):
        for i in range(1, Nx_p):
            u[j,i] = u[j,i] - (dt/dx)*(p_prime[j,i] - p_prime[j,i-1])/rho
    
    for j in range(1, Ny_p):
        for i in range(1, Nx_p+1):
            v[j,i] = v[j,i] - (dt/dy)*(p_prime[j,i-1] - p_prime[j-1,i-1])/rho
    
    # Reapply boundary conditions
    u[-1, :] = u_e
    u[0, :] = 0
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    
    # Monitor convergence (optional)
    if n % 50 == 0:
        print(f"Iteration {n}, Max u change: {np.max(np.abs(u - u_old))}")
    if n in u_steps:
        u_profile[n] = u.copy()
    if n in v_steps:
        v_profile[n] = v.copy()

i_station = 15  # Corresponds to x ≈ 0.35 ft (from book)
plt.figure(figsize=(10, 6))

for step in u_steps:
    plt.plot(u_profile[step][:, i_station], y_u, label=f'Step {step}')

plt.plot([0, u_e], [0, D], 'r--', linewidth=2, label='Exact Couette')
plt.xlabel('u velocity (ft/s)', fontsize=12)
plt.ylabel('y (ft)', fontsize=12)
plt.title(f'Velocity Profile at x = {x_u[i_station]:.2f} ft after {Nt} iterations', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Focus on the v-velocity spike decay at (i=15, j=5)
j_spike = 5  # y-index where spike was initialized
i_spike = 15  # x-index where spike was initialized

plt.figure(figsize=(10, 6))

# Plot v-velocity at spike location over time
steps_to_plot = sorted(v_steps)
v_spike_values = [v_profile[step][j_spike, i_spike] for step in steps_to_plot]

plt.plot(steps_to_plot, v_spike_values, 'bo-', label='v-velocity at spike')
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Time step')
plt.ylabel('v velocity (ft/s)')
plt.title(f'Decay of Initial v-Velocity Spike at (x={x_v[i_spike]:.2f}ft, y={y_v[j_spike]:.2f}ft)')
plt.legend()
plt.grid(True)
plt.show()

# Additional diagnostic: Print spike values
print("v-velocity spike decay:")
for step in steps_to_plot:
    print(f"Step {step}: v = {v_profile[step][j_spike, i_spike]:.6f} ft/s")

plt.show()