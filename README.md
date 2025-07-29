# Couette Flow Simulation using Finite Difference Methods

This project explores the simulation of **Couette flow** using two different numerical methods in Python. All work is based on *"Introduction to Computational Fluid Dynamics"* by John D. Anderson.

---

## What is Couette Flow?

Couette flow refers to viscous flow between two parallel plates, where one plate is moving and the other is stationary. It’s a classic case for testing viscous flow solvers and understanding momentum diffusion.

---

##  Methods Implemented

### 1. Crank-Nicholson Method (1D)
- Solves the unsteady Navier–Stokes equation in 1D.
- Semi-implicit time discretization using **Crank–Nicholson scheme**.
- Results in a **tridiagonal system**, solved using the **Thomas algorithm**.
- Validates velocity development over time until a steady linear profile is reached.

### 2. SIMPLE Algorithm (2D)
- Solves full **2D incompressible Couette flow** using the **SIMPLE algorithm** on a **staggered grid**.
- A **velocity spike** is introduced at the inlet to visualize how disturbances decay in viscous flow.
- Captures the development of the velocity field and illustrates 2D effects not seen in 1D.


