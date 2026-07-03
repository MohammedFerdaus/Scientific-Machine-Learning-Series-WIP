# Project 02 — Lotka-Volterra Neural ODE

Part of the SciML Foundations series — a structured sequence of scientific machine learning projects built entirely from scratch in Julia.
Series: Phase 1 Foundations (Projects 1–7)
*   PINN — Harmonic Oscillator
*   NODE — Lotka-Volterra Neural ODE ← you are here
*   DeepONet — Anti-derivative Operator
*   PINN — 1D Wave Equation
*   FNO — 1D Burgers Equation
*   GNN — Spring-Mass System
*   DeepONet — ODE Solution Operator

---

## The Problem

The Lotka-Volterra equations describe the non-linear dynamics of biological systems where two species interact as predator $y$ and prey $x$. The governing system of non-linear differential equations is given by:

$$\frac{dx}{dt} = \alpha x - \beta xy$$

$$\frac{dy}{dt} = \delta xy - \gamma y$$

where $\alpha, \beta, \delta, \gamma$ are parameters defining growth, predation, and mortality rates. 

This project simulates the true underlying ecosystem trajectory, injects artificial experimental noise, and uses a Neural Ordinary Differential Equation (NODE) to learn the latent vector field directly from the noisy time-series observations without assuming the physical functional forms beforehand.

---

## Repository Structure

```text
p02_lotka_volterra/
├── log_rk4_solver.jl          — Classical RK4 and log-space forward integrators
├── true_dynamics.jl           — Lotka-Volterra vector fields and data generation
├── node_architecture.jl       — Multi-layer perceptron and data-driven vector fields
├── adjoint.jl                 — Custom backpropagation and augmented adjoint ODE solvers
├── node_train.jl              — Mean squared error loss tracking and Adam optimizer
└── test.jl                    — Unit tests, analytic gradient verification, and plotting
```

---

## Method 1 — Classical RK4 & Log-RK4 Numerical Solvers

### The Mathematics

A classical fourth-order Runge-Kutta method (RK4) steps a state forward in time by taking a weighted average of four directional derivative samples across a timestep $\Delta t$:

$$k_1 = f(s_n, t_n)$$

$$k_2 = f\left(s_n + \frac{\Delta t}{2}k_1, t_n + \frac{\Delta t}{2}\right)$$

$$k_3 = f\left(s_n + \frac{\Delta t}{2}k_2, t_n + \frac{\Delta t}{2}\right)$$

$$k_4 = f(s_n + \Delta t k_3, t_n + \Delta t)$$

$$s_{n+1} = s_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

To prevent negative population values and gracefully handle extreme exponential scaling, a log-transformed variant is implemented. Substituting $u = \log(x)$ and $v = \log(y)$, the dynamics map to:

$$\frac{du}{dt} = \alpha - \beta e^v$$

$$\frac{dv}{dt} = \delta e^u - \gamma$$

Integrating in log-space ensures structural positivity ($x, y > 0$) upon taking the elementwise exponential $\exp([u, v]^T)$.

---

## Method 2 — Neural ODE via Adjoint Sensitivity Method

### The Data-Driven Vector Field

Instead of enforcing a rigid structural equation, an MLP network acts as a universal function approximator to parameterize the latent vector field:

$$\frac{dz}{dt} = f_\theta(z)$$

The network features an architecture of `[2, 32, 2]`—taking a 2D state input vector ($x, y$), processing it through a hidden layer with `tanh` activations, and producing a 2D derivative output.

### The Adjoint Engine (Manual Backpropagation)

To avoid tracking a massive computational tape across thousands of forward time steps, gradients are propagated backwards using the Adjoint Sensitivity Method. The engine uses a custom backward pass built completely from scratch:

1. **Jacobian Evaluation:** The network's layer-by-layer analytical Jacobian with respect to the state ($z$) and parameters ($\theta$) is evaluated manually via backpropagation matrix products.
2. **Augmented Dynamics:** An augmented ODE system is solved backwards in time (stepping $h = -\Delta t$) to compute the simultaneous evolution of the state $z$, the adjoint state $a$, and parameter sensitivities $a_\theta$:

$$\frac{dz}{dt} = f_\theta(z)$$

$$\frac{da}{dt} = -\left(\frac{\partial f_\theta}{\partial z}\right)^T a$$

$$\frac{d a_\theta}{dt} = -a^T \left(\frac{\partial f_\theta}{\partial \theta}\right)$$

3. **Augmented RK4 Step:** Both state and adjoint fields are unified and marched backward in time using a custom-built reverse-time integrator `rk4_step_augmented`.

### Training and Loss

The model tracks a global Mean Squared Error (MSE) loss against noisy trajectory observations:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \| z_{\text{pred}}(t_i) - z_{\text{obs}}(t_i) \|^2$$

Optimization is handled over **5000 epochs** via a manual Adam implementation implementing bias correction:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## Results

### Verification & Performance

| Test Component | Status | Metric | Notes |
| :--- | :--- | :--- | :--- |
| **True Dynamics Test** | Pass | State Positivity ($>0$) | Validates structural integrity of the baseline data generator. |
| **Neural Network Topology** | Pass | Shape Match | Confirms parameter configurations match layer structures exactly. |
| **Gradient Validation** | Pass | Relative Error $< 10^{-3}$ | Cross-checks analytical adjoint gradients against finite difference approximations. |
| **Training Execution** | Pass | Final Loss $\approx 6.8 \times 10^{-3}$ | Stabilizes within 5000 epochs under a 0.01 learning rate. |

---

## Possible Improvements

*   **Mini-batching & Sub-sequencing:** Training on long time horizons introduces severe non-linear gradient explosions. Splitting the full trajectory into multiple shorter overlapping sub-sequences relaxes the optimization landscape.
*   **Log-Space Neural ODE Formulation:** Enforcing the Neural ODE vector field inside log-coordinates directly structuralizes positivity inside the data-driven model, eliminating non-physical negative species predictions during early exploration.
*   **Regularization on Vector Field Jacobians:** Penalizing the Frobenius norm of the network's Jacobian $\left\|\frac{\partial f_\theta}{\partial z}\right\|_F$ forces the system to find smoother, more stable velocity fields that handle noise robustly.

---

## Libraries

*   `LinearAlgebra` — Matrix operations and structural identity setups.
*   `Statistics` — Mean loss computing operations.
*   `Random` — Pseudo-random parameter initialization and reproducible Gaussian noise injection.
*   `CairoMakie` — Clean rendering of vector fields, phase portraits, and loss progression.
 
 All neural architecture parameters, forward trajectories, adjoint engines, gradient verifications, and optimization loops are constructed purely from scratch.

---

## How to Run

Execute the validation pipeline directly from your terminal:

```bash
julia test.jl
```

Running this updates unit tests, runs the analytical gradient check, and writes `loss_curve_adam.png`, `phase_portrait_adam.png`, and `vector_field_comparison.jpg` to your directory.
