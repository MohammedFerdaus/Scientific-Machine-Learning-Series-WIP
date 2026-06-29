# Project 01 — Harmonic Oscillator

Part of the SciML Foundations series — a structured sequence of scientific machine learning projects built entirely from scratch in Julia.

**Series:** Phase 1 Foundations (Projects 1–7)
1. PINN — Harmonic Oscillator ← you are here
2. NODE — Lotka-Volterra Neural ODE
3. DeepONet — Anti-derivative Operator
4. PINN — 1D Wave Equation
5. FNO — 1D Burgers Equation
6. GNN — Spring-Mass System
7. DeepONet — ODE Solution Operator

---

## The Problem

The simple harmonic oscillator describes any system where a restoring force is proportional to displacement — a mass on a spring, a pendulum at small angles, an LC circuit. The governing equation is:

$$u''(t) + \omega^2 u(t) = 0, \quad t \in [0, T]$$

with initial conditions $u(0) = 1$, $u'(0) = 0$ and analytical solution $u(t) = \cos(\omega t)$.

This project solves it three ways and compares them against the exact solution:

1. Classical RK4 numerical integration
2. Physics-Informed Neural Network with finite difference gradients
3. Physics-Informed Neural Network with a reverse mode autograd engine built from scratch

---

## Repository Structure

```
p01_harmonic_oscillator/
├── numerical_version.jl       — RK4 solver
├── pinn_architecture.jl       — MLP, forward pass, analytical derivatives
├── pinn_train.jl              — PINN with finite difference gradients
├── autograd.jl                — Reverse mode autograd engine
├── pinn_train_autograd.jl     — PINN with autograd gradients
└── test.jl                    — Three-way comparison and plots
```

---

## Method 1 — RK4 Numerical Solver

### The Mathematics

A second order ODE cannot be integrated directly by a stepping method. The first step is to reformulate it as a first order system by introducing velocity $v = u'$ as an auxiliary variable:

$$\frac{d}{dt}\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ -\omega^2 & 0 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix}$$

The matrix on the right is the system matrix $A$. The dynamics are now a linear first order ODE $\dot{\mathbf{s}} = A\mathbf{s}$ which any stepping method can integrate.

The classical fourth order Runge-Kutta method advances the state $\mathbf{s} = [u, v]^T$ one timestep by sampling the derivative at four points — the start, two midpoints, and the end — and taking a weighted average:

$$k_1 = f(\mathbf{s}_n)$$
$$k_2 = f\!\left(\mathbf{s}_n + \tfrac{h}{2}k_1\right)$$
$$k_3 = f\!\left(\mathbf{s}_n + \tfrac{h}{2}k_2\right)$$
$$k_4 = f(\mathbf{s}_n + h\,k_3)$$
$$\mathbf{s}_{n+1} = \mathbf{s}_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

The local truncation error is $O(h^5)$ giving a global error of $O(h^4)$. For the harmonic oscillator this translates to near machine precision accuracy over many cycles.

### Result

| Metric | Value |
|--------|-------|
| Max absolute error | `6.6e-10` |
| RMS error | `3.2e-10` |

---

## Method 2 — PINN with Finite Difference Gradients

### The Physics-Informed Approach

Instead of marching forward in time, a PINN learns a global function $\hat{u}(t)$ that satisfies the ODE everywhere on $[0, T]$ simultaneously. No trajectory data is needed — the physics itself is the training signal.

The network is an MLP with architecture `[1, 32, 32, 32, 1]` — scalar time input, three hidden layers of width 32 with tanh activations, scalar displacement output. Tanh is essential because it is infinitely differentiable, allowing $u''(t)$ to be computed analytically through the network.

### The Loss Function

Training minimizes a loss with two terms:

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_{i=1}^{N}\left[u''(t_i) + \omega^2 u(t_i)\right]^2}_{\mathcal{L}_{\text{physics}}} + \lambda\underbrace{\left[(u(0) - 1)^2 + (u'(0))^2\right]}_{\mathcal{L}_{\text{IC}}}$$

The physics residual penalizes violation of the ODE at $N$ collocation points sampled uniformly on $[0, T]$. If the network were a perfect solution this term would be exactly zero everywhere. The IC residual anchors the solution to the correct initial position and velocity. $\lambda$ controls the relative weight of the two terms.

Time is normalized to $[0, 1]$ before entering the network to reduce spectral bias. The chain rule correction factor $\frac{1}{T^2}$ is applied to $u''$ to account for this rescaling.

### Computing Derivatives Through the Network

The first and second derivatives of the network output with respect to $t$ are computed analytically using the chain rule. Starting from the output and propagating backwards through each layer:

For the first derivative, a sensitivity vector $s$ is initialized to $[1.0]$ and updated at each hidden layer:

$$s \leftarrow s \odot \sigma'(z^{(l)})$$
$$s \leftarrow (W^{(l)})^T s$$

where $\sigma'(z) = 1 - \tanh^2(z)$ is the elementwise tanh derivative. For the second derivative, two signals are propagated simultaneously. The second derivative of tanh is:

$$\sigma''(z) = -2\tanh(z)(1 - \tanh^2(z))$$

At each layer the signals update via the product rule:

$$s'' \leftarrow s'' \odot \sigma'(z) + s' \odot \sigma''(z)$$
$$s' \leftarrow s' \odot \sigma'(z)$$

Both signals then multiply by $(W^{(l)})^T$.

### Gradient Computation

Without an autograd library, gradients of the loss with respect to each weight and bias are approximated using central finite differences:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-5}$$

This requires two full loss evaluations per parameter per epoch — approximately 4400 evaluations for the `[1,32,32,32,1]` network. Correct but slow.

### The Adam Optimizer

Parameters are updated using Adam — an adaptive gradient descent optimizer that maintains running estimates of the first and second moments of the gradient:

$$m_i = \beta_1 m_i + (1 - \beta_1) g_i$$
$$v_i = \beta_2 v_i + (1 - \beta_2) g_i^2$$

Bias corrected estimates account for the zero initialization of $m$ and $v$:

$$\hat{m}_i = \frac{m_i}{1 - \beta_1^t}, \quad \hat{v}_i = \frac{v_i}{1 - \beta_2^t}$$

The parameter update scales each gradient by the inverse of its historical magnitude:

$$\theta_i \leftarrow \theta_i - \alpha \frac{\hat{m}_i}{\sqrt{\hat{v}_i} + \epsilon}$$

Standard values $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ are used throughout.

---

## Method 3 — PINN with Reverse Mode Autograd

### Why Autograd

Central finite differences require two loss evaluations per parameter. For a network with $n$ parameters that is $2n$ forward passes per epoch. Reverse mode automatic differentiation computes exact gradients for all parameters in a single forward pass plus a single backward pass — regardless of parameter count.

### The Tape

Every arithmetic operation during the forward pass registers itself on a tape as a `TapeEntry` storing its output value, the indices of its inputs on the tape, and a backward function encoding the local Jacobian rule. After the forward pass completes, `backward!` seeds the loss node with gradient $1.0$ and walks the tape in reverse, calling each entry's backward function to accumulate gradients:

$$\frac{\partial \mathcal{L}}{\partial u_i} \mathrel{+}= \frac{\partial \mathcal{L}}{\partial v} \cdot \frac{\partial v}{\partial u_i}$$

The local Jacobian rules for each primitive operation are:

| Operation | Forward | Backward |
|-----------|---------|----------|
| $v = a + b$ | $a + b$ | $\bar{a} \mathrel{+}= \bar{v}$, $\bar{b} \mathrel{+}= \bar{v}$ |
| $v = a \cdot b$ | $a \cdot b$ | $\bar{a} \mathrel{+}= \bar{v} b$, $\bar{b} \mathrel{+}= \bar{v} a$ |
| $v = \tanh(a)$ | $\tanh(a)$ | $\bar{a} \mathrel{+}= \bar{v}(1 - v^2)$ |
| $v = a^2$ | $a^2$ | $\bar{a} \mathrel{+}= 2a\bar{v}$ |
| $v = \frac{1}{N}\sum a_i$ | mean | $\bar{a}_i \mathrel{+}= \bar{v}/N$ |

### Second Derivative via Tracked Finite Differences

The second derivative $u''(t)$ in the physics residual is approximated using three tracked forward passes:

$$u''(\tau) \approx \frac{\hat{u}(\tau + \epsilon) - 2\hat{u}(\tau) + \hat{u}(\tau - \epsilon)}{\epsilon^2}$$

All three evaluations go through the tape so gradients flow correctly to all weights and biases. This avoids the tape explosion that results from implementing the full analytical second derivative through tracked primitives.

---

## Results

| Method | Max Error | Notes |
|--------|-----------|-------|
| RK4 | `6.6e-10` | Near machine precision |
| PINN finite differences | `~1.6` | 100 epochs, small network |
| PINN autograd | `~1.07` | 500 epochs |

---

## Possible Improvements

**L-BFGS second order optimizer** — Adam stalls near local minima because it has no curvature information. L-BFGS approximates the inverse Hessian from gradient history using a two loop recursion and uses a line search with Wolfe conditions to guarantee progress every step. Standard PINN implementations use Adam to find the right basin then switch to L-BFGS for final convergence — this is what enables loss below `1e-6`.

**Fourier input features** — the network struggles to learn oscillatory behavior from a raw scalar input due to spectral bias. Replacing $t$ with Fourier encodings $[\cos(2\pi f_k t), \sin(2\pi f_k t)]$ for several frequencies $f_k$ directly addresses this and typically yields an order of magnitude accuracy improvement.

**Adaptive collocation** — rather than fixed uniform collocation points, resampling points in regions of high residual focuses training effort where the network is most wrong and speeds convergence.

**Loss gradient balancing** — instead of a fixed $\lambda$, dynamically adjust the IC weight each epoch based on the ratio of physics to IC gradient norms so neither term dominates.

---

## Libraries

| Purpose | Library |
|---------|---------|
| Linear algebra | `LinearAlgebra` (stdlib) |
| Statistics | `Statistics` (stdlib) |
| Random seeding | `Random` (stdlib) |
| Visualization | `CairoMakie` |

All solvers, neural network layers, derivative computations, autograd engine, and optimizers are implemented from scratch.

---

## How to Run

```bash
julia test.jl
```

Produces `numerical_results.png` and `comparison_results.png`.
