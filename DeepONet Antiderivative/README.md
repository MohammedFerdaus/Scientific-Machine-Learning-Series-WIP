# Project 03 — DeepONet: Anti-derivative Operator

Part of the SciML Foundations series — a structured sequence of scientific machine learning projects built entirely from scratch in Julia.

**Series:** Phase 1 Foundations (Projects 1–7)
1. PINN — Harmonic Oscillator
2. NODE — Lotka-Volterra Neural ODE
3. DeepONet — Anti-derivative Operator ← you are here
4. PINN — 1D Wave Equation
5. FNO — 1D Burgers Equation
6. GNN — Spring-Mass System
7. DeepONet — ODE Solution Operator

---

## The Problem

A Deep Operator Network (DeepONet) learns a mapping between functions, not between points — the input is an entire function, the output is another function evaluated at a query location. The simplest possible test case is the antiderivative operator:

$$G(u)(y) = \int_0^y u(t)\,dt, \quad y \in [0,1]$$

Input functions $u$ are sampled as random truncated Fourier series and evaluated at $m = 100$ fixed sensor points on $[0,1]$. Given those sensor values and a query point $y$, the network predicts the scalar $G(u)(y)$.

This project builds the branch/trunk architecture from scratch, verifies its gradients numerically, and investigates how the training function distribution and network capacity affect generalization to function shapes the network was not trained on.

---

## Repository Structure

```
p03_deeponet_antiderivative/
├── function_sampler.jl         — Truncated Fourier series random function generation and sensor evaluation
├── true_antiderivative.jl      — Closed-form antiderivative ground truth for sampled Fourier functions
├── test_functions.jl           — Fixed benchmark set (constant, sinusoid, polynomial) with exact antiderivatives
├── branch_net.jl               — MLP mapping sensor values to branch coefficients
├── trunk_net.jl                — MLP mapping a query point to trunk coefficients
├── deeponet.jl                 — Combines branch and trunk outputs via dot product plus bias
├── deeponet_train.jl           — Analytic backpropagation, Adam optimizer, and training loop
└── test.jl                     — Evaluation on fixed test functions, plotting predicted vs. true antiderivatives
```

---

## Method — DeepONet Architecture

### The Mathematics

Random training functions are generated as truncated Fourier series:

$$u(t) = a_0 + \sum_{k=1}^{K}\left[a_k\cos(2\pi k t) + b_k\sin(2\pi k t)\right]$$

with coefficients drawn as $a_0 \sim \mathcal{N}(0,\sigma^2)$ and $a_k, b_k \sim \mathcal{N}(0, \sigma^2/k^{2p})$, where $p$ is a decay power controlling how quickly higher harmonics are suppressed (see "Hyperparameter Investigation" below). Since integration is linear, the antiderivative is exact and closed-form:

$$G(u)(y) = a_0 y + \sum_{k=1}^{K}\left[\frac{a_k}{2\pi k}\sin(2\pi k y) - \frac{b_k}{2\pi k}\left(\cos(2\pi k y) - 1\right)\right]$$

The branch net maps the sensor-value vector to a $p$-dimensional coefficient vector, and the trunk net maps the query point $y$ to a $p$-dimensional basis vector:

$$\mathbf{b} = \text{MLP}_{\text{branch}}\big([u(x_1),\dots,u(x_m)]\big), \qquad \mathbf{t} = \text{MLP}_{\text{trunk}}(y)$$

The prediction is their dot product plus a learned scalar bias:

$$\hat{G}(u)(y) = \sum_{i=1}^{p} b_i t_i + \beta$$

This structure follows the universal approximation theorem for operators (Chen & Chen, 1995): any continuous operator can be approximated by a sum of products of a function of the input (branch) and a function of the query location (trunk).

Both networks use 3 hidden layers with ReLU activations and He initialization. The branch net has no activation on its output layer, since it produces unconstrained coefficients. The trunk net applies ReLU on its output layer as well, since it is producing basis function values.

### Backpropagation

Given the loss gradient at the prediction, $\delta_{\text{out}} = \frac{\partial \mathcal{L}}{\partial \hat{G}}$, the gradient seeding each network is the *other* network's forward output, scaled by $\delta_{\text{out}}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \delta_{\text{out}} \cdot \mathbf{t}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{t}} = \delta_{\text{out}} \cdot \mathbf{b}, \qquad \frac{\partial \mathcal{L}}{\partial \beta} = \delta_{\text{out}}$$

From there, standard layer-by-layer backpropagation applies to each MLP independently, using $\sigma'(z) = \mathbb{1}[z > 0]$ for the ReLU derivative. The branch net's final layer passes its seed gradient straight through with no gating (no activation); the trunk net's final layer gates through $\sigma'(z^{(4)})$ (ReLU is applied there).

Analytic gradients were verified against a central finite-difference check on a branch net weight before being trusted for training:

| Check | Value |
|-------|-------|
| Analytic gradient | `0.5958372284442722` |
| Numerical gradient | `0.5958372284453617` |
| Difference | `1.09e-12` |

### The Adam Optimizer

Parameters are updated using Adam, implemented from scratch, with full-batch gradient accumulation (one update per epoch across all training pairs):

$$m_i = \beta_1 m_i + (1-\beta_1)g_i, \qquad v_i = \beta_2 v_i + (1-\beta_2)g_i^2$$
$$\hat{m}_i = \frac{m_i}{1-\beta_1^t}, \qquad \hat{v}_i = \frac{v_i}{1-\beta_2^t}$$
$$\theta_i \leftarrow \theta_i - \alpha\frac{\hat{m}_i}{\sqrt{\hat{v}_i}+\epsilon}$$

Standard values $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ are used throughout.

---

## What Actually Ran

- Training data: 200 randomly sampled functions per epoch, 5 query points per function (1000 training pairs per epoch), regenerated fresh every epoch.
- Fixed test set: 9 functions across 3 classes — constant $c \in \{0.5, 1.5, -1.0\}$, sinusoid $k \in \{1,3,5\}$, polynomial $n \in \{1,2,3\}$ — each with an exact closed-form antiderivative, evaluated on a 100-point grid.
- Final configuration: hidden width = 256, $p$ = 256, decay power = 0 (flat harmonic variance), learning rate = `2e-3`. Converged at epoch 4038 (loss threshold `0.01`, reached loss `0.0098`) out of a 5000-epoch cap.

### Result

| Test Function | Max Error | MSE |
|---|---|---|
| constant, c=0.5 | `0.0231` | `6.92e-5` |
| constant, c=1.5 | `0.0717` | `6.63e-4` |
| constant, c=-1.0 | `0.0576` | `7.88e-4` |
| sinusoid, k=1 | `0.0664` | `1.32e-3` |
| sinusoid, k=3 | `0.0759` | `1.05e-3` |
| sinusoid, k=5 | `0.0837` | `1.23e-3` |
| polynomial, n=1 | `0.0406` | `5.01e-4` |
| polynomial, n=2 | `0.0411` | `4.44e-4` |

---

## Known Bugs Fixed During Development

- `evaluate_antiderivative` in `true_antiderivative.jl` originally used in-place broadcasting assignment (`.+=`, `.-=`) on a variable that could be a plain scalar when `y` was a single query point. Julia scalars are immutable, so this crashed with a `MethodError` on `copyto!` the first time training called the function with a scalar `y`. Fixed by switching to plain reassignment (`result = result .+ (...)`), which works for both scalar and vector inputs.
- `decay_power` threading through `sample_fourier_coefficients` and `generate_function_batch` was edited in conversation but the edits did not initially land in the actual files, causing `MethodError`s from argument-count mismatches further down the call chain. Resolved by re-confirming every function signature in the chain against the file contents directly.

---

## Hyperparameter Investigation: Fourier Decay Rate and Network Capacity

The original design used variance decaying as $1/k^2$ across harmonics, matching typical practice for generating smooth random functions. Testing against the fixed benchmark set revealed a weakness: single-frequency test sinusoids at $k=3$ and $k=5$ were predicted poorly, losing the oscillation entirely past the first cycle or two, because those harmonics are rare and low-amplitude in a $1/k^2$-decayed training distribution.

Three decay rates were compared at fixed network size (hidden width = 128, $p$ = 128, 500 epochs each):

| Decay power | Low-k / constants | High-k sinusoids | Polynomials (out-of-distribution) |
|---|---|---|---|
| 1 (original, $1/k^2$ variance) | Best | Worst | Worst |
| 0.5 | Middling, inconsistent | Best of the three | Worst |
| 0 (flat variance) | Worst on some constants | Good | Best |

No decay rate was uniformly better; each represented a real tradeoff between accuracy on in-distribution low-frequency content and generalization to higher-frequency or non-Fourier shapes, at this network size.

Increasing network capacity (hidden width and $p$ both raised from 128 to 256, decay power fixed at 0, learning rate raised from `1e-3` to `2e-3`, trained to convergence at 5000 epochs) resolved most of this tradeoff — the larger, fully-converged network performed within a tight band (max error roughly 0.02–0.08) across all 9 test functions, with no category showing the severe failure seen in any single smaller-network configuration. This is the configuration shipped in this project.

**Caveats:**
- This conclusion is based on single runs per configuration, not averaged across multiple random seeds. Run-to-run noise was substantial — two identical-configuration runs at decay power 1 varied by roughly 50% on several test functions.
- The larger network required roughly 8x the epochs and a higher learning rate to reach convergence, compared to the smaller network's 500-epoch runs (which themselves had not converged to the loss threshold).
- Sinusoid $k=5$ and constant $c=1.5$ remain the weakest categories in the final configuration — improved relative to earlier configurations, but not eliminated.

---

## Possible Improvements

**Systematic multi-seed sweep** — the decay power and capacity comparisons above are each based on a single run. A proper sweep averaged over several random seeds per configuration would separate genuine architectural effects from run-to-run training noise, which was shown to be substantial.

**Isolating decay power from capacity** — the final configuration changed decay power and network capacity together. A controlled study varying one at a time at the larger network size would clarify how much each change contributed independently.

**Additional test function classes** — a Gaussian bump and a two-frequency sinusoid sum were considered for the fixed test set but deferred. Both would further probe generalization beyond the Fourier training distribution.

**L-BFGS second order optimizer** — as in Project 01, Adam was used throughout; L-BFGS for final-stage convergence was deferred, and is particularly relevant to Ferrum's eventual surrogate and preconditioner training needs.

**Chebyshev sensor placement** — sensors were placed uniformly on $[0,1]$; comparing against Chebyshev-spaced sensors (an explicitly suggested extension in the original project spec) was not investigated.

---

## Libraries

| Purpose | Library |
|---|---|
| Linear algebra | `LinearAlgebra` (stdlib) |
| Statistics | `Statistics` (stdlib) |
| Random seeding | `Random` (stdlib) |
| Visualization | `CairoMakie` |

All neural network layers, backpropagation, and the Adam optimizer are implemented from scratch.

---

## How to Run

```bash
julia test.jl
```

Produces `test_constant.png`, `test_sinusoid.png`, and `test_polynomial.png`.
