include("log_rk4_solver.jl")
include("true_dynamics.jl")
include("node_architecture.jl")
include("adjoint.jl")
include("node_train.jl")

using CairoMakie
 
function test_true_dynamics()
    alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0
    initial_state = [1.0, 1.0]
    tspan = (0, 3.5)
    dt = 0.01

    trajectory, times = generate_trajectory(initial_state, tspan, dt, alpha, beta, delta, gamma)
    expected_rows = length(times)

    try
        @assert size(trajectory, 1) == expected_rows "shape mismatch: expected $expected_rows rows, got $(size(trajectory, 1))"
        @assert size(trajectory, 2) == 2 "shape mismatch: expected 2 columns (states), got $(size(trajectory, 2))"
        @assert isapprox(trajectory[1, :], initial_state, atol=1e-5) "initial state mismatch: Expected $initial_state, got $(trajectory[1, :])"
        @assert all(trajectory .> 0) "negative population error"

        println("test_true_dynamics: pass")

    catch e
        if isa(e, AssertionError)
            println("test_true_dynamics: fail")
            println("Reason: ", e.msg)
        else
            rethrow(e)
        end
    end
end

function test_node()
    layer_sizes = [2, 16, 16, 2]
    seed = 42
    params = init_params(layer_sizes, seed)

    try
        @assert length(params) == length(layer_sizes) - 1

        for i in 1:length(params)
            W, b = params[i]
            @assert size(W) == (layer_sizes[i+1], layer_sizes[i])
            @assert size(b) == (layer_sizes[i+1],)
        end

        test_state = [1.0, 0.5]
        output = vector_field(test_state, params, :data_driven)

        @assert length(output) == 2
        @assert all(isfinite, output)

        println("test_node: pass")
    catch e
        println("test_node: fail - ", e)
    end
end

function gradient_check()
    layer_sizes = [2, 16, 16, 2]
    params = init_params(layer_sizes, 42)

    initial_state = [1.0, 1.0]
    tspan = (0.0, 2.0)
    dt = 0.1
    trajectory, times = generate_trajectory(initial_state, tspan, dt, 1.5, 1.0, 1.0, 3.0)
    noisy_trajectory = add_gaussian_noise(trajectory, 0.05, 42)

    layer_idx = 1
    row, col = 1, 1
    epsilon = 1e-5

    loss, predicted = compute_loss(params, noisy_trajectory, times)
    grad_params = integrate_adjoint(predicted, noisy_trajectory, times, params)
    analytical_grad = grad_params[layer_idx][1][row, col]

    params_plus = deepcopy(params)
    params_plus[layer_idx][1][row, col] += epsilon
    loss_plus, _ = compute_loss(params_plus, noisy_trajectory, times)

    params_minus = deepcopy(params)
    params_minus[layer_idx][1][row, col] -= epsilon
    loss_minus, _ = compute_loss(params_minus, noisy_trajectory, times)

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    relative_error = abs(analytical_grad - numerical_grad) / (abs(numerical_grad) + 1e-10)

    println("Analytical: ", analytical_grad)
    println("Numerical:  ", numerical_grad)
    println("Relative error: ", relative_error)

    if relative_error < 1e-3
        println("GRADIENT CHECK: PASS")
    else
        println("GRADIENT CHECK: FAIL")
    end
end

function plot_phase_portrait(true_trajectory, learned_trajectory; filename="phase_portrait.png", title="Phase Portrait")
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Prey (x)", ylabel="Predator (y)",
              title=title)

    lines!(ax, true_trajectory[:,1], true_trajectory[:,2], color=:blue, label="True")
    lines!(ax, learned_trajectory[:,1], learned_trajectory[:,2], color=:red, linestyle=:dash, label="Learned")

    axislegend(ax)
    save(filename, fig)

    return fig
end

function plot_vector_field(params, alpha, beta, delta, gamma; filename="vector_field_comparison.png")
    x_range = range(0.1, 4.0, length=15)
    y_range = range(0.1, 4.0, length=15)

    true_u = zeros(length(x_range), length(y_range))
    true_v = zeros(length(x_range), length(y_range))
    learned_u = zeros(length(x_range), length(y_range))
    learned_v = zeros(length(x_range), length(y_range))

    for (i, x) in enumerate(x_range)
        for (j, y) in enumerate(y_range)
            state = [x, y]

            true_deriv = lotka_volterra_dynamics(state, alpha, beta, delta, gamma)
            true_u[i,j] = true_deriv[1]
            true_v[i,j] = true_deriv[2]

            learned_deriv = vector_field(state, params, :data_driven)
            learned_u[i,j] = learned_deriv[1]
            learned_v[i,j] = learned_deriv[2]
        end
    end

    fig = Figure(size=(1000, 500))
    ax1 = Axis(fig[1,1], title="True Vector Field", xlabel="x", ylabel="y")
    ax2 = Axis(fig[1,2], title="Learned Vector Field", xlabel="x", ylabel="y")

    arrows!(ax1, x_range, y_range, true_u, true_v, lengthscale=0.1)
    arrows!(ax2, x_range, y_range, learned_u, learned_v, lengthscale=0.1)

    save(filename, fig)
    return fig
end

function plot_loss_curve(loss_history; filename="loss_curve.png", title="Training Loss")
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Epoch", ylabel="Loss", title=title, yscale=log10)
    lines!(ax, 1:length(loss_history), loss_history, color=:blue)

    save(filename, fig)
    return fig
end

# Run tests
test_true_dynamics()
test_node()
gradient_check()

# Adam baseline run (3.5s horizon)
alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0
initial_state = [1.0, 1.0]
tspan = (0, 3.5)
dt = 0.01

true_trajectory, times = generate_trajectory(initial_state, tspan, dt, alpha, beta, delta, gamma)
noisy_trajectory = add_gaussian_noise(true_trajectory, 0.05, 42)

layer_sizes = [2, 32, 2]
num_epochs = 5000
learning_rate = 0.01

learned_params, loss_history = train(noisy_trajectory, times, num_epochs, learning_rate, layer_sizes, 42)

dynamics_fn = (state, t) -> vector_field(state, learned_params, :data_driven)
learned_trajectory = integrate_forward(initial_state, times, dynamics_fn)

plot_phase_portrait(true_trajectory, learned_trajectory; filename="phase_portrait_adam.png", title="Adam Phase Portrait (3.5s)")
plot_vector_field(learned_params, alpha, beta, delta, gamma)
plot_loss_curve(loss_history; filename="loss_curve_adam.png", title="Adam Training Loss")
