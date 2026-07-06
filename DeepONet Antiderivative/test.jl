include("deeponet.jl")
include("deeponet_train.jl")
include("test_functions.jl")

using CairoMakie

function evaluate_on_test_function(net::DeepONet, test_function, y_grid)
    sensor_values = evaluate_u_at_sensors(test_function)
    
    predicted = [forward_deeponet(net, sensor_values, y) for y in y_grid]
    
    true_values = evaluate_antiderivative(test_function, y_grid)

    return predicted, true_values
end
 
function compute_error_metrics(predicted, true_values)
    errors = predicted .- true_values
    max_absolute_error = maximum(abs.(errors))
    mean_squared_error = sum(errors.^2) / length(errors)
    
    return (max_absolute_error, mean_squared_error)
end

function plot_function_class(class_name, labeled_functions, net::DeepONet, y_grid)
    fig = Figure(size = (1200, 400))
    
    for (i, (label, test_function)) in enumerate(labeled_functions)
        predicted, true_values = evaluate_on_test_function(net, test_function, y_grid)
        max_error, mse = compute_error_metrics(predicted, true_values)
        println(label, ": max error = ", max_error, ", MSE = ", mse)

        ax = Axis(fig[1, i], title=label, xlabel="y", ylabel="G(u)(y)")
        lines!(ax, y_grid, true_values, label="True", color=:blue)
        lines!(ax, y_grid, predicted, label="Predicted", color=:red, linestyle=:dash)
        axislegend(ax)
    end
    
    save("test_$(class_name).png", fig)
    
    return fig
end

function run_full_test(net::DeepONet)
    test_set = get_fixed_test_set()
    y_grid = range(0, 1, length=100)

    constant_functions   = [(label, f) for (label, f) in test_set if f isa ConstantFunction]
    sinusoid_functions   = [(label, f) for (label, f) in test_set if f isa SinusoidFunction]
    polynomial_functions = [(label, f) for (label, f) in test_set if f isa PolynomialFunction]

    plot_function_class("constant", constant_functions, net, y_grid)
    plot_function_class("sinusoid", sinusoid_functions, net, y_grid)
    plot_function_class("polynomial", polynomial_functions, net, y_grid)
end

net_bigger = train_deeponet(200, 5, 5, 1.0, 0, 2e-3, 0.01, 5000)
run_full_test(net_bigger)
