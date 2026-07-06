include("function_sampler.jl")

struct ConstantFunction
    c::Float64
end

struct SinusoidFunction
    k::Int
end

struct PolynomialFunction 
    n::Int
end

function evaluate_u(f::ConstantFunction, t)
    return f.c .+ zero(t)
end
 
function evaluate_u(f::SinusoidFunction, t)
    return sin.(2 * pi * f.k .* t)
end

function evaluate_u(f::PolynomialFunction, t)
    return t .^ f.n
end

function evaluate_antiderivative(f::ConstantFunction, y)
    return f.c .* y
end

function evaluate_antiderivative(f::SinusoidFunction, y)
    return (1 .- cos.(2 * pi * f.k .* y)) ./ (2 * pi * f.k)
end

function evaluate_antiderivative(f::PolynomialFunction, y)
    return (y .^ (f.n + 1)) ./ (f.n + 1)
end

function evaluate_u_at_sensors(f)
    return evaluate_u(f, sensor_points)
end

function get_fixed_test_set()
    return [
        ("constant_c=0.5", ConstantFunction(0.5)),
        ("constant_c=1.5", ConstantFunction(1.5)),
        ("constant_c=-1.0", ConstantFunction(-1.0)),

        ("sinusoid_k=1", SinusoidFunction(1)),
        ("sinusoid_k=3", SinusoidFunction(3)),
        ("sinusoid_k=5", SinusoidFunction(5)),

        ("polynomial_n=1", PolynomialFunction(1)),
        ("polynomial_n=2", PolynomialFunction(2)),
        ("polynomial_n=3", PolynomialFunction(3))
    ]
end
