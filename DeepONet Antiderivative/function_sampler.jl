using Random

const sensor_points = range(0, 1, length = 100)

struct FourierFunctionSample{T<:Real}
    a0::T
    a_coeffs::Vector{T}
    b_coeffs::Vector{T}
    K::Int
end

function sample_fourier_coefficients(K, sigma, decay_power)
    a0 = sigma * randn()
    
    a_coeffs = [(sigma / k^decay_power) * randn() for k in 1:K]
    b_coeffs = [(sigma / k^decay_power) * randn() for k in 1:K]

    return FourierFunctionSample(a0, a_coeffs, b_coeffs, K)
end
 
function evaluate_u(sample::FourierFunctionSample, t)
    result = sample.a0 .+ zero(t) 
    
    for k in 1:sample.K
        result .+= sample.a_coeffs[k] .* cos.(2 * pi * k .* t)
        result .+= sample.b_coeffs[k] .* sin.(2 * pi * k .* t)
    end

    return result
end

function evaluate_u_at_sensors(sample::FourierFunctionSample)
    return evaluate_u(sample, sensor_points)
end

function generate_function_batch(num_functions, K, sigma, decay_power)
    samples = [sample_fourier_coefficients(K, sigma, decay_power) for _ in 1:num_functions]
    return samples
end
