function evaluate_antiderivative(sample::FourierFunctionSample, y)
    result = sample.a0 .* y

    for k in 1:sample.K
        result = result .+ (sample.a_coeffs[k] / (2*pi*k)) .* sin.(2*pi*k .* y)
        result = result .- (sample.b_coeffs[k] / (2*pi*k)) .* (cos.(2*pi*k .* y) .- 1)
    end

    return result
end