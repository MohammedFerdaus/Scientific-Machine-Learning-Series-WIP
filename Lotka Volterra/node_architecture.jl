using Random

function init_params(layer_sizes, random_seed)
    rng = MersenneTwister(random_seed)
    params = Vector{Tuple{Matrix{Float64}, Vector{Float64}}}()
    
    for i in 1:(length(layer_sizes) - 1)
        n_in = layer_sizes[i]
        n_out = layer_sizes[i+1]
        
        limit = sqrt(6.0 / (n_in + n_out))

        w = -limit .+ (2 * limit) .* rand(rng, n_out, n_in)
        b = zeros(n_out)

        push!(params, (w, b))
    end

    return params
end

function forward(state_input, params)
    activation = state_input

    for (W, b) in params[1:end-1]
        activation = tanh.(W * activation + b)
    end

    (W_out, b_out) = params[end]
    output = W_out * activation + b_out

    return output
end

function vector_field(state, params, mode, dynamics_fn=nothing)
    if mode == :data_driven
        return forward(state, params)
    end

    if mode == :solver
        return dynamics_fn(state)
    end
end