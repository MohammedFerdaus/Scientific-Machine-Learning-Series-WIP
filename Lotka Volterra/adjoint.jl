using LinearAlgebra

function compute_jacobian(state, params)
    activations = [state]
    pre_activations = []

    for (W, b) in params[1:end-1]
        z_pre = W * activations[end] + b
        push!(pre_activations, z_pre)
        push!(activations, tanh.(z_pre))
    end

    (W_out, b_out) = params[end]
    delta = W_out

    for i in length(pre_activations):-1:1
        sech2 = 1 .- tanh.(pre_activations[i]).^2
        delta = (delta .* sech2') * params[i][1]
    end

    jacobian_z = delta

    jacobian_theta = Vector{Tuple{Matrix{Float64}, Vector{Float64}}}()

    sensitivity = Matrix{Float64}(I, 2, 2)
    push!(jacobian_theta, (sensitivity, activations[end]))

    for i in length(pre_activations):-1:1
        sech2 = 1 .- tanh.(pre_activations[i]).^2
        sensitivity = (params[i+1][1]' * sensitivity) .* sech2
        push!(jacobian_theta, (sensitivity, activations[i]))
    end

    reverse!(jacobian_theta)

    return jacobian_z, jacobian_theta
end

function adjoint_dynamics(z, adjoint, params)
    jacobian_z, jacobian_theta = compute_jacobian(z, params)

    dz = vector_field(z, params, :data_driven)
    da = -(jacobian_z' * adjoint)

    grad_params = []
    for (sensitivity, activation_in) in jacobian_theta
    contracted = sensitivity * adjoint  
    dW = -(contracted * activation_in')
    db = -contracted
    push!(grad_params, (dW, db))
end

    return dz, da, grad_params
end

function rk4_step_augmented(z, adjoint, grad_params, dt, params)
    h = -dt
    dz1, da1, dgrad1 = adjoint_dynamics(z, adjoint, params)

    z2 = z + 0.5*h*dz1
    a2 = adjoint + 0.5*h*da1
    dz2, da2, dgrad2 = adjoint_dynamics(z2, a2, params)

    z3 = z + 0.5*h*dz2
    a3 = adjoint + 0.5*h*da2
    dz3, da3, dgrad3 = adjoint_dynamics(z3, a3, params)

    z4 = z + h*dz3
    a4 = adjoint + h*da3
    dz4, da4, dgrad4 = adjoint_dynamics(z4, a4, params)

    z_new = z + (h/6) * (dz1 + 2*dz2 + 2*dz3 + dz4)
    adjoint_new = adjoint + (h/6) * (da1 + 2*da2 + 2*da3 + da4)

    grad_params_new = []
    for l in 1:length(params)
        dW = (h/6) * (dgrad1[l][1] + 2*dgrad2[l][1] + 2*dgrad3[l][1] + dgrad4[l][1])
        db = (h/6) * (dgrad1[l][2] + 2*dgrad2[l][2] + 2*dgrad3[l][2] + dgrad4[l][2])
        push!(grad_params_new, (grad_params[l][1] + dW, grad_params[l][2] + db))
    end

    return z_new, adjoint_new, grad_params_new
end

function integrate_adjoint(predicted, observed, times, params)
    N = length(times)
    z = predicted[N, :]
    adjoint = zeros(2)
    grad_params = [(zeros(size(W)), zeros(size(b))) for (W,b) in params]

    for i in N:-1:2
        dt = times[i] - times[i-1]
        adjoint = adjoint + (2/N) * (predicted[i, :] - observed[i, :])
        z, adjoint, grad_params = rk4_step_augmented(z, adjoint, grad_params, dt, params)
    end
    
    return grad_params
end