include("deeponet.jl")
include("function_sampler.jl")
include("true_antiderivative.jl")

struct BranchNetGradients{T<:Real}
    dW1::Matrix{T}
    db1::Vector{T}
    dW2::Matrix{T}
    db2::Vector{T}
    dW3::Matrix{T}
    db3::Vector{T}
    dW4::Matrix{T}
    db4::Vector{T}
end
 
struct TrunkNetGradients{T<:Real}
    dW1::Matrix{T}
    db1::Vector{T}
    dW2::Matrix{T}
    db2::Vector{T}
    dW3::Matrix{T}
    db3::Vector{T}
    dW4::Matrix{T}
    db4::Vector{T}
end

function relu_derivative(z)
    return Float64.(z .> 0)
end

function backward_branch(net, sensor_values, seed_gradient)
    z1 = net.W1 * sensor_values .+ net.b1
    a1 = relu.(z1)
    
    z2 = net.W2 * a1 .+ net.b2
    a2 = relu.(z2)

    z3 = net.W3 * a2 .+ net.b3
    a3 = relu.(z3)

    z4 = net.W4 * a3 .+ net.b4
    a4 = z4

    delta4 = seed_gradient
    dW4 = delta4 * a3'
    db4 = delta4

    delta3_a = net.W4' * delta4
    delta3_z = delta3_a .* relu_derivative(z3)
    dW3 = delta3_z * a2'
    db3 = delta3_z

    delta2_a = net.W3' * delta3_z
    delta2_z = delta2_a .* relu_derivative(z2)
    dW2 = delta2_z * a1'
    db2 = delta2_z
    
    delta1_a = net.W2' * delta2_z
    delta1_z = delta1_a .* relu_derivative(z1)
    dW1 = delta1_z * sensor_values'
    db1 = delta1_z

    return BranchNetGradients(dW1, db1, dW2, db2, dW3, db3, dW4, db4)
end

function backward_trunk(net, y, seed_gradient)
    y_vec = [y]
    
    z1 = net.W1 * y_vec .+ net.b1
    a1 = relu.(z1)
    
    z2 = net.W2 * a1 .+ net.b2
    a2 = relu.(z2)
    
    z3 = net.W3 * a2 .+ net.b3
    a3 = relu.(z3)
    
    z4 = net.W4 * a3 .+ net.b4
    a4 = relu.(z4) # activation on output layer, unlike branch

    delta4_a = seed_gradient
    delta4_z = delta4_a .* relu_derivative(z4) 
    dW4 = delta4_z * a3'
    db4 = delta4_z

    delta3_a = net.W4' * delta4_z
    delta3_z = delta3_a .* relu_derivative(z3)
    dW3 = delta3_z * a2'
    db3 = delta3_z

    delta2_a = net.W3' * delta3_z
    delta2_z = delta2_a .* relu_derivative(z2)
    dW2 = delta2_z * a1'
    db2 = delta2_z

    delta1_a = net.W2' * delta2_z
    delta1_z = delta1_a .* relu_derivative(z1)
    dW1 = delta1_z * y_vec'
    db1 = delta1_z

    return TrunkNetGradients(dW1, db1, dW2, db2, dW3, db3, dW4, db4)
end

mutable struct AdamState
    m::Vector{Any}
    v::Vector{Any}
    t::Int
end

function init_adam_state(net::DeepONet)
    params = [
        net.branch.W1, net.branch.b1, net.branch.W2, net.branch.b2,
        net.branch.W3, net.branch.b3, net.branch.W4, net.branch.b4,
        net.trunk.W1,  net.trunk.b1,  net.trunk.W2,  net.trunk.b2,
        net.trunk.W3,  net.trunk.b3,  net.trunk.W4,  net.trunk.b4,
        net.bias # include scalar bias at the end
    ]
    
    m = [zero(p) for p in params]
    v = [zero(p) for p in params]

    return AdamState(m, v, 0)
end

function adam_update!(param, grad, m, v, t, learning_rate, beta1, beta2, epsilon)
    m .= beta1 .* m .+ (1 - beta1) .* grad
    v .= beta2 .* v .+ (1 - beta2) .* grad.^2

    m_hat = m ./ (1 - beta1^t)
    v_hat = v ./ (1 - beta2^t)

    param .-= learning_rate .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
end

function generate_training_batch(num_functions, queries_per_function, K, sigma, decay_power)
    function_samples = generate_function_batch(num_functions, K, sigma, decay_power)
    training_pairs = []
    
    for sample in function_samples
        sensor_values = evaluate_u_at_sensors(sample)
        for q in 1:queries_per_function
            y = rand() 
            target = evaluate_antiderivative(sample, y)
            
            push!(training_pairs, (sensor_values, y, target))
        end
    end

    return training_pairs
end

function compute_batch_gradients(net::DeepONet, training_pairs)
    dW1_b, db1_b = zero(net.branch.W1), zero(net.branch.b1)
    dW2_b, db2_b = zero(net.branch.W2), zero(net.branch.b2)
    dW3_b, db3_b = zero(net.branch.W3), zero(net.branch.b3)
    dW4_b, db4_b = zero(net.branch.W4), zero(net.branch.b4)

    dW1_t, db1_t = zero(net.trunk.W1), zero(net.trunk.b1)
    dW2_t, db2_t = zero(net.trunk.W2), zero(net.trunk.b2)
    dW3_t, db3_t = zero(net.trunk.W3), zero(net.trunk.b3)
    dW4_t, db4_t = zero(net.trunk.W4), zero(net.trunk.b4)

    bias_grad = 0.0
    total_loss = 0.0
    N = length(training_pairs)

    for (sensor_values, y, target) in training_pairs
        b = forward_branch(net.branch, sensor_values)
        trunk_output  = forward_trunk(net.trunk, y)
        prediction = sum(b .* trunk_output ) + net.bias

        total_loss += (prediction - target)^2
        
        delta_out = (2 / N) * (prediction - target)

        seed_gradient_branch = delta_out .* trunk_output 
        seed_gradient_trunk = delta_out .* b

        b_grads = backward_branch(net.branch, sensor_values, seed_gradient_branch)
        t_grads = backward_trunk(net.trunk, y, seed_gradient_trunk)

        dW1_b .+= b_grads.dW1; db1_b .+= b_grads.db1
        dW2_b .+= b_grads.dW2; db2_b .+= b_grads.db2
        dW3_b .+= b_grads.dW3; db3_b .+= b_grads.db3
        dW4_b .+= b_grads.dW4; db4_b .+= b_grads.db4

        dW1_t .+= t_grads.dW1; db1_t .+= t_grads.db1
        dW2_t .+= t_grads.dW2; db2_t .+= t_grads.db2
        dW3_t .+= t_grads.dW3; db3_t .+= t_grads.db3
        dW4_t .+= t_grads.dW4; db4_t .+= t_grads.db4

        bias_grad += delta_out
    end

    branch_grads = BranchNetGradients(dW1_b, db1_b, dW2_b, db2_b, dW3_b, db3_b, dW4_b, db4_b)
    trunk_grads = TrunkNetGradients(dW1_t, db1_t, dW2_t, db2_t, dW3_t, db3_t, dW4_t, db4_t)

    return (branch_grads, trunk_grads, bias_grad), (total_loss / N)
end

function train_deeponet(num_functions, queries_per_function, K, sigma,
                        decay_power, learning_rate, loss_threshold, max_epochs)
    
    net = init_deeponet(100, 1, 256, 256)
    adam_state = init_adam_state(net)
    loss_history = Float64[]

    for epoch in 1:max_epochs
        training_pairs = generate_training_batch(num_functions, queries_per_function, K, sigma, decay_power)
        grads_tuple, avg_loss = compute_batch_gradients(net, training_pairs)
        branch_grads, trunk_grads, bias_grad = grads_tuple

        adam_state.t += 1
        t = adam_state.t

        params = [
            net.branch.W1, net.branch.b1, net.branch.W2, net.branch.b2,
            net.branch.W3, net.branch.b3, net.branch.W4, net.branch.b4,
            net.trunk.W1,  net.trunk.b1,  net.trunk.W2,  net.trunk.b2,
            net.trunk.W3,  net.trunk.b3,  net.trunk.W4,  net.trunk.b4
        ]
        
        grads = [
            branch_grads.dW1, branch_grads.db1, branch_grads.dW2, branch_grads.db2,
            branch_grads.dW3, branch_grads.db3, branch_grads.dW4, branch_grads.db4,
            trunk_grads.dW1,  trunk_grads.db1,  trunk_grads.dW2,  trunk_grads.db2,
            trunk_grads.dW3,  trunk_grads.db3,  trunk_grads.dW4,  trunk_grads.db4
        ]

        for i in 1:length(params)
            adam_update!(params[i], grads[i], adam_state.m[i], adam_state.v[i], t, learning_rate, 0.9, 0.999, 1e-8)
        end

        b_m = adam_state.m[end] = 0.9 * adam_state.m[end] + (1 - 0.9) * bias_grad
        b_v = adam_state.v[end] = 0.999 * adam_state.v[end] + (1 - 0.999)* bias_grad^2
        m_hat = b_m / (1 - 0.9^t)
        v_hat = b_v / (1 - 0.999^t)
        net.bias -= learning_rate * m_hat / (sqrt(v_hat) + 1e-8)

        push!(loss_history, avg_loss)

        if epoch % 50 == 0
            println("Epoch ", epoch, ": loss = ", avg_loss)
        end

        if avg_loss < loss_threshold
            println("Converged at epoch $epoch with loss $avg_loss")
            break
        end
    end

    return net
end
