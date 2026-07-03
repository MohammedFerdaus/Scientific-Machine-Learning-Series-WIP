function compute_loss(params, observed_trajectory, times)
    dynamics_fn = (state, t) -> vector_field(state, params, :data_driven)
    predicted = integrate_forward(observed_trajectory[1, :], times, dynamics_fn)

    N = length(times)
    loss = 0.0
    for i in 1:N
        diff = predicted[i, :] - observed_trajectory[i, :]
        loss += sum(diff .^ 2)
    end

    loss = loss / N
    
    return loss, predicted
end

function adam_step(params, grad_params, adam_state, learning_rate, step)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    new_params = []
    new_adam_state = []

    for l in 1:length(params)
        W, b = params[l]
        dW, db = grad_params[l]
        mW, vW, mb, vb = adam_state[l] 

        mW = beta1*mW + (1-beta1)*dW
        vW = beta2*vW + (1-beta2)*(dW.^2)
        mb = beta1*mb + (1-beta1)*db
        vb = beta2*vb + (1-beta2)*(db.^2)

        mW_hat = mW / (1 - beta1^step)
        vW_hat = vW / (1 - beta2^step)
        mb_hat = mb / (1 - beta1^step)
        vb_hat = vb / (1 - beta2^step)

        W_new = W - learning_rate * mW_hat ./ (sqrt.(vW_hat) .+ epsilon)
        b_new = b - learning_rate * mb_hat ./ (sqrt.(vb_hat) .+ epsilon)

        push!(new_params, (W_new, b_new))
        push!(new_adam_state, (mW, vW, mb, vb))
    end
    
    return new_params, new_adam_state
end

function train(observed_trajectory, times, num_epochs, learning_rate, layer_sizes, random_seed)
    params = init_params(layer_sizes, random_seed)

    adam_state = []
    for (W, b) in params
        push!(adam_state, (zeros(size(W)), zeros(size(W)), zeros(size(b)), zeros(size(b))))
    end
    
    loss_history = []
    for epoch in 1:num_epochs  # Fixed: 1:num_epochs instead of 1:length(num_epochs)
        loss, predicted = compute_loss(params, observed_trajectory, times)
        push!(loss_history, loss)

        grad_params = integrate_adjoint(predicted, observed_trajectory, times, params)
        params, adam_state = adam_step(params, grad_params, adam_state, learning_rate, epoch)

        if epoch % 100 == 0
            println("Epoch: ", epoch, " | Loss: ", loss)
        end
    end

    return params, loss_history
end