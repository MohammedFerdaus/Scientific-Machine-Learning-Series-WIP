function rk4_step(state, t, dt, dynamics_fn)
    k1 = dynamics_fn(state, t)
    k2 = dynamics_fn(state + 0.5*dt*k1, t + 0.5*dt)
    k3 = dynamics_fn(state + 0.5*dt*k2, t + 0.5*dt)
    k4 = dynamics_fn(state + dt*k3,     t + dt)

    return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
end

function rk4_step_log(log_state, t, dt, alpha, beta, delta, gamma)
    k1 = log_dynamics(log_state, alpha, beta, delta, gamma)
    k2 = log_dynamics(log_state + 0.5*dt*k1, alpha, beta, delta, gamma)
    k3 = log_dynamics(log_state + 0.5*dt*k2, alpha, beta, delta, gamma)
    k4 = log_dynamics(log_state + dt*k3, alpha, beta, delta, gamma)

    return log_state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
end

function integrate_forward(initial_state, times, dynamics_fn)
    trajectory = [initial_state]
    for i in 1:length(times) - 1
        dt = times[i+1] - times[i]
        next_state = rk4_step(trajectory[i], times[i], dt, dynamics_fn)

        push!(trajectory, next_state)
    end

    return reduce(hcat, trajectory)'
end

function integrate_forward_log(initial_state, times, alpha, beta, delta, gamma)
    log_initial = log.(initial_state)
    trajectory = [log_initial]
    for i in 1:length(times) - 1
        dt = times[i+1] - times[i]
        next_log_state = rk4_step_log(trajectory[i], times[i], dt, alpha, beta, delta, gamma )
        
        push!(trajectory, next_log_state)
    end
    
    log_trajectory = reduce(hcat, trajectory)'
    return exp.(log_trajectory)
end
