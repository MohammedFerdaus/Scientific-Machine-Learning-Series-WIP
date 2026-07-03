using Random

function lotka_volterra_dynamics(state, alpha, beta, delta, gamma)
    x, y = state
    dx = alpha*x - beta*x*y
    dy = delta*x*y - gamma*y

    return [dx, dy]
end

function log_dynamics(log_state, alpha, beta, delta, gamma)
    u, v = log_state
    du = alpha - beta*exp(v)
    dv = delta*exp(u) - gamma
    
    return [du, dv]
end

function generate_trajectory(initial_state, tspan, dt, alpha, beta, delta, gamma)
    times = collect(tspan[1] : dt : tspan[2])
    dynamics_fn = (state, t) -> begin
        lotka_volterra_dynamics(state, alpha, beta, delta, gamma)
    end
    trajectory = integrate_forward(initial_state, times, dynamics_fn)

    return trajectory, times
end

function add_gaussian_noise(trajectory, noise_std, random_seed)
    rng = Random.MersenneTwister(random_seed)
    noise = randn(rng, size(trajectory)) * noise_std

    return trajectory + noise
end