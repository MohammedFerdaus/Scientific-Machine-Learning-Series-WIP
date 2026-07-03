# Benchmark: Neural ODE for Lotka-Volterra using DiffEqFlux/Lux/Optimization.jl
# Adapted from Chris Rackauckas' DiffEqFlux Neural ODE tutorial, updated to
# the current (post-FastChain/sciml_train deprecation) API: Lux for the
# network, NeuralODE as the differential equation layer, and Optimization.jl
# with OptimizationOptimisers (ADAM) then OptimizationOptimJL (BFGS).
#
# This file is purely an external reference point using established
# libraries. It is NOT part of the from-scratch SciML Foundations
# implementation -- it exists to sanity-check our hand-built adjoint
# Neural ODE against a known-working library implementation.

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random, Plots

rng = Xoshiro(0)

function lotka_volterra(du, u, p, t)
    x, y = u
    alpha, beta, delta, gamma = p
    du[1] = alpha*x - beta*x*y
    du[2] = delta*x*y - gamma*y
end

u0 = Float32[1.0, 1.0]
p_true = Float32[1.5, 1.0, 1.0, 3.0]

tspan_full = Float32.((0.0, 6.5))
datasize_full = 150
tsteps_full = range(tspan_full[1], tspan_full[2]; length=datasize_full)

prob_trueode = ODEProblem(lotka_volterra, u0, tspan_full, p_true)
ode_data_full = Array(solve(prob_trueode, Tsit5(); saveat=tsteps_full))

dudt2 = Lux.Chain(Lux.Dense(2, 16, tanh), Lux.Dense(16, 16, tanh), Lux.Dense(16, 2))
p_init, st = Lux.setup(rng, dudt2)
p_init = ComponentArray(p_init)

function run_stage(current_params, tshort, adam_iters, bfgs_iters, stage_label)
    mask = tsteps_full .<= tshort
    tsteps_stage = tsteps_full[mask]
    data_stage = ode_data_full[:, mask]

    prob_neuralode = NeuralODE(dudt2, (0.0f0, Float32(tshort)), Tsit5(); saveat=tsteps_stage)

    function predict(p)
        Array(prob_neuralode(u0, p, st)[1])
    end

    function loss(p, _)
        pred = predict(p)
        return sum(abs2, data_stage .- pred)
    end

    iter = 0
    callback = function (state, l)
        iter += 1
        if iter % 50 == 0
            println("[", stage_label, "] ADAM Iter: ", iter, " | Loss: ", l)
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction(loss, adtype)
    optprob = OptimizationProblem(optf, current_params)

    result_adam = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01);
                                       callback=callback, maxiters=adam_iters)

    iter = 0
    callback2 = function (state, l)
        iter += 1
        if iter % 50 == 0
            println("[", stage_label, "] BFGS Iter: ", iter, " | Loss: ", l)
        end
        return false
    end

    optprob2 = remake(optprob; u0=result_adam.u)
    result_bfgs = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm=0.01f0);
                                   callback=callback2, maxiters=bfgs_iters, allow_f_increases=true)

    println("[", stage_label, "] Final loss: ", loss(result_bfgs.u, nothing))

    return result_bfgs.u
end

adam_epochs = 2500
bfgs_epochs = 2500

println("===== RUN A: Three-stage (1 cycle -> 10s -> 15s) =====")
paramsA = p_init
paramsA = run_stage(paramsA, 3.2, adam_epochs, bfgs_epochs, "A-Stage1(3.2s)")
paramsA = run_stage(paramsA, 10.0, adam_epochs, bfgs_epochs, "A-Stage2(10s)")
paramsA = run_stage(paramsA, 15.0, adam_epochs, bfgs_epochs, "A-Stage3(15s)")

function predict_full(params)
    prob = NeuralODE(dudt2, tspan_full, Tsit5(); saveat=tsteps_full)
    Array(prob(u0, params, st)[1])
end

predA = predict_full(paramsA)

pl_phase = plot(ode_data_full[1,:], ode_data_full[2,:], label="True", linewidth=2,
                  xlabel="Prey (x)", ylabel="Predator (y)", title="Curriculum Comparison")
plot!(pl_phase, predA[1,:], predA[2,:], label="Run A (3-stage)", linestyle=:dash)
savefig(pl_phase, "curriculum_comparison_phase.png")

loss_A_final = sum(abs2, ode_data_full .- predA)
println("Run A full-horizon loss: ", loss_A_final)