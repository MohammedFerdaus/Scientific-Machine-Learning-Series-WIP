include("branch_net.jl")

mutable struct TrunkNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
    W3::Matrix{T}
    b3::Vector{T}
    W4::Matrix{T}
    b4::Vector{T}
end

function init_trunk_net(query_dim, hidden_dim, p)
    W1 = he_init(hidden_dim, query_dim)
    b1 = zeros(hidden_dim)

    W2 = he_init(hidden_dim, hidden_dim)
    b2 = zeros(hidden_dim)

    W3 = he_init(hidden_dim, hidden_dim)
    b3 = zeros(hidden_dim)

    W4 = he_init(p, hidden_dim)
    b4 = zeros(p)

    return TrunkNet(W1, b1, W2, b2, W3, b3, W4, b4)
end

function forward_trunk(net::TrunkNet, y)
    y_vec = [y]
    h1 = relu.(net.W1 * y_vec .+ net.b1)
    h2 = relu.(net.W2 * h1 .+ net.b2)
    h3 = relu.(net.W3 * h2 .+ net.b3)
    output = relu.(net.W4 * h3 .+ net.b4)

    return output
end