mutable struct BranchNet{T<:Real}
    W1::Matrix{T}
    b1::Vector{T}
    W2::Matrix{T}
    b2::Vector{T}
    W3::Matrix{T}
    b3::Vector{T}
    W4::Matrix{T}
    b4::Vector{T}
end
 
function he_init(fan_out, fan_in)
    std = sqrt(2 / fan_in)
    return std .* randn(fan_out, fan_in)
end

function init_branch_net(sensor_dim, hidden_dim, p)
    W1 = he_init(hidden_dim, sensor_dim)
    b1 = zeros(hidden_dim)
    
    W2 = he_init(hidden_dim, hidden_dim)
    b2 = zeros(hidden_dim)
    
    W3 = he_init(hidden_dim, hidden_dim)
    b3 = zeros(hidden_dim)
    
    W4 = he_init(p, hidden_dim)
    b4 = zeros(p)
    
    return BranchNet(W1, b1, W2, b2, W3, b3, W4, b4)
end

function relu(x)
    return max.(0, x)
end

function forward_branch(net::BranchNet, sensor_values)
    h1 = relu.(net.W1 * sensor_values .+ net.b1)
    h2 = relu.(net.W2 * h1 .+ net.b2)
    h3 = relu.(net.W3 * h2 .+ net.b3)
    output = net.W4 * h3 .+ net.b4

    return output
end
