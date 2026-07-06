include("branch_net.jl")
include("trunk_net.jl")

mutable struct DeepONet{T<:Real}
    branch::BranchNet{T}
    trunk::TrunkNet{T}
    bias::T
end

function init_deeponet(sensor_dim, query_dim, hidden_dim, p)
    branch = init_branch_net(sensor_dim, hidden_dim, p)
    trunk = init_trunk_net(query_dim, hidden_dim, p)
    bias = 0.0

    return DeepONet(branch, trunk, bias)
end
 
function forward_deeponet(net::DeepONet, sensor_values, y)
    b = forward_branch(net.branch, sensor_values)
    t = forward_trunk(net.trunk, y)
    prediction = sum(b .* t) + net.bias

    return prediction
end
