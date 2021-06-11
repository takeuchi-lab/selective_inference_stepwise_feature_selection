module SIforSFS

export compute_solution_path, truncated_interval, parametric_SFS_SI, parametric_SFS_CI, SelectionEvent, Active, ActiveSign, ActiveOrder, ActiveSignOrder

using InvertedIndices
using LinearAlgebra
using IntervalSets
using Distributions
include("sfs.jl")
using .SFS
include("../selective_p.jl")
using .MultiTruncatedDistributions
include("../ci.jl")
using .ConfidenceInterval

const ε = 1e-4

function parametric_representation(𝜼, 𝐲, 𝚺)
    z = 𝜼' * 𝐲
    σ²_z = 𝜼' * 𝚺 * 𝜼
    𝐛 = 𝚺 * 𝜼 / σ²_z
    𝐚 = 𝐲 - 𝐛 * 𝜼' * 𝐲
    @assert 𝐚 + 𝐛 * z ≈ 𝐲
    return (z, σ²_z, 𝐚, 𝐛)
end

function polytope_t(𝐗, 𝐚, 𝐛, 𝐴, 𝐬, t)
    L, U = -Inf, +Inf
    for i = 1:t
        𝐗_𝐴ᵢ = 𝐗[:, 𝐴[1:(i - 1)]]
        𝐏 = I - 𝐗_𝐴ᵢ / (𝐗_𝐴ᵢ' * 𝐗_𝐴ᵢ) * 𝐗_𝐴ᵢ'
        for j′ in collect(1:size(𝐗, 2))[Not(𝐴[1:i])]
            𝐱_j′ = 𝐗[:, j′] # remain feature
            𝐱_jᵢ = 𝐗[:, 𝐴[i]] # i-th selected feature
            a₊, b₊ = (+𝐱_j′ - 𝐬[i] * 𝐱_jᵢ)' * 𝐏 * [𝐚 𝐛] # selection event: a₊ + b₊*z ≤ 0
            a₋, b₋ = (-𝐱_j′ - 𝐬[i] * 𝐱_jᵢ)' * 𝐏 * [𝐚 𝐛] # selection event: a₋ + b₋*z ≤ 0
            if b₊ < 0
                L = max(L, -a₊ / b₊)
            elseif b₊ > 0
                U = min(U, -a₊ / b₊)
            end
            if b₋ < 0
                L = max(L, -a₋ / b₋)
            elseif b₋ > 0
                U = min(U, -a₋ / b₋)
            end
        end
    end
    return L..U
end

function polytope(𝐗, 𝐲_z, 𝐚, 𝐛, k; until=k)
    𝐴_z, 𝐬_z = sfs(𝐗, 𝐲_z, k)
    until != k && (𝐴_z = 𝐴_z[1:until]; 𝐬_z = 𝐬_z[1:until])
    I_z = polytope_t(𝐗, 𝐚, 𝐛, 𝐴_z, 𝐬_z, until)
    return (I_z, 𝐴_z, 𝐬_z)
end

function compute_solution_path(𝐗, 𝐚, 𝐛, k, z_min, z_max; until=k)
    z = z_min
    𝐳 = float([z_min]) # transition point list
    𝑨 = Vector{Vector{Int}}(undef, 0) # active set list
    𝐒 = Vector{Vector{Float64}}(undef, 0) # sign vector list
    while z < z_max
        𝐲_z = 𝐚 + 𝐛 * z
        I_z, 𝐴_z, 𝐬_z = polytope(𝐗, 𝐲_z, 𝐚, 𝐛, k; until=until)
        z = I_z.right + ε
        z < z_max ? push!(𝐳, z - ε) : push!(𝐳, z_max)
        push!(𝑨, 𝐴_z)
        push!(𝐒, 𝐬_z)
    end
    return (𝐳, 𝑨, 𝐒)
end

function parametric_SFS_SI(𝐗, 𝐲_obs, k, 𝚺; selection_event=Active())
    𝐴_obs, 𝐬_obs = sfs(𝐗, 𝐲_obs, k)
    𝐗_obs = 𝐗[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    p_values = Vector{Float64}(undef, length(𝐴_obs)) # selective p-value list
    for j = 1:length(𝐴_obs)
        # parametric representation for 𝐲(z)
        z_obs, σ²_z, 𝐚, 𝐛 = parametric_representation(𝜼[:, j], 𝐲_obs, 𝚺)

        # compute Z = {z | selection_event}
        Z = truncated_interval(selection_event, 𝐴_obs, 𝐬_obs, 𝐗, 𝐚, 𝐛, k, -abs(z_obs) - 10 * √(σ²_z), abs(z_obs) + 10 * √(σ²_z)) 

        𝜋_j = cdf(TruncatedDistribution(Normal(0, √(σ²_z)), Z), z_obs)
        p_values[j] = 2 * min(𝜋_j, 1 - 𝜋_j)
    end
    return (𝐴_obs, p_values)
end

function parametric_SFS_CI(𝐗, 𝐲_obs, k, 𝚺; selection_event=Active(), α=0.05)
    𝐴_obs, 𝐬_obs = sfs(𝐗, 𝐲_obs, k)
    𝐗_obs = 𝐗[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    CIs = Vector{ClosedInterval{Float64}}(undef, length(𝐴_obs)) # selective CI list
    for j = 1:length(𝐴_obs)
        # parametric representation for 𝐲(z)
        z_obs, σ²_z, 𝐚, 𝐛 = parametric_representation(𝜼[:, j], 𝐲_obs, 𝚺)

        # compute Z = {z | selection_event}
        Z = truncated_interval(selection_event, 𝐴_obs, 𝐬_obs, 𝐗, 𝐚, 𝐛, k, -abs(z_obs) - 10 * √(σ²_z), abs(z_obs) + 10 * √(σ²_z))

        CIs[j] = confidence_interval(z_obs, √(σ²_z), Z; α=α)
    end
    return (𝐴_obs, CIs)
end

abstract type SelectionEvent end
struct Active <: SelectionEvent end
struct ActiveSign <: SelectionEvent end
struct ActiveOrder <: SelectionEvent end
struct ActiveSignOrder <: SelectionEvent end

function truncated_interval(::Active, 𝐴_obs, 𝐬_obs, params...; kwargs...)
    𝐳, 𝑨, _ = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(𝐳)}}(undef, 0) # truncated intevals
    for i = 1:length(𝑨)
        Set(𝐴_obs) == Set(𝑨[i]) && push!(Z, (𝐳[i]..(𝐳[i + 1]))) # selective event: 𝐴(z) = 𝐴_obs
    end
    return Z
end
function truncated_interval(::ActiveSign, 𝐴_obs, 𝐬_obs, params...; kwargs...)
    𝐳, 𝑨, 𝐒 = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(𝐳)}}(undef, 0) # truncated intevals
    for i = 1:length(𝑨)
        Set(𝐴_obs .* 𝐬_obs) == Set(𝑨[i] .* 𝐒[i]) && push!(Z, (𝐳[i]..(𝐳[i + 1]))) # selective event: 𝐴(z) = 𝐴_obs and 𝐬(z) = 𝐬_obs
    end
    return Z
end
function truncated_interval(::ActiveOrder, 𝐴_obs, 𝐬_obs, params...; kwargs...)
    𝐳, 𝑨, _ = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(𝐳)}}(undef, 0) # truncated intevals
    for i = 1:length(𝑨)
        𝐴_obs == 𝑨[i] && push!(Z, (𝐳[i]..(𝐳[i + 1]))) # selective event: 𝐴(z) = 𝐴_obs and selection order
    end
    return Z
end
function truncated_interval(::ActiveSignOrder, 𝐴_obs, 𝐬_obs, params...; kwargs...)
    𝐳, 𝑨, 𝐒 = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(𝐳)}}(undef, 0) # truncated intevals
    for i = 1:length(𝑨)
        𝐴_obs == 𝑨[i] && 𝐬_obs == 𝐒[i] && push!(Z, (𝐳[i]..(𝐳[i + 1]))) # selective event: 𝐴(z) = 𝐴_obs and 𝐬(z) = 𝐬_obs and selection order
    end
    return Z
end

end # module SIforSFS