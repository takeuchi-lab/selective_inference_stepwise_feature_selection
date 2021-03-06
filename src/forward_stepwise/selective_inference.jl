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

const ฮต = 1e-4

function parametric_representation(๐ผ, ๐ฒ, ๐บ)
    z = ๐ผ' * ๐ฒ
    ฯยฒ_z = ๐ผ' * ๐บ * ๐ผ
    ๐ = ๐บ * ๐ผ / ฯยฒ_z
    ๐ = ๐ฒ - ๐ * ๐ผ' * ๐ฒ
    @assert ๐ + ๐ * z โ ๐ฒ
    return (z, ฯยฒ_z, ๐, ๐)
end

function polytope_t(๐, ๐, ๐, ๐ด, ๐ฌ, t)
    L, U = -Inf, +Inf
    for i = 1:t
        ๐_๐ดแตข = ๐[:, ๐ด[1:(i - 1)]]
        ๐ = I - ๐_๐ดแตข / (๐_๐ดแตข' * ๐_๐ดแตข) * ๐_๐ดแตข'
        for jโฒ in collect(1:size(๐, 2))[Not(๐ด[1:i])]
            ๐ฑ_jโฒ = ๐[:, jโฒ] # remain feature
            ๐ฑ_jแตข = ๐[:, ๐ด[i]] # i-th selected feature
            aโ, bโ = (+๐ฑ_jโฒ - ๐ฌ[i] * ๐ฑ_jแตข)' * ๐ * [๐ ๐] # selection event: aโ + bโ*z โค 0
            aโ, bโ = (-๐ฑ_jโฒ - ๐ฌ[i] * ๐ฑ_jแตข)' * ๐ * [๐ ๐] # selection event: aโ + bโ*z โค 0
            if bโ < 0
                L = max(L, -aโ / bโ)
            elseif bโ > 0
                U = min(U, -aโ / bโ)
            end
            if bโ < 0
                L = max(L, -aโ / bโ)
            elseif bโ > 0
                U = min(U, -aโ / bโ)
            end
        end
    end
    return L..U
end

function polytope(๐, ๐ฒ_z, ๐, ๐, k; until=k)
    ๐ด_z, ๐ฌ_z = sfs(๐, ๐ฒ_z, k)
    until != k && (๐ด_z = ๐ด_z[1:until]; ๐ฌ_z = ๐ฌ_z[1:until])
    I_z = polytope_t(๐, ๐, ๐, ๐ด_z, ๐ฌ_z, until)
    return (I_z, ๐ด_z, ๐ฌ_z)
end

function compute_solution_path(๐, ๐, ๐, k, z_min, z_max; until=k)
    z = z_min
    ๐ณ = float([z_min]) # transition point list
    ๐จ = Vector{Vector{Int}}(undef, 0) # active set list
    ๐ = Vector{Vector{Float64}}(undef, 0) # sign vector list
    while z < z_max
        ๐ฒ_z = ๐ + ๐ * z
        I_z, ๐ด_z, ๐ฌ_z = polytope(๐, ๐ฒ_z, ๐, ๐, k; until=until)
        z = I_z.right + ฮต
        z < z_max ? push!(๐ณ, z - ฮต) : push!(๐ณ, z_max)
        push!(๐จ, ๐ด_z)
        push!(๐, ๐ฌ_z)
    end
    return (๐ณ, ๐จ, ๐)
end

function parametric_SFS_SI(๐, ๐ฒ_obs, k, ๐บ; selection_event=Active())
    ๐ด_obs, ๐ฌ_obs = sfs(๐, ๐ฒ_obs, k)
    ๐_obs = ๐[:, ๐ด_obs]
    ๐ผ = ๐_obs / (๐_obs' * ๐_obs)
    p_values = Vector{Float64}(undef, length(๐ด_obs)) # selective p-value list
    for j = 1:length(๐ด_obs)
        # parametric representation for ๐ฒ(z)
        z_obs, ฯยฒ_z, ๐, ๐ = parametric_representation(๐ผ[:, j], ๐ฒ_obs, ๐บ)

        # compute Z = {z | selection_event}
        Z = truncated_interval(selection_event, ๐ด_obs, ๐ฌ_obs, ๐, ๐, ๐, k, -abs(z_obs) - 10 * โ(ฯยฒ_z), abs(z_obs) + 10 * โ(ฯยฒ_z)) 

        ๐_j = cdf(TruncatedDistribution(Normal(0, โ(ฯยฒ_z)), Z), z_obs)
        p_values[j] = 2 * min(๐_j, 1 - ๐_j)
    end
    return (๐ด_obs, p_values)
end

function parametric_SFS_CI(๐, ๐ฒ_obs, k, ๐บ; selection_event=Active(), ฮฑ=0.05)
    ๐ด_obs, ๐ฌ_obs = sfs(๐, ๐ฒ_obs, k)
    ๐_obs = ๐[:, ๐ด_obs]
    ๐ผ = ๐_obs / (๐_obs' * ๐_obs)
    CIs = Vector{ClosedInterval{Float64}}(undef, length(๐ด_obs)) # selective CI list
    for j = 1:length(๐ด_obs)
        # parametric representation for ๐ฒ(z)
        z_obs, ฯยฒ_z, ๐, ๐ = parametric_representation(๐ผ[:, j], ๐ฒ_obs, ๐บ)

        # compute Z = {z | selection_event}
        Z = truncated_interval(selection_event, ๐ด_obs, ๐ฌ_obs, ๐, ๐, ๐, k, -abs(z_obs) - 10 * โ(ฯยฒ_z), abs(z_obs) + 10 * โ(ฯยฒ_z))

        CIs[j] = confidence_interval(z_obs, โ(ฯยฒ_z), Z; ฮฑ=ฮฑ)
    end
    return (๐ด_obs, CIs)
end

abstract type SelectionEvent end
struct Active <: SelectionEvent end
struct ActiveSign <: SelectionEvent end
struct ActiveOrder <: SelectionEvent end
struct ActiveSignOrder <: SelectionEvent end

function truncated_interval(::Active, ๐ด_obs, ๐ฌ_obs, params...; kwargs...)
    ๐ณ, ๐จ, _ = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(๐ณ)}}(undef, 0) # truncated intevals
    for i = 1:length(๐จ)
        Set(๐ด_obs) == Set(๐จ[i]) && push!(Z, (๐ณ[i]..(๐ณ[i + 1]))) # selective event: ๐ด(z) = ๐ด_obs
    end
    return Z
end
function truncated_interval(::ActiveSign, ๐ด_obs, ๐ฌ_obs, params...; kwargs...)
    ๐ณ, ๐จ, ๐ = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(๐ณ)}}(undef, 0) # truncated intevals
    for i = 1:length(๐จ)
        Set(๐ด_obs .* ๐ฌ_obs) == Set(๐จ[i] .* ๐[i]) && push!(Z, (๐ณ[i]..(๐ณ[i + 1]))) # selective event: ๐ด(z) = ๐ด_obs and ๐ฌ(z) = ๐ฌ_obs
    end
    return Z
end
function truncated_interval(::ActiveOrder, ๐ด_obs, ๐ฌ_obs, params...; kwargs...)
    ๐ณ, ๐จ, _ = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(๐ณ)}}(undef, 0) # truncated intevals
    for i = 1:length(๐จ)
        ๐ด_obs == ๐จ[i] && push!(Z, (๐ณ[i]..(๐ณ[i + 1]))) # selective event: ๐ด(z) = ๐ด_obs and selection order
    end
    return Z
end
function truncated_interval(::ActiveSignOrder, ๐ด_obs, ๐ฌ_obs, params...; kwargs...)
    ๐ณ, ๐จ, ๐ = compute_solution_path(params...; kwargs...)
    Z = Array{ClosedInterval{eltype(๐ณ)}}(undef, 0) # truncated intevals
    for i = 1:length(๐จ)
        ๐ด_obs == ๐จ[i] && ๐ฌ_obs == ๐[i] && push!(Z, (๐ณ[i]..(๐ณ[i + 1]))) # selective event: ๐ด(z) = ๐ด_obs and ๐ฌ(z) = ๐ฌ_obs and selection order
    end
    return Z
end

end # module SIforSFS