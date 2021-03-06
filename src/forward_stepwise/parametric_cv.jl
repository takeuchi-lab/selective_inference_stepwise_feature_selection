module ParametricCVforSFS

export parametric_SFS_SI_TVS, parametric_SFS_SI_CV, parametric_SFS_CI_TVS, parametric_SFS_CI_CV

include("../data.jl")
using .DataUtil
include("sfs.jl")
using .SFS
include("selective_inference.jl")
using .SIforSFS
include("../quadratic.jl")
using .Quadratic
include("../intervals.jl")
using .Intersection
include("../selective_p.jl")
using .MultiTruncatedDistributions
include("../ci.jl")
using .ConfidenceInterval
using IntervalSets
using Distributions

const ฮต = 1e-4

function validation_error(๐_train, ๐_val, ๐_train, ๐_val, ๐_train, ๐_val)
    ๐โ = (๐_train' * ๐_train) \ ๐_train' # pseudo inverse matrix of ๐_train
    ๐ = ๐_val .- ๐_val * ๐โ * ๐_train
    ๐ = ๐_val .- ๐_val * ๐โ * ๐_train
    return quadratic(๐' * ๐, 2 * ๐' * ๐, ๐' * ๐) # E(z) = ๐แต๐ + 2๐แต๐z + ๐แต๐zยฒ
end

function compute_val_error_path(k, ๐_train, ๐_val, ๐_train, ๐_val, ๐_train, ๐_val, z_min, z_max)
    ๐ณ_k, ๐จ_train_k, _ = compute_solution_path(๐_train, ๐_train, ๐_train, k, z_min, z_max)
    ๐_k = (๐ด -> validation_error((@view ๐_train[:, ๐ด]), (@view ๐_val[:, ๐ด]), ๐_train, ๐_val, ๐_train, ๐_val)).(๐จ_train_k)
    return ๐ณ_k, ๐_k
end

function compute_cv_path(k, ๐, ๐, ๐, ๐, z_min, z_max)
    ๐_fold = KFold(length(๐), ๐) # ๐-fold Cross Validation
    paths = (i -> begin
        ๐_train, ๐_val = train_val_split(๐, ๐_fold.parts[i], ๐_fold.parts[i + 1] - 1)
        ๐_train, ๐_val = train_val_split(๐, ๐_fold.parts[i], ๐_fold.parts[i + 1] - 1)
        ๐_train, ๐_val = train_val_split(๐, ๐_fold.parts[i], ๐_fold.parts[i + 1] - 1)
        compute_val_error_path(k, ๐_train, ๐_val, ๐_train, ๐_val, ๐_train, ๐_val, z_min, z_max)
    end).(1:๐)
    ๐ณs = [paths[i][1] for i = 1:๐]
    ๐s = [paths[i][2] for i = 1:๐]
    # merge
    ๐ณ_k = [z_min]
    ๐_k = (eltype(๐s[begin]))[]
    pointers = ones(Int, ๐)
    z_left, z_right = z_min, z_min
    while z_right < z_max
        i_next = argmin((i -> ๐ณs[i][pointers[i] + 1]).(1:๐))
        z_right = ๐ณs[i_next][pointers[i_next] + 1]

        push!(๐ณ_k, z_right)
        push!(๐_k, mean(๐s[i][pointers[i]] for i = 1:๐))

        z_left = z_right
        pointers[i_next] += 1
    end
    @assert z_right == z_max
    return ๐ณ_k, ๐_k
end

function compute_Z_CV(k_obs, K, ๐ณ, ๐)
    z_min, z_max = ๐ณ[k_obs][begin], ๐ณ[k_obs][end]
    # ๅkใงใฎ๐ณ_kใงใใผใธใฝใผใใใฆใใใชใใๅๅบ้ใงE_kobsใจE_kโฒใๆฏ่ผใใฆE_kobs = min E_kใจใชใๅบ้ใๆฑใใ
    Z_CV = ClosedInterval{eltype(๐ณ[k_obs])}[] # init Z_CV
    pointers = Dict(k => 1 for k โ K)
    z_left, z_right = z_min, z_min
    while z_right < z_max
        # this algorithm is just like merge sort for ๐ณ_k (k โ K)
        k_next = K[argmin((k -> ๐ณ[k][pointers[k] + 1]).(K))]
        z_right = ๐ณ[k_next][pointers[k_next] + 1]
        z_left == z_right && (pointers[k_next] += 1; continue)
        I = z_left..z_right # current interval

        Z_I = ClosedInterval{eltype(๐ณ[k_obs])}[] # truncated interval on current I
        for kโฒ โ K
            k_obs == kโฒ && continue
            Z_I = I โฉ (๐[k_obs][pointers[k_obs]] โค ๐[kโฒ][pointers[kโฒ]])
            isempty(Z_I) && break
        end # Z_I = {z โ I | E_{k_obs}(z) = min_k E_k(z)}
        Z_CV = vcat(Z_CV, Z_I)

        z_left = z_right
        pointers[k_next] += 1
    end
    @assert z_right == z_max
    return Z_CV
end

function parametric_SFS_SI_TVS(๐, ๐ฒ_obs, ๐บ, K; ratio=0.5, selection_event=Active())
    ๐ด_obs, ๐ฌ_obs, k_obs = sfs_TVS(๐, ๐ฒ_obs, K; ratio=ratio)
    ๐_obs = ๐[:, ๐ด_obs]
    ๐ผ = ๐_obs / (๐_obs' * ๐_obs)
    p_values = Vector{Float64}(undef, length(๐ด_obs)) # selective p-value list
    for j = 1:length(๐ด_obs)
        # parametric representation for ๐ฒ(z)
        z_obs, ฯยฒ_z, ๐, ๐ = SIforSFS.parametric_representation(๐ผ[:, j], ๐ฒ_obs, ๐บ)

        # compute Z_CV = {z | k(z) = k_obs}
        ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
        ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
        ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
        paths = (k -> compute_val_error_path(k, ๐_train, ๐_val, ๐_train, ๐_val, ๐_train, ๐_val, -abs(z_obs) - 3 * โ(ฯยฒ_z), abs(z_obs) + 3 * โ(ฯยฒ_z))).(K)
        ๐ณ = Dict(k => paths[i][1] for (i, k) โ enumerate(K))
        ๐ = Dict(k => paths[i][2] for (i, k) โ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, ๐ณ, ๐)

        # compute Z_alg = {z | ๐ด(z) = ๐ด_obs}
        Z_alg = truncated_interval(selection_event, ๐ด_obs, ๐ฌ_obs, ๐, ๐, ๐, k_obs, -abs(z_obs) - 3 * โ(ฯยฒ_z), abs(z_obs) + 3 * โ(ฯยฒ_z))

        Z = Z_alg โฉ Z_CV
        ๐_j = cdf(TruncatedDistribution(Normal(0, โ(ฯยฒ_z)), Z), z_obs)
        p_values[j] = 2 * min(๐_j, 1 - ๐_j)
    end
    return (๐ด_obs, p_values)
end

function parametric_SFS_SI_CV(๐, ๐ฒ_obs, ๐บ, K; ๐=length(๐ฒ_obs), selection_event=Active())
    ๐ด_obs, ๐ฌ_obs, k_obs = sfs_CV(๐, ๐ฒ_obs, K; ๐=๐)
    ๐_obs = ๐[:, ๐ด_obs]
    ๐ผ = ๐_obs / (๐_obs' * ๐_obs)
    p_values = Vector{Float64}(undef, length(๐ด_obs)) # selective p-value list
    for j = 1:length(๐ด_obs)
        # parametric representation for ๐ฒ(z)
        z_obs, ฯยฒ_z, ๐, ๐ = SIforSFS.parametric_representation(๐ผ[:, j], ๐ฒ_obs, ๐บ)

        # compute Z_CV = {z | k(z) = k_obs}
        paths = (k -> compute_cv_path(k, ๐, ๐, ๐, ๐, -abs(z_obs) - 3 * โ(ฯยฒ_z), abs(z_obs) + 3 * โ(ฯยฒ_z))).(K)
        ๐ณ = Dict(k => paths[i][1] for (i, k) โ enumerate(K))
        ๐ = Dict(k => paths[i][2] for (i, k) โ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, ๐ณ, ๐)

        # compute Z_alg = {z | ๐ด(z) = ๐ด_obs}
        Z_alg = truncated_interval(selection_event, ๐ด_obs, ๐ฌ_obs, ๐, ๐, ๐, k_obs, -abs(z_obs) - 3 * โ(ฯยฒ_z), abs(z_obs) + 3 * โ(ฯยฒ_z))

        Z = Z_alg โฉ Z_CV
        ๐_j = cdf(TruncatedDistribution(Normal(0, โ(ฯยฒ_z)), Z), z_obs)
        p_values[j] = 2 * min(๐_j, 1 - ๐_j)
    end
    return (๐ด_obs, p_values)
end

function parametric_SFS_CI_TVS(๐, ๐ฒ_obs, ๐บ, K; ratio=0.5, selection_event=Active(), ฮฑ=0.05)
    ๐ด_obs, ๐ฌ_obs, k_obs = sfs_TVS(๐, ๐ฒ_obs, K; ratio=ratio)
    ๐_obs = ๐[:, ๐ด_obs]
    ๐ผ = ๐_obs / (๐_obs' * ๐_obs)
    CIs = Vector{ClosedInterval{Float64}}(undef, length(๐ด_obs)) # selective CI list
    for j = 1:length(๐ด_obs)
        # parametric representation for ๐ฒ(z)
        z_obs, ฯยฒ_z, ๐, ๐ = SIforSFS.parametric_representation(๐ผ[:, j], ๐ฒ_obs, ๐บ)

        # compute Z_CV = {z | k(z) = k_obs}
        ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
        ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
        ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
        paths = (k -> compute_val_error_path(k, ๐_train, ๐_val, ๐_train, ๐_val, ๐_train, ๐_val, -20 * โ(ฯยฒ_z), 20 * โ(ฯยฒ_z))).(K)
        ๐ณ = Dict(k => paths[i][1] for (i, k) โ enumerate(K))
        ๐ = Dict(k => paths[i][2] for (i, k) โ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, ๐ณ, ๐)

        # compute Z_alg = {z | ๐ด(z) = ๐ด_obs}
        Z_alg = truncated_interval(selection_event, ๐ด_obs, ๐ฌ_obs, ๐, ๐, ๐, k_obs, -20 * โ(ฯยฒ_z), 20 * โ(ฯยฒ_z))

        Z = Z_alg โฉ Z_CV

        CIs[j] = confidence_interval(z_obs, โ(ฯยฒ_z), Z; ฮฑ=ฮฑ)
    end
    return (๐ด_obs, CIs)
end

function parametric_SFS_CI_CV(๐, ๐ฒ_obs, ๐บ, K; ๐=length(๐ฒ_obs), selection_event=Active(), ฮฑ=0.05)
    ๐ด_obs, ๐ฌ_obs, k_obs = sfs_CV(๐, ๐ฒ_obs, K; ๐=๐)
    ๐_obs = ๐[:, ๐ด_obs]
    ๐ผ = ๐_obs / (๐_obs' * ๐_obs)
    CIs = Vector{ClosedInterval{Float64}}(undef, length(๐ด_obs)) # selective CI list
    for j = 1:length(๐ด_obs)
        # parametric representation for ๐ฒ(z)
        z_obs, ฯยฒ_z, ๐, ๐ = SIforSFS.parametric_representation(๐ผ[:, j], ๐ฒ_obs, ๐บ)

        # compute Z_CV = {z | k(z) = k_obs}
        paths = (k -> compute_cv_path(k, ๐, ๐, ๐, ๐, -20 * โ(ฯยฒ_z), 20 * โ(ฯยฒ_z))).(K)
        ๐ณ = Dict(k => paths[i][1] for (i, k) โ enumerate(K))
        ๐ = Dict(k => paths[i][2] for (i, k) โ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, ๐ณ, ๐)

        # compute Z_alg = {z | ๐ด(z) = ๐ด_obs}
        Z_alg = truncated_interval(selection_event, ๐ด_obs, ๐ฌ_obs, ๐, ๐, ๐, k_obs, -20 * โ(ฯยฒ_z), 20 * โ(ฯยฒ_z))

        Z = Z_alg โฉ Z_CV
        CIs[j] = confidence_interval(z_obs, โ(ฯยฒ_z), Z; ฮฑ=ฮฑ)
    end
    return (๐ด_obs, CIs)
end

end # module ParametricCVforSFS


# # test
# N = 100
# ฮฒโ = 0.0
# ฮฒ = rand(10)
# K = (1, 3, 5, 7, 9)
# ratio = 0.9
# ๐ = 5
# using LinearAlgebra
# ๐บ = I

# ๐, ๐ฒ_obs = make_dataset(ฮฒโ, ฮฒ, N)
# ๐, ๐ฒ_obs = centering(๐, ๐ฒ_obs)
# ๐ด_obs, _, k_obs = sfs_TVS(๐, ๐ฒ_obs, K; ratio=ratio)
# ๐ด_obs, _, k_obs = sfs_CV(๐, ๐ฒ_obs, K; ๐=๐)

# ๐_obs = ๐[:, ๐ด_obs]
# ๐ผ = ๐_obs / (๐_obs' * ๐_obs)
# ๐ผโฑผ = ๐ผ[:, 1]
# z_obs = ๐ผโฑผ' * ๐ฒ_obs
# ฯยฒ_z = ๐ผโฑผ' * ๐บ * ๐ผโฑผ # variance of z_obs
# ๐ = ๐บ * ๐ผโฑผ / ฯยฒ_z
# ๐ = ๐ฒ_obs - ๐ * ๐ผโฑผ' * ๐ฒ_obs
# @assert ๐ + ๐ * z_obs โ ๐ฒ_obs
# # ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
# # ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
# # ๐_train, ๐_val = train_val_split(๐; ratio=ratio)
# # paths = (k -> compute_val_error_path(k, ๐_train, ๐_val, ๐_train, ๐_val, ๐_train, ๐_val, -20 * โ(ฯยฒ_z), 20 * โ(ฯยฒ_z))).(1:K) # TVS version
# paths = (k -> compute_cv_path(k, ๐, ๐, ๐, ๐, -20 * โ(ฯยฒ_z), 20 * โ(ฯยฒ_z))).(K) # CV version
# ๐ณ = Dict(k => paths[i][1] for (i, k) โ enumerate(K))
# ๐ = Dict(k => paths[i][2] for (i, k) โ enumerate(K))
# Z_CV = compute_Z_CV(k_obs, K, ๐ณ, ๐)
# # create Z_new
# Z_new = eltype(Z_CV)[Z_CV[begin]]
# for I in (@view Z_CV[2:end])
#     if abs(Z_new[end].right - I.left) < 1e-2 * โ(ฯยฒ_z) 
#         Z_new[end] = (Z_new[end].left)..(I.right)
#     else
#         push!(Z_new, I)
#     end
# end
# Z_new
# function val_error(๐ณ_k, ๐_k)
#     T = length(๐_k) # change point size
#     input = Vector{Float64}[]
#     output = Vector{Float64}[]
#     for t in 1:T
#         z_grid = range(๐ณ_k[t], ๐ณ_k[t + 1], length=10000)
#         input = vcat(input, z_grid)
#         output = vcat(output, ๐_k[t].(z_grid))
#     end
#     return input, output
# end
# using Plots
# plot()
# minE, maxE = Ref(Inf), Ref(-Inf)
# for k โ K
#     input, output = val_error(๐ณ[k], ๐[k])
#     minE[] = min(minE[], minimum(output))
#     maxE[] = max(maxE[], maximum(output))
#     plot!(input, output; label="k=$(k)")
# end
# for interval in Z_new # (Z_CV)
#     z_cv = range(interval.left, interval.right; length=1000)
#     plot!(z_cv, zero.(z_cv) .+ (minE[] - 0.01); lw=3, c=:red, label=nothing)
# end
# function point!(x, y; kwargs...) # plot 1 data point
#     scatter!([x], [y]; kwargs...)
# end
# function stem!(x, y_min, y_max; kwargs...) # plot x = x line
#     plot!([x, x], [y_min, y_max]; kwargs...)
# end
# point!(z_obs, minE[] - 0.01; c=:red, label=nothing)
# for interval in Z_new
#     stem!(interval.left, minE[] - 0.01, maxE[] + 0.01; label=nothing, c=:gray, ฮฑ=0.5, ls=:dashdot)
#     stem!(interval.right, minE[] - 0.01, maxE[] + 0.01; label=nothing, c=:gray, ฮฑ=0.5, ls=:dashdot)
# end
# using LaTeXStrings
# p = plot!(; xlabel=L"z", ylabel=L"E_k(z)", title=L"k_{\mathrm{obs}}=%$(k_obs)")
# savefig(p, "cv_demo.pdf")