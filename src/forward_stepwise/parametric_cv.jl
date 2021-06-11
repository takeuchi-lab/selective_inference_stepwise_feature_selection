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

const ε = 1e-4

function validation_error(𝐗_train, 𝐗_val, 𝐚_train, 𝐚_val, 𝐛_train, 𝐛_val)
    𝐗₊ = (𝐗_train' * 𝐗_train) \ 𝐗_train' # pseudo inverse matrix of 𝐗_train
    𝐚 = 𝐚_val .- 𝐗_val * 𝐗₊ * 𝐚_train
    𝐛 = 𝐛_val .- 𝐗_val * 𝐗₊ * 𝐛_train
    return quadratic(𝐛' * 𝐛, 2 * 𝐚' * 𝐛, 𝐚' * 𝐚) # E(z) = 𝐚ᵀ𝐚 + 2𝐚ᵀ𝐛z + 𝐛ᵀ𝐛z²
end

function compute_val_error_path(k, 𝐗_train, 𝐗_val, 𝐚_train, 𝐚_val, 𝐛_train, 𝐛_val, z_min, z_max)
    𝐳_k, 𝑨_train_k, _ = compute_solution_path(𝐗_train, 𝐚_train, 𝐛_train, k, z_min, z_max)
    𝐄_k = (𝐴 -> validation_error((@view 𝐗_train[:, 𝐴]), (@view 𝐗_val[:, 𝐴]), 𝐚_train, 𝐚_val, 𝐛_train, 𝐛_val)).(𝑨_train_k)
    return 𝐳_k, 𝐄_k
end

function compute_cv_path(k, 𝐗, 𝐚, 𝐛, 𝑘, z_min, z_max)
    𝑘_fold = KFold(length(𝐚), 𝑘) # 𝑘-fold Cross Validation
    paths = (i -> begin
        𝐗_train, 𝐗_val = train_val_split(𝐗, 𝑘_fold.parts[i], 𝑘_fold.parts[i + 1] - 1)
        𝐚_train, 𝐚_val = train_val_split(𝐚, 𝑘_fold.parts[i], 𝑘_fold.parts[i + 1] - 1)
        𝐛_train, 𝐛_val = train_val_split(𝐛, 𝑘_fold.parts[i], 𝑘_fold.parts[i + 1] - 1)
        compute_val_error_path(k, 𝐗_train, 𝐗_val, 𝐚_train, 𝐚_val, 𝐛_train, 𝐛_val, z_min, z_max)
    end).(1:𝑘)
    𝐳s = [paths[i][1] for i = 1:𝑘]
    𝐄s = [paths[i][2] for i = 1:𝑘]
    # merge
    𝐳_k = [z_min]
    𝐄_k = (eltype(𝐄s[begin]))[]
    pointers = ones(Int, 𝑘)
    z_left, z_right = z_min, z_min
    while z_right < z_max
        i_next = argmin((i -> 𝐳s[i][pointers[i] + 1]).(1:𝑘))
        z_right = 𝐳s[i_next][pointers[i_next] + 1]

        push!(𝐳_k, z_right)
        push!(𝐄_k, mean(𝐄s[i][pointers[i]] for i = 1:𝑘))

        z_left = z_right
        pointers[i_next] += 1
    end
    @assert z_right == z_max
    return 𝐳_k, 𝐄_k
end

function compute_Z_CV(k_obs, K, 𝐳, 𝐄)
    z_min, z_max = 𝐳[k_obs][begin], 𝐳[k_obs][end]
    # 各kでの𝐳_kでマージソートしていきながら各区間でE_kobsとE_k′を比較してE_kobs = min E_kとなる区間を求める
    Z_CV = ClosedInterval{eltype(𝐳[k_obs])}[] # init Z_CV
    pointers = Dict(k => 1 for k ∈ K)
    z_left, z_right = z_min, z_min
    while z_right < z_max
        # this algorithm is just like merge sort for 𝐳_k (k ∈ K)
        k_next = K[argmin((k -> 𝐳[k][pointers[k] + 1]).(K))]
        z_right = 𝐳[k_next][pointers[k_next] + 1]
        z_left == z_right && (pointers[k_next] += 1; continue)
        I = z_left..z_right # current interval

        Z_I = ClosedInterval{eltype(𝐳[k_obs])}[] # truncated interval on current I
        for k′ ∈ K
            k_obs == k′ && continue
            Z_I = I ∩ (𝐄[k_obs][pointers[k_obs]] ≤ 𝐄[k′][pointers[k′]])
            isempty(Z_I) && break
        end # Z_I = {z ∈ I | E_{k_obs}(z) = min_k E_k(z)}
        Z_CV = vcat(Z_CV, Z_I)

        z_left = z_right
        pointers[k_next] += 1
    end
    @assert z_right == z_max
    return Z_CV
end

function parametric_SFS_SI_TVS(𝐗, 𝐲_obs, 𝚺, K; ratio=0.5, selection_event=Active())
    𝐴_obs, 𝐬_obs, k_obs = sfs_TVS(𝐗, 𝐲_obs, K; ratio=ratio)
    𝐗_obs = 𝐗[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    p_values = Vector{Float64}(undef, length(𝐴_obs)) # selective p-value list
    for j = 1:length(𝐴_obs)
        # parametric representation for 𝐲(z)
        z_obs, σ²_z, 𝐚, 𝐛 = SIforSFS.parametric_representation(𝜼[:, j], 𝐲_obs, 𝚺)

        # compute Z_CV = {z | k(z) = k_obs}
        𝐗_train, 𝐗_val = train_val_split(𝐗; ratio=ratio)
        𝐚_train, 𝐚_val = train_val_split(𝐚; ratio=ratio)
        𝐛_train, 𝐛_val = train_val_split(𝐛; ratio=ratio)
        paths = (k -> compute_val_error_path(k, 𝐗_train, 𝐗_val, 𝐚_train, 𝐚_val, 𝐛_train, 𝐛_val, -abs(z_obs) - 3 * √(σ²_z), abs(z_obs) + 3 * √(σ²_z))).(K)
        𝐳 = Dict(k => paths[i][1] for (i, k) ∈ enumerate(K))
        𝐄 = Dict(k => paths[i][2] for (i, k) ∈ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, 𝐳, 𝐄)

        # compute Z_alg = {z | 𝐴(z) = 𝐴_obs}
        Z_alg = truncated_interval(selection_event, 𝐴_obs, 𝐬_obs, 𝐗, 𝐚, 𝐛, k_obs, -abs(z_obs) - 3 * √(σ²_z), abs(z_obs) + 3 * √(σ²_z))

        Z = Z_alg ∩ Z_CV
        𝜋_j = cdf(TruncatedDistribution(Normal(0, √(σ²_z)), Z), z_obs)
        p_values[j] = 2 * min(𝜋_j, 1 - 𝜋_j)
    end
    return (𝐴_obs, p_values)
end

function parametric_SFS_SI_CV(𝐗, 𝐲_obs, 𝚺, K; 𝑘=length(𝐲_obs), selection_event=Active())
    𝐴_obs, 𝐬_obs, k_obs = sfs_CV(𝐗, 𝐲_obs, K; 𝑘=𝑘)
    𝐗_obs = 𝐗[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    p_values = Vector{Float64}(undef, length(𝐴_obs)) # selective p-value list
    for j = 1:length(𝐴_obs)
        # parametric representation for 𝐲(z)
        z_obs, σ²_z, 𝐚, 𝐛 = SIforSFS.parametric_representation(𝜼[:, j], 𝐲_obs, 𝚺)

        # compute Z_CV = {z | k(z) = k_obs}
        paths = (k -> compute_cv_path(k, 𝐗, 𝐚, 𝐛, 𝑘, -abs(z_obs) - 3 * √(σ²_z), abs(z_obs) + 3 * √(σ²_z))).(K)
        𝐳 = Dict(k => paths[i][1] for (i, k) ∈ enumerate(K))
        𝐄 = Dict(k => paths[i][2] for (i, k) ∈ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, 𝐳, 𝐄)

        # compute Z_alg = {z | 𝐴(z) = 𝐴_obs}
        Z_alg = truncated_interval(selection_event, 𝐴_obs, 𝐬_obs, 𝐗, 𝐚, 𝐛, k_obs, -abs(z_obs) - 3 * √(σ²_z), abs(z_obs) + 3 * √(σ²_z))

        Z = Z_alg ∩ Z_CV
        𝜋_j = cdf(TruncatedDistribution(Normal(0, √(σ²_z)), Z), z_obs)
        p_values[j] = 2 * min(𝜋_j, 1 - 𝜋_j)
    end
    return (𝐴_obs, p_values)
end

function parametric_SFS_CI_TVS(𝐗, 𝐲_obs, 𝚺, K; ratio=0.5, selection_event=Active(), α=0.05)
    𝐴_obs, 𝐬_obs, k_obs = sfs_TVS(𝐗, 𝐲_obs, K; ratio=ratio)
    𝐗_obs = 𝐗[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    CIs = Vector{ClosedInterval{Float64}}(undef, length(𝐴_obs)) # selective CI list
    for j = 1:length(𝐴_obs)
        # parametric representation for 𝐲(z)
        z_obs, σ²_z, 𝐚, 𝐛 = SIforSFS.parametric_representation(𝜼[:, j], 𝐲_obs, 𝚺)

        # compute Z_CV = {z | k(z) = k_obs}
        𝐗_train, 𝐗_val = train_val_split(𝐗; ratio=ratio)
        𝐚_train, 𝐚_val = train_val_split(𝐚; ratio=ratio)
        𝐛_train, 𝐛_val = train_val_split(𝐛; ratio=ratio)
        paths = (k -> compute_val_error_path(k, 𝐗_train, 𝐗_val, 𝐚_train, 𝐚_val, 𝐛_train, 𝐛_val, -20 * √(σ²_z), 20 * √(σ²_z))).(K)
        𝐳 = Dict(k => paths[i][1] for (i, k) ∈ enumerate(K))
        𝐄 = Dict(k => paths[i][2] for (i, k) ∈ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, 𝐳, 𝐄)

        # compute Z_alg = {z | 𝐴(z) = 𝐴_obs}
        Z_alg = truncated_interval(selection_event, 𝐴_obs, 𝐬_obs, 𝐗, 𝐚, 𝐛, k_obs, -20 * √(σ²_z), 20 * √(σ²_z))

        Z = Z_alg ∩ Z_CV

        CIs[j] = confidence_interval(z_obs, √(σ²_z), Z; α=α)
    end
    return (𝐴_obs, CIs)
end

function parametric_SFS_CI_CV(𝐗, 𝐲_obs, 𝚺, K; 𝑘=length(𝐲_obs), selection_event=Active(), α=0.05)
    𝐴_obs, 𝐬_obs, k_obs = sfs_CV(𝐗, 𝐲_obs, K; 𝑘=𝑘)
    𝐗_obs = 𝐗[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    CIs = Vector{ClosedInterval{Float64}}(undef, length(𝐴_obs)) # selective CI list
    for j = 1:length(𝐴_obs)
        # parametric representation for 𝐲(z)
        z_obs, σ²_z, 𝐚, 𝐛 = SIforSFS.parametric_representation(𝜼[:, j], 𝐲_obs, 𝚺)

        # compute Z_CV = {z | k(z) = k_obs}
        paths = (k -> compute_cv_path(k, 𝐗, 𝐚, 𝐛, 𝑘, -20 * √(σ²_z), 20 * √(σ²_z))).(K)
        𝐳 = Dict(k => paths[i][1] for (i, k) ∈ enumerate(K))
        𝐄 = Dict(k => paths[i][2] for (i, k) ∈ enumerate(K))
        Z_CV = compute_Z_CV(k_obs, K, 𝐳, 𝐄)

        # compute Z_alg = {z | 𝐴(z) = 𝐴_obs}
        Z_alg = truncated_interval(selection_event, 𝐴_obs, 𝐬_obs, 𝐗, 𝐚, 𝐛, k_obs, -20 * √(σ²_z), 20 * √(σ²_z))

        Z = Z_alg ∩ Z_CV
        CIs[j] = confidence_interval(z_obs, √(σ²_z), Z; α=α)
    end
    return (𝐴_obs, CIs)
end

end # module ParametricCVforSFS


# # test
# N = 100
# β₀ = 0.0
# β = rand(10)
# K = (1, 3, 5, 7, 9)
# ratio = 0.9
# 𝑘 = 5
# using LinearAlgebra
# 𝚺 = I

# 𝐗, 𝐲_obs = make_dataset(β₀, β, N)
# 𝐗, 𝐲_obs = centering(𝐗, 𝐲_obs)
# 𝐴_obs, _, k_obs = sfs_TVS(𝐗, 𝐲_obs, K; ratio=ratio)
# 𝐴_obs, _, k_obs = sfs_CV(𝐗, 𝐲_obs, K; 𝑘=𝑘)

# 𝐗_obs = 𝐗[:, 𝐴_obs]
# 𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
# 𝜼ⱼ = 𝜼[:, 1]
# z_obs = 𝜼ⱼ' * 𝐲_obs
# σ²_z = 𝜼ⱼ' * 𝚺 * 𝜼ⱼ # variance of z_obs
# 𝐛 = 𝚺 * 𝜼ⱼ / σ²_z
# 𝐚 = 𝐲_obs - 𝐛 * 𝜼ⱼ' * 𝐲_obs
# @assert 𝐚 + 𝐛 * z_obs ≈ 𝐲_obs
# # 𝐗_train, 𝐗_val = train_val_split(𝐗; ratio=ratio)
# # 𝐚_train, 𝐚_val = train_val_split(𝐚; ratio=ratio)
# # 𝐛_train, 𝐛_val = train_val_split(𝐛; ratio=ratio)
# # paths = (k -> compute_val_error_path(k, 𝐗_train, 𝐗_val, 𝐚_train, 𝐚_val, 𝐛_train, 𝐛_val, -20 * √(σ²_z), 20 * √(σ²_z))).(1:K) # TVS version
# paths = (k -> compute_cv_path(k, 𝐗, 𝐚, 𝐛, 𝑘, -20 * √(σ²_z), 20 * √(σ²_z))).(K) # CV version
# 𝐳 = Dict(k => paths[i][1] for (i, k) ∈ enumerate(K))
# 𝐄 = Dict(k => paths[i][2] for (i, k) ∈ enumerate(K))
# Z_CV = compute_Z_CV(k_obs, K, 𝐳, 𝐄)
# # create Z_new
# Z_new = eltype(Z_CV)[Z_CV[begin]]
# for I in (@view Z_CV[2:end])
#     if abs(Z_new[end].right - I.left) < 1e-2 * √(σ²_z) 
#         Z_new[end] = (Z_new[end].left)..(I.right)
#     else
#         push!(Z_new, I)
#     end
# end
# Z_new
# function val_error(𝐳_k, 𝐄_k)
#     T = length(𝐄_k) # change point size
#     input = Vector{Float64}[]
#     output = Vector{Float64}[]
#     for t in 1:T
#         z_grid = range(𝐳_k[t], 𝐳_k[t + 1], length=10000)
#         input = vcat(input, z_grid)
#         output = vcat(output, 𝐄_k[t].(z_grid))
#     end
#     return input, output
# end
# using Plots
# plot()
# minE, maxE = Ref(Inf), Ref(-Inf)
# for k ∈ K
#     input, output = val_error(𝐳[k], 𝐄[k])
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
#     stem!(interval.left, minE[] - 0.01, maxE[] + 0.01; label=nothing, c=:gray, α=0.5, ls=:dashdot)
#     stem!(interval.right, minE[] - 0.01, maxE[] + 0.01; label=nothing, c=:gray, α=0.5, ls=:dashdot)
# end
# using LaTeXStrings
# p = plot!(; xlabel=L"z", ylabel=L"E_k(z)", title=L"k_{\mathrm{obs}}=%$(k_obs)")
# savefig(p, "cv_demo.pdf")