module DS

export data_splitting, DS_CI

using Random
using Distributions
using IntervalSets
using LinearAlgebra: I
include("../data.jl")
using .DataUtil
include("sfs.jl")
using .SFS

function data_splitting(𝐗, 𝐲, k, 𝚺; ratio=0.5)
    # data splitting
    idx = shuffle(1:length(𝐲)) # randomize data
    𝐗_FS, 𝐗_SI = train_val_split(𝐗[idx, :]; ratio=ratio)
    𝐲_FS, 𝐲_SI = train_val_split(𝐲[idx]; ratio=ratio)

    # feature selection
    𝐴_obs, _ = sfs(𝐗_FS, 𝐲_FS, k)

    # statistical inference (hypothesis test)
    𝐗_obs = 𝐗_SI[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    p_values = Vector{Float64}(undef, length(𝐴_obs)) # p-value list
    for j = 1:length(𝐴_obs)
        𝜼ⱼ = 𝜼[:, j]
        z_obs = 𝜼ⱼ' * 𝐲_SI
        σ²_z = 𝜼ⱼ' * 𝚺 * 𝜼ⱼ # variance of z_obs
        𝜋_j = cdf(Normal(0, √(σ²_z)), z_obs)
        p_values[j] = 2 * min(𝜋_j, 1 - 𝜋_j)
    end
    return (𝐴_obs, p_values)
end

function DS_CI(𝐗, 𝐲, k, 𝚺; ratio=0.5, α=0.05)
    # data splitting
    idx = shuffle(1:length(𝐲)) # randomize data
    𝐗_FS, 𝐗_SI = train_val_split(𝐗[idx, :]; ratio=ratio)
    𝐲_FS, 𝐲_SI = train_val_split(𝐲[idx]; ratio=ratio)

    # feature selection
    𝐴_obs, _ = sfs(𝐗_FS, 𝐲_FS, k) 

    # statistical inference (estimate confidence interval)
    Φ = cquantile(Normal(0, 1), α / 2)
    𝐗_obs = 𝐗_SI[:, 𝐴_obs]
    𝜼 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
    CIs = Vector{ClosedInterval{Float64}}(undef, length(𝐴_obs)) # CI list
    for j = 1:length(𝐴_obs)
        𝜼ⱼ = 𝜼[:, j]
        z_obs = 𝜼ⱼ' * 𝐲_SI
        σ_z = √(𝜼ⱼ' * 𝚺 * 𝜼ⱼ) # standard deviation of z_obs
        CIs[j] = (z_obs - σ_z * Φ)..(z_obs + σ_z * Φ)
    end
    return (𝐴_obs, CIs)
end

function FPR_experiment(β₀, 𝛃, N, k; α=0.05, denominator_upper=100, iter_size=5, ratio=0.5)
    H_0 = filter(i -> iszero(𝛃[i]), 1:length(𝛃)) # {j ∈ {1,...,d} | H_{0,j} is true}
    FPR = Array{Float64}(undef, length(H_0), iter_size)
    for l = 1:iter_size
        numerator = zeros(Int, length(H_0))
        denominator = zeros(Int, length(H_0))
        for _ = 1:denominator_upper
            𝐗, 𝐲_obs = centering(make_dataset(β₀, 𝛃, N)...)
            𝐴_obs, p_values = data_splitting(𝐗, 𝐲_obs, k, I; ratio=ratio)
            for (j, p_value) in zip(𝐴_obs, p_values)
                for (i, j′) in enumerate(H_0)
                    if j == j′ # H_{0,j} is true
                        numerator[i] += p_value < α ? 1 : 0 
                        denominator[i] += 1
                        break
                    end
                end
            end
        end
        FPR[:, l] = numerator ./ denominator
    end
    return FPR
end

function TPR_experiment(β₀, 𝛃, N, k; α=0.05, denominator_upper=100, iter_size=5, ratio=0.5)
    H_1 = filter(i -> !iszero(𝛃[i]), 1:length(𝛃)) # {j ∈ {1,...,d} | H_{1,j} is true}
    TPR = Array{Float64}(undef, length(H_1), iter_size)
    for l = 1:iter_size
        numerator = zeros(Int, length(H_1))
        denominator = zeros(Int, length(H_1))
        for _ = 1:denominator_upper
            𝐗, 𝐲_obs = centering(make_dataset(β₀, 𝛃, N)...)
            𝐴_obs, p_values = data_splitting(𝐗, 𝐲_obs, k, I; ratio=ratio)
            for (j, p_value) in zip(𝐴_obs, p_values)
                for (i, j′) in enumerate(H_1)
                    if j == j′ # H_{1,j} is true
                        numerator[i] += p_value < α ? 1 : 0 
                        denominator[i] += 1
                        break
                    end
                end
            end
        end
        TPR[:, l] = numerator ./ denominator
    end
    return TPR
end

end # module DS