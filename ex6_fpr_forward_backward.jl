using Random; Random.seed!(1234);

include("src/data.jl")
using .DataUtil
include("src/stepwise/SelectiveInference.jl")
using .SelectiveInference
include("src/selective_p.jl")
using .MultiTruncatedDistributions
using LinearAlgebra:I
using Distributions
using IntervalSets
using Parameters
using ProgressMeter
using Plots

function selective_p_rand(test_prob::SelectiveHypothesisTest; isOC=false)
    @unpack alg, alg_result, test_stats, 𝐗, 𝐲_obs, 𝚺 = test_prob
    M_obs = collect(alg_result.model.M)
    isempty(M_obs) && return NaN
    test_stat = test_stats[rand(1:length(M_obs))]
    z_obs, σ²_z = test_stat.z_obs, test_stat.σ²_z

    # compute Z = {z | selection_event}
    z_min = -abs(z_obs) - 10 * √(σ²_z); z_max = abs(z_obs) + 10 * √(σ²_z) 
    Z = isOC ? SelectiveInference.region(𝐗, 𝐲_obs, 𝚺, test_stat, alg)[2] ∩ [z_min..z_max] : SelectiveInference.truncated_interval(alg_result.model, 𝐗, 𝚺, test_stat, alg; z_min=z_min, z_max=z_max) 

    𝜋 = cdf(TruncatedDistribution(Normal(0, √(σ²_z)), Z), z_obs)
    return 2 * min(𝜋, 1 - 𝜋)
end

# FPR experiment
function FPR_experiment(param::DataParam, alg::StepwiseFeatureSelection;
    isOC=false,
    α=0.05,
    denominator_upper=100,
    iter_size=5)
    FPR = fill(NaN, iter_size)
    @showprogress for l = 1:iter_size
        numerator = 0
        denominator = 0
        for _ = 1:denominator_upper
            try
                𝐗, 𝐲_obs = make_dataset(param)
                alg_result = stepwise(𝐗, 𝐲_obs, I, alg)
                p_value = selective_p_rand(SelectiveHypothesisTest(alg, alg_result, 𝐗, 𝐲_obs, I); isOC=isOC)
                isnan(p_value) && continue
                numerator += p_value < α ? 1 : 0 
                denominator += 1
            catch e
println(e)
                continue
            end
        end
        FPR[l] = numerator / denominator
    end
    return FPR
end

using JLD
function main(alg::StepwiseFeatureSelection)
    # parameters
    β₀ = 0.0
    Ns = (50, 100, 150)
    ps = (10, 20, 50)

    iter_size = 20
        FPR = Array{Float64}(undef, iter_size, length(Ns), length(ps))
    FPR_OC = similar(FPR)
    α = 0.05
    for (i, N) in enumerate(Ns)
        for (j, p) in enumerate(ps)
            FPR[:,i,j] = FPR_experiment(DataParam(N=N, β₀=β₀, 𝛃=zeros(p)), alg; isOC=false, α=α, iter_size=iter_size)
            FPR_OC[:,i,j] = FPR_experiment(DataParam(N=N, β₀=β₀, 𝛃=zeros(p)), alg; isOC=true, α=α, iter_size=iter_size)
        end
    end

    FPR_mean = reshape(mean(FPR, dims=1), size(FPR, 2), size(FPR, 3))
    FPR_OC_mean = reshape(mean(FPR_OC, dims=1), size(FPR_OC, 2), size(FPR_OC, 3))

    h_cs = (:red2, :violetred2, :darkred)
    o_cs = (:blue2, :skyblue, :navyblue)
    p = plot()
    for j = 1:size(FPR, 3) # σ
        plot!(p, 1:length(Ns), FPR_mean[:,j], label="Homotopy, p=$(ps[j])", color=h_cs[j], markershape=:circle)
        plot!(p, 1:length(Ns), FPR_OC_mean[:,j], label="Quadratic, p=$(ps[j])", color=o_cs[j], markershape=:rect)
    end
    # plot setting
    plot!(p; ylims=(0, 0.3))
    plot!(p; xlabel="Sample size", ylabel="False Positive Rate (FPR)")
    plot!(p; legend=:topright)
    plot!(p; guidefontsize=16, legendfontsize=12, tickfontsize=10)
    plot!(p; xticks=(1:length(Ns), Ns))
    savefig(p, "img/FPR_forward_backward.pdf")
end
