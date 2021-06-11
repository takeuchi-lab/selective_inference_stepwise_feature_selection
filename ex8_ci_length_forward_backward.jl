using Random; Random.seed!(1234);

include("src/data.jl")
using .DataUtil
include("src/stepwise/SelectiveInference.jl")
using .SelectiveInference
include("src/ci.jl")
using .ConfidenceInterval
using LinearAlgebra:I
using Distributions
using IntervalSets
using Parameters
using ProgressMeter
using Plots, StatsPlots, Statistics, DataFrames

function selective_CI_rand(test_prob::SelectiveHypothesisTest; α=0.05, isOC=false)
    @unpack alg, alg_result, test_stats, 𝐗, 𝐲_obs, 𝚺 = test_prob
    M_obs = collect(alg_result.model.M)
    isempty(M_obs) && return nothing
    test_stat = test_stats[rand(1:length(M_obs))]
    z_obs, σ²_z = test_stat.z_obs, test_stat.σ²_z

    # compute Z = {z | selection_event}
    z_min = -abs(z_obs) - 10 * √(σ²_z); z_max = abs(z_obs) + 10 * √(σ²_z) 
    Z = isOC ? SelectiveInference.region(𝐗, 𝐲_obs, 𝚺, test_stat, alg)[2] ∩ [z_min..z_max] : SelectiveInference.truncated_interval(alg_result.model, 𝐗, 𝚺, test_stat, alg; z_min=z_min, z_max=z_max) 

    return confidence_interval(z_obs, √(σ²_z), Z; α=α)
end

# CI experiment
function CI_experiment(param::DataParam, alg::StepwiseFeatureSelection; α=0.05, isOC=false, iter_size=5)
    CI_lengths = fill(NaN, iter_size)
    @showprogress for i = 1:iter_size
        try
            𝐗, 𝐲_obs = make_dataset(param)
            alg_result = stepwise(𝐗, 𝐲_obs, I, alg)
            test_prob = SelectiveHypothesisTest(alg, alg_result, 𝐗, 𝐲_obs, I)
            CI = selective_CI_rand(test_prob; α=α, isOC=isOC)
            isnothing(CI) && continue
            CI_lengths[i] = CI.right - CI.left
        catch e
            println(e)
            continue
        end
    end
    return CI_lengths
end

using JLD
function main(alg::StepwiseFeatureSelection)
    # parameters
    N = 100
    β₀ = 0.0
    𝛃 = randn(10)
    α = 0.05
    param = DataParam(N=N, β₀=β₀, 𝛃=𝛃)

    iter_size = 100
    CI_lengths = CI_experiment(param, alg; α=α, isOC=false, iter_size=iter_size)
    CI_lengths_OC = CI_experiment(param, alg; α=α, isOC=true, iter_size=iter_size)

    # plot result
    p = plot()
    violin!(p, fill(1, length(CI_lengths)), CI_lengths; c=:red2, label=nothing)
    violin!(p, fill(2, length(CI_lengths_OC)), CI_lengths_OC; c=:blue2, label=nothing)
    plot!(p; xticks=(1:2, ["homotopy", "over-conditioning"]))
    plot!(p; yscale=:log10)
    plot!(p; xlabel="method", ylabel="length of confidence interval")
    plot!(p; guidefontsize=16, legendfontsize=12, tickfontsize=10)
    savefig(p, "CI_length_forward_backward.pdf")
end

main(StepwiseFeatureSelection())