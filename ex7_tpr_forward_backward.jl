using Random; Random.seed!(1234);

include("src/data.jl")
using .DataUtil
include("src/stepwise/SelectiveInference.jl")
using .SelectiveInference
include("src/selective_p.jl")
using .MultiTruncatedDistributions
include("src/intervals.jl")
using .Intersection
using LinearAlgebra:I
using Distributions
using IntervalSets
using Parameters
using ProgressMeter

using Plots
default(:bglegend, plot_color(default(:bg), 0.8))
default(:fglegend, plot_color(ifelse(isdark(plot_color(default(:bg))), :white, :black), 0.6))

function selective_p1(test_prob::SelectiveHypothesisTest; isOC=false)
    @unpack alg, alg_result, test_stats, 𝐗, 𝐲_obs, 𝚺 = test_prob
    M_obs = alg_result.model.M
    1 ∈ M_obs || return NaN
    for (i, j) in enumerate(M_obs)
        # β₁ ≠ 0
        j == 1 || continue
        test_stat = test_stats[i]
        z_obs, σ²_z = test_stat.z_obs, test_stat.σ²_z

        # compute Z = {z | selection_event}
        z_min = -abs(z_obs) - 10 * √(σ²_z); z_max = abs(z_obs) + 10 * √(σ²_z) 
        Z = isOC ?
        SelectiveInference.region(𝐗, 𝐲_obs, 𝚺, test_stat, alg)[2] ∩ [z_min..z_max] :
        SelectiveInference.truncated_interval(alg_result.model, 𝐗, 𝚺, test_stat, alg;
            z_min=z_min, z_max=z_max) 

        𝜋 = cdf(TruncatedDistribution(Normal(0, √(σ²_z)), Z), z_obs)
        return 2 * min(𝜋, 1 - 𝜋)
    end
end

# TPR experiment
    function TPR_experiment(param::DataParam, alg::StepwiseFeatureSelection;
    isOC=false,
    α=0.05,
    denominator_upper=100,
    iter_size=5)
    @assert param.𝛃[1] ≠ 0
    TPR = fill(NaN, iter_size)
    @showprogress "[isOC:$(isOC),N:$(param.N),p:$(length(param.𝛃)),β:$(param.𝛃[1])]" for l = 1:iter_size
        numerator = 0
        denominator = 0
        for _ = 1:denominator_upper
            try
                𝐗, 𝐲_obs = make_dataset(param)
                alg_result = stepwise(𝐗, 𝐲_obs, I, alg)
                p_value = selective_p1(SelectiveHypothesisTest(alg, alg_result, 𝐗, 𝐲_obs, I); isOC=isOC)
                isnan(p_value) && continue
                numerator += p_value < α ? 1 : 0 
                denominator += 1
            catch e
                println(e)
                continue
            end
        end
        TPR[l] = numerator / denominator
    end
    return TPR
end

function main(alg::StepwiseFeatureSelection)
    # parameters
    β₀ = 0.0
    Ns = (50, 100, 150)
    ps = (10, 20, 50)
    βs = (0.01, 0.25, 0.5, 1.)

    iter_size = 10
    TPR = Array{Float64}(undef, iter_size, length(Ns), length(ps), length(βs))
    TPR_OC = similar(TPR)
    α = 0.05

    # run experiment
    for (i, N) in enumerate(Ns)
        for (j, p) in enumerate(ps)
            for (k, β) in enumerate(βs)
                𝛃 = [fill(β, floor(Int, p / 2)); zeros(ceil(Int, p / 2))]
                param = DataParam(N=N, β₀=β₀, 𝛃=𝛃)
                TPR[:,i,j,k] = TPR_experiment(param, alg; isOC=false, α=α, iter_size=iter_size)
                TPR_OC[:,i,j,k] = TPR_experiment(param, alg; isOC=true, α=α, iter_size=iter_size)
            end
        end
    end

    # plot results
    TPR_mean = reshape(mean(TPR, dims=1), size(TPR, 2), size(TPR, 3), size(TPR, 4))
    TPR_SE = reshape(std(TPR, dims=1) ./ √size(TPR, 1), size(TPR, 2), size(TPR, 3), size(TPR, 4))
    TPR_OC_mean = reshape(mean(TPR_OC, dims=1), size(TPR_OC, 2), size(TPR_OC, 3), size(TPR_OC, 4))
    TPR_OC_SE = reshape(std(TPR_OC, dims=1) ./ √size(TPR_OC, 1), size(TPR_OC, 2), size(TPR_OC, 3), size(TPR_OC, 4))
    p = Array{typeof(plot())}(undef, length(βs))
    h_cs = (:red2, :violetred2, :darkred)
    o_cs = (:blue2, :skyblue, :darkblue)
    for k = eachindex(βs)
        p[k] = plot()
        for j = eachindex(ps) # p
            # Homotopy
            plot!(p[k], 1:length(Ns), (TPR_mean - TPR_SE)[:,j,k]; fillrange=(TPR_mean + TPR_SE)[:,j,k], fillalpha=0.2, α=0, label=nothing, color=h_cs[j])
            plot!(p[k], 1:length(Ns), TPR_mean[:,j,k], label="Homotopy, p=$(ps[j])", color=h_cs[j], markershape=:circle)
            # Quadratic
            plot!(p[k], 1:length(Ns), (TPR_OC_mean - TPR_OC_SE)[:,j,k]; fillrange=(TPR_OC_mean + TPR_OC_SE)[:,j,k], fillalpha=0.2, α=0, label=nothing, color=o_cs[j])
            plot!(p[k], 1:length(Ns), TPR_OC_mean[:,j,k], label="Quadratic, p=$(ps[j])", color=o_cs[j], markershape=:rect)
        end
        # plot setting
        plot!(p[k]; ylims=(0, 1))
        plot!(p[k]; legend=:best)
        plot!(p[k]; xlabel="Sample size", ylabel="True Positive Rate (TPR)")
        plot!(p[k]; guidefontsize=14, legendfontsize=9, tickfontsize=9)
        plot!(p[k]; xticks=(1:length(Ns), Ns))
        plot!(p[k]; title="signal:$(βs[k])")
    end
    fig = plot(p..., layout=(1, 4), titlefontsize=16, size=(900, 900))
    savefig(fig, "img/TPR_forward_backward.pdf")
end

main(StepwiseFeatureSelection())