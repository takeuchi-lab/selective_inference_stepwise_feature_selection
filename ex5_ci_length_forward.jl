using Random; Random.seed!(1234)

include("src/data.jl")
using .DataUtil
include("src/forward_stepwise/selective_inference.jl")
using .SIforSFS
include("src/forward_stepwise/data_splitting.jl")
import .DS
using LinearAlgebra:I
using ProgressMeter
using Plots, StatsPlots, Statistics

function main()
    # parameters
    N = 100
    β₀ = 0.0
    𝛃 = [0.25, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0]
    p = length(𝛃)
    k = 9
    α = 0.05
    iter_size = 100

    # generating objects of comparison methods
    events = (Active(), ActiveOrder(), ActiveSign(), ActiveSignOrder())

    CI_DS = fill(NaN, p, iter_size)
    CI_selective = fill(NaN, p, iter_size, length(events))

    CI_length(CI) = CI.right - CI.left
    # experiment
    @showprogress for i = 1:iter_size
        𝐗, 𝐲_obs = centering(make_dataset(β₀, 𝛃, N)...)
        try
            # selective
            for (j, e) in enumerate(events)
                𝐴_obs, CIs = parametric_SFS_CI(𝐗, 𝐲_obs, k, I; selection_event=e, α=α)
                CI_selective[𝐴_obs, i, j] = CI_length.(CIs)
            end
            # DS
            𝐴_obs, CIs = DS.DS_CI(𝐗, 𝐲_obs, k, I; α=α)
            CI_DS[𝐴_obs, i] = CI_length.(CIs)
        catch e
            print("ignore this result: ")
            println(e)
        end
    end

    filter_nan(CIs) = filter(ci -> !isnan(ci), CIs)
    results = [filter_nan(CI_selective[:,:,j]) for j in eachindex(events)]
    push!(results, filter_nan(CI_DS))
    Cs = [:red3, :orange3, :green3, :blue3, :purple3]
    p = plot()
    for i in eachindex(results)
        violin!(p, fill(i, length(results[i])), results[i]; c=Cs[i], label=nothing)
    end
    plot!(p; xticks=(1:length(results), ["Homotopy", "Homotopy-H", "Homotopy-S", "Polytope", "DS"]))
    plot!(p; xlabel="method", ylabel="length of confidence interval")
    # scatter!(p, 1:length(results), median.(results); c=:magenta, label=nothing)
    plot!(p; guidefontsize=16, legendfontsize=12, tickfontsize=10)
    plot!(p; ylims=(0, 7))
    savefig(p, "img/CI_length.pdf")
end
