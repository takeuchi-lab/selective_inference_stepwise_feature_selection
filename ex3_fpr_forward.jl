using Random; Random.seed!(1234)

include("src/data.jl")
using .DataUtil
include("src/forward_stepwise/selective_inference.jl")
using .SIforSFS
include("src/forward_stepwise/data_splitting.jl")
import .DS
using LinearAlgebra:I
using Distributions
using ProgressMeter

const N = 100
const Ī²ā = 0.0
const š = (Float64)[0, 0, 0, 0, 0] # null model
const k = 3

# FPR experiment
function FPR_experiment(Ī²ā, š, N, k; event=Active(), Ī±=0.05, denominator_upper=100, iter_size=5)
    H0 = filter(i -> iszero(š[i]), 1:length(š)) # {j ā {1,...,d} | H_{0,j} is true}
    FPR = Array{Float64}(undef, length(H0), iter_size)
    @showprogress "[$(N), $(event)]: " for l = 1:iter_size
        numerator = zeros(Int, length(H0))
        denominator = zeros(Int, length(H0))
        for _ = 1:denominator_upper
            š, š²_obs = centering(make_dataset(Ī²ā, š, N)...)
            š“_obs, p_values = parametric_SFS_SI(š, š²_obs, k, I; selection_event=event)
            for (j, p_value) in zip(š“_obs, p_values)
                for (i, jā²) in enumerate(H0)
                    if j == jā² # H_{0,j} is true
                        numerator[i] += p_value < Ī± ? 1 : 0 
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

Ī± = 0.05
zero_num = length(filter(x -> iszero(x), š))
Ns = [50, 100, 150, 200]
iter_size = 20
FPR_A = fill(NaN, zero_num, iter_size, length(Ns))
FPR_Ao = fill(NaN, zero_num, iter_size, length(Ns))
FPR_As = fill(NaN, zero_num, iter_size, length(Ns))
FPR_Aso = fill(NaN, zero_num, iter_size, length(Ns))
FPR_DS = fill(NaN, zero_num, iter_size, length(Ns))
for (i, N) in enumerate(Ns)
    FPR_A[:,:,i] = Main.FPR_experiment(Ī²ā, š, N, k; event=Active(), Ī±=Ī±, iter_size=iter_size)
    FPR_As[:,:,i] = Main.FPR_experiment(Ī²ā, š, N, k; event=ActiveSign(), Ī±=Ī±, iter_size=iter_size)
    FPR_Ao[:,:,i] = Main.FPR_experiment(Ī²ā, š, N, k; event=ActiveOrder(), Ī±=Ī±, iter_size=iter_size)
    FPR_Aso[:,:,i] = Main.FPR_experiment(Ī²ā, š, N, k; event=ActiveSignOrder(), Ī±=Ī±, iter_size=iter_size)
    FPR_DS[:,:,i] = DS.FPR_experiment(Ī²ā, š, N, k; Ī±=Ī±, iter_size=iter_size)
end

using Statistics
FPR_A_mean = reshape(mean(FPR_A, dims=2), zero_num, length(Ns))
FPR_As_mean = reshape(mean(FPR_As, dims=2), zero_num, length(Ns))
FPR_Ao_mean = reshape(mean(FPR_Ao, dims=2), zero_num, length(Ns))
FPR_Aso_mean = reshape(mean(FPR_Aso, dims=2), zero_num, length(Ns))
FPR_DS_mean = reshape(mean(FPR_DS, dims=2), zero_num, length(Ns))

using Plots
Cs = [:red, :orange, :green, :blue, :purple]
p = Array{typeof(plot())}(undef, zero_num)
for j in 1:zero_num
    plot()
    plot!(1:length(Ns), FPR_A_mean[j,:], label="Homotopy", color=Cs[1], markershape=:circle)
    plot!(1:length(Ns), FPR_Ao_mean[j,:], label="Homotopy-H", color=Cs[2], markershape=:circle)
    plot!(1:length(Ns), FPR_As_mean[j,:], label="Homotopy-S", color=Cs[3], markershape=:circle)
    plot!(1:length(Ns), FPR_Aso_mean[j,:], label="Polytope", color=Cs[4], markershape=:circle)
    plot!(1:length(Ns), FPR_DS_mean[j,:], label="DS", color=Cs[5], markershape=:circle)
    # plot setting
    plot!(; ylims=(0, 0.3))
    plot!(; xlabel="Sample size", ylabel="False Positive Rate (FPR)")
    plot!(; legend=:topright)
    plot!(; guidefontsize=16, legendfontsize=12, tickfontsize=10)
    p[j] = plot!(; xticks=(1:length(Ns), Ns))
    savefig(p[j], "img/FPR/FPR$(j).pdf")
end