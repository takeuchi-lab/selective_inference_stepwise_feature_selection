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
const β₀ = 0.0
const 𝛃 = (Float64)[0.25, 0.25, 0, 0, 0]
const k = 3

# TPR experiment
function TPR_experiment(β₀, 𝛃, N, k; event=Active(), α=0.05, denominator_upper=100, iter_size=5)
    H_1 = filter(i -> !iszero(𝛃[i]), 1:length(𝛃)) # {j ∈ {1,...,d} | H_{1,j} is true}
    TPR = Array{Float64}(undef, length(H_1), iter_size)
    @showprogress "[$(N), $(event)]: " for l = 1:iter_size
        numerator = zeros(Int, length(H_1))
        denominator = zeros(Int, length(H_1))
        for _ = 1:denominator_upper
            𝐗, 𝐲_obs = centering(make_dataset(β₀, 𝛃, N)...)
            𝐴_obs, p_values = parametric_SFS_SI(𝐗, 𝐲_obs, k, I; selection_event=event)
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

α = 0.05
nonzero_num = length(filter(x -> !iszero(x), 𝛃))
Ns = [50, 100, 150, 200]
iter_size = 5
TPR_A = fill(NaN, nonzero_num, iter_size, length(Ns))
TPR_Ao = fill(NaN, nonzero_num, iter_size, length(Ns))
TPR_As = fill(NaN, nonzero_num, iter_size, length(Ns))
TPR_Aso = fill(NaN, nonzero_num, iter_size, length(Ns))
TPR_DS = fill(NaN, nonzero_num, iter_size, length(Ns))
for (i, N) in enumerate(Ns)
    TPR_A[:,:,i] = Main.TPR_experiment(β₀, 𝛃, N, k; ℰ=Active(), α=α, iter_size=iter_size)
    TPR_Ao[:,:,i] = Main.TPR_experiment(β₀, 𝛃, N, k; ℰ=ActiveOrder(), α=α, iter_size=iter_size)
    TPR_As[:,:,i] = Main.TPR_experiment(β₀, 𝛃, N, k; ℰ=ActiveSign(), α=α, iter_size=iter_size)
    TPR_Aso[:,:,i] = Main.TPR_experiment(β₀, 𝛃, N, k; ℰ=ActiveSignOrder(), α=α, iter_size=iter_size)
    TPR_DS[:,:,i] = DS.TPR_experiment(β₀, 𝛃, N, k; α=α, iter_size=iter_size)
end

using Statistics
TPR_A_mean = reshape(mean(TPR_A, dims=2), nonzero_num, length(Ns))
TPR_A_SE = reshape(std(TPR_A, dims=2) ./ √iter_size, nonzero_num, length(Ns))
TPR_Ao_mean = reshape(mean(TPR_Ao, dims=2), nonzero_num, length(Ns))
TPR_Ao_SE = reshape(std(TPR_Ao, dims=2) ./ √iter_size, nonzero_num, length(Ns))
TPR_As_mean = reshape(mean(TPR_As, dims=2), nonzero_num, length(Ns))
TPR_As_SE = reshape(std(TPR_As, dims=2) ./ √iter_size, nonzero_num, length(Ns))
TPR_Aso_mean = reshape(mean(TPR_Aso, dims=2), nonzero_num, length(Ns))
TPR_Aso_SE = reshape(std(TPR_Aso, dims=2) ./ √iter_size, nonzero_num, length(Ns))
TPR_DS_mean = reshape(mean(TPR_DS, dims=2), nonzero_num, length(Ns))
TPR_DS_SE = reshape(std(TPR_DS, dims=2) ./ √iter_size, nonzero_num, length(Ns))

using Plots
default(:bglegend, plot_color(default(:bg), 0.5))
default(:fglegend, plot_color(ifelse(isdark(plot_color(default(:bg))), :white, :black), 0.6))
Cs = [:red, :orange, :green, :blue, :purple]

p = Array{typeof(plot())}(undef, nonzero_num)
for j in 1:nonzero_num
    plot()
    # Active
    plot!(1:length(Ns), (TPR_A_mean - TPR_A_SE)[j,:]; fillrange=(TPR_A_mean + TPR_A_SE)[j,:], fillalpha=0.2, α=0, label=nothing, color=Cs[1])
    plot!(1:length(Ns), TPR_A_mean[j,:], label="Homotopy", color=Cs[1], markershape=:circle)
    # ActiveOrder
    plot!(1:length(Ns), (TPR_Ao_mean - TPR_Ao_SE)[j,:]; fillrange=(TPR_Ao_mean + TPR_Ao_SE)[j,:], fillalpha=0.2, α=0, label=nothing, color=Cs[2])
    plot!(1:length(Ns), TPR_Ao_mean[j,:], label="Homotopy-H", color=Cs[2], markershape=:circle)
    # ActiveSign
    plot!(1:length(Ns), (TPR_As_mean - TPR_As_SE)[j,:]; fillrange=(TPR_As_mean + TPR_As_SE)[j,:], fillalpha=0.2, α=0, label=nothing, color=Cs[3])
    plot!(1:length(Ns), TPR_As_mean[j,:], label="Homotopy-S", color=Cs[3], markershape=:circle)
    # ActiveSignOrder
    plot!(1:length(Ns), (TPR_Aso_mean - TPR_Aso_SE)[j,:]; fillrange=(TPR_Aso_mean + TPR_Aso_SE)[j,:], fillalpha=0.2, α=0, label=nothing, color=Cs[4])
    plot!(1:length(Ns), TPR_Aso_mean[j,:], label="Polytope", color=Cs[4], markershape=:circle)
    # DS
    plot!(1:length(Ns), (TPR_DS_mean - TPR_DS_SE)[j,:]; fillrange=(TPR_DS_mean + TPR_DS_SE)[j,:], fillalpha=0.2, α=0, label=nothing, color=Cs[5])
    plot!(1:length(Ns), TPR_DS_mean[j,:], label="DS", color=Cs[5], markershape=:circle)
    # plot setting
    plot!(; ylims=(0, 1))
    plot!(; legend=:topleft)
    plot!(; xlabel="Sample size", ylabel="True Positive Rate (TPR)")
    plot!(; guidefontsize=16, legendfontsize=12, tickfontsize=10)
    p[j] = plot!(; xticks=(1:length(Ns), Ns))
    savefig(p[j], "img/TPR/TPR$(j).pdf")
end