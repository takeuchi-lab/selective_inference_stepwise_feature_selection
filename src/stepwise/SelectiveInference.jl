module SelectiveInference

using LinearAlgebra
using IntervalSets
using Distributions
using Parameters
using Reexport

include("stepwise.jl")
@reexport using .Stepwise
include("../quadratic.jl")
using .Quadratic
include("../intervals.jl")
using .Intersection
include("../selective_p.jl")
using .MultiTruncatedDistributions
include("../ci.jl")
using .ConfidenceInterval

export SelectiveHypothesisTest, selective_p, selective_CI

const ε = 1e-4

struct TestStatistic{A,B,C,D}
    z_obs::A
    σ²_z::B
    𝐚::C
    𝐛::D
end
function parametric_representation(𝛈, 𝐲, 𝚺)
    z = 𝛈' * 𝐲
    σ²_z = 𝛈' * 𝚺 * 𝛈 
    𝐛 = 𝚺 * 𝛈 / σ²_z
    𝐚 = 𝐲 - 𝐛 * 𝛈' * 𝐲
    @assert 𝐚 + 𝐛 * z ≈ 𝐲
    return TestStatistic(z, σ²_z, 𝐚, 𝐛)
end
"Make the hypothesis test problem: βⱼ = 0 or not for all j ∈ `alg_result.model.M`, where βⱼ is the coefficient of the j-th feature."
struct SelectiveHypothesisTest{A <: AbstractAlgorithm,B <: AbstractAlgResult,C,D,E,F}
    alg::A # used feature selection algorithm
    alg_result::B # the result of the algorithm `alg` for `𝐗`, `𝐲_obs`, `𝚺`
    test_stats::C # the information of test statistic for all hypotheses
    𝐗::D # used covariate matrix
    𝐲_obs::E # used response vector
    𝚺::F # used covariance matrix for response vector

    function SelectiveHypothesisTest(alg::S, alg_result::T, 𝐗::U, 𝐲_obs::V, 𝚺::W) where {S,T,U,V,W}
        M_obs = alg_result.model.M
        𝐗_obs = 𝐗[:, M_obs]
        𝚮 = 𝐗_obs / (𝐗_obs' * 𝐗_obs)
        test_stats = [parametric_representation(𝛈, 𝐲_obs, 𝚺) for 𝛈 in eachcol(𝚮)]
        new{S,T,typeof(test_stats),U,V,W}(alg, alg_result, test_stats, 𝐗, 𝐲_obs, 𝚺)
    end
end

function (::AIC)(𝐗, 𝚺⁻¹, test_stat::TestStatistic)
    𝐚, 𝐛 = test_stat.𝐚, test_stat.𝐛
    𝐀 = 𝚺⁻¹ - 𝚺⁻¹ * 𝐗 * pinv(𝐗' * 𝚺⁻¹ * 𝐗) * 𝐗' * 𝚺⁻¹
    return quadratic(𝐛' * 𝐀 * 𝐛, 2 * (𝐚' * 𝐀 * 𝐛), 𝐚' * 𝐀 * 𝐚 + 2 * size(𝐗, 2))
end
function (::BIC)(𝐗, 𝚺⁻¹, test_stat::TestStatistic)
    𝐚, 𝐛 = test_stat.𝐚, test_stat.𝐛
    𝐀 = 𝚺⁻¹ - 𝚺⁻¹ * 𝐗 * pinv(𝐗' * 𝚺⁻¹ * 𝐗) * 𝐗' * 𝚺⁻¹
    return quadratic(𝐛' * 𝐀 * 𝐛, 2 * (𝐚' * 𝐀 * 𝐛), 𝐚' * 𝐀 * 𝐚 + log(size(𝐗, 1)) * size(𝐗, 2))
end

# compute the region correspond to the selected model at `𝐲_z`
function region(𝐗, 𝐲_z, 𝚺,
    test_stat::TestStatistic,
    direction::Bidirection,
    criterion::AbstractCriterion;
    max_step=typemax(Int))
    alg_result = stepwise(𝐗, 𝐲_z, 𝚺, direction, criterion; max_step=max_step)
    history = alg_result.history

    d = size(𝐗, 2)
    model = Model(direction, d)
    𝚺⁻¹ = I / 𝚺
    Ω_z = [(-Inf)..Inf]

    for i in eachindex(history)
        @unpack M, Mᶜ = model
        # i-th selected feature
        idx = history[i]
        𝐗ᵢ = idx > 0 ? 𝐗[:, union(M, idx)] : 𝐗[:, setdiff(M, -idx)]
        ℓᵢ = criterion(𝐗ᵢ, 𝚺⁻¹, test_stat)

        # ℓᵢ is smaller than the value of previous model M
        i != 1 && (Ω_z = Ω_z ∩ (ℓᵢ ≤ criterion(𝐗[:, M], 𝚺⁻¹, test_stat)))
            # delete
        for j ∈ M
            Ω_z = Ω_z ∩ (ℓᵢ ≤ criterion(𝐗[:, setdiff(M, j)], 𝚺⁻¹, test_stat))
        end
        # add
        for j′ ∈ Mᶜ
            Ω_z = Ω_z ∩ (ℓᵢ ≤ criterion(𝐗[:, union(M, j′)], 𝚺⁻¹, test_stat))
        end

        update!(model, idx)
    end; @assert model == alg_result.model

    # stopping rule
    length(history) == max_step && return model, Ω_z
    @unpack M, Mᶜ = model
    minℓ = criterion(𝐗[:, M], 𝚺⁻¹, test_stat)
    ## delete
    for j ∈ M
        Ω_z = Ω_z ∩ (minℓ ≤ criterion(𝐗[:, setdiff(M, j)], 𝚺⁻¹, test_stat))
    end
    ## add
    for j′ ∈ Mᶜ
        Ω_z = Ω_z ∩ (minℓ ≤ criterion(𝐗[:, union(M, j′)], 𝚺⁻¹, test_stat))
    end

    return model, Ω_z
end
function region(𝐗, 𝐲_z, 𝚺,
    test_stat::TestStatistic,
    direction::Forward,
    criterion::AbstractCriterion;
    max_step=typemax(Int))
    alg_result = stepwise(𝐗, 𝐲_z, 𝚺, direction, criterion; max_step=max_step)
    history = alg_result.history

    d = size(𝐗, 2)
    model = Model(direction, d)
    𝚺⁻¹ = I / 𝚺
    Ω_z = [(-Inf)..Inf]
        
    for i in eachindex(history)
        @unpack M, Mᶜ = model
        # i-th selected feature
        idx = history[i]
        𝐗ᵢ = 𝐗[:, union(M, idx)]
        ℓᵢ = criterion(𝐗ᵢ, 𝚺⁻¹, test_stat)

        # ℓᵢ is smaller than the value of previous model M
        i != 1 && (Ω_z = Ω_z ∩ (ℓᵢ ≤ criterion(𝐗[:, M], 𝚺⁻¹, test_stat)))
        # add
        for j′ ∈ Mᶜ
            Ω_z = Ω_z ∩ (ℓᵢ ≤ criterion(𝐗[:, union(M, j′)], 𝚺⁻¹, test_stat))
        end

        update!(model, idx)
    end; @assert model == alg_result.model

    # stopping rule
    length(history) == max_step && return model, Ω_z
    @unpack M, Mᶜ = model
    minℓ = criterion(𝐗[:, M], 𝚺⁻¹, test_stat)
    ## add
    for j′ ∈ Mᶜ
        Ω_z = Ω_z ∩ (minℓ ≤ criterion(𝐗[:, union(M, j′)], 𝚺⁻¹, test_stat))
    end

    return model, Ω_z
end
function region(𝐗, 𝐲_z, 𝚺,
    test_stat::TestStatistic,
    direction::Backward,
    criterion::AbstractCriterion;
    max_step=typemax(Int))
    alg_result = stepwise(𝐗, 𝐲_z, 𝚺, direction, criterion; max_step=max_step)
    history = alg_result.history

    d = size(𝐗, 2)
    model = Model(direction, d)
    𝚺⁻¹ = I / 𝚺
    Ω_z = [(-Inf)..Inf]
        
    for i in eachindex(history)
        M = model.M
        # i-th selected feature
        idx = history[i]
        𝐗ᵢ = 𝐗[:, setdiff(M, -idx)]
        ℓᵢ = criterion(𝐗ᵢ, 𝚺⁻¹, test_stat)

        # ℓᵢ is smaller than the value of previous model M
        i != 1 && (Ω_z = Ω_z ∩ (ℓᵢ ≤ criterion(𝐗[:, M], 𝚺⁻¹, test_stat)))
            # delete
        for j ∈ M
            Ω_z = Ω_z ∩ (ℓᵢ ≤ criterion(𝐗[:, setdiff(M, j)], 𝚺⁻¹, test_stat))
        end

        update!(model, idx)
    end; @assert model == alg_result.model

    # stopping rule
    length(history) == max_step && return model, Ω_z
    M = model.M
    minℓ = criterion(𝐗[:, M], 𝚺⁻¹, test_stat)
    ## delete
    for j ∈ M
        Ω_z = Ω_z ∩ (minℓ ≤ criterion(𝐗[:, setdiff(M, j)], 𝚺⁻¹, test_stat))
    end

    return model, Ω_z
end
function region(𝐗, 𝐲_z, 𝚺, test_stat::TestStatistic, alg::StepwiseFeatureSelection) 
    @unpack direction, criterion, max_step = alg
    region(𝐗, 𝐲_z, 𝚺, test_stat, direction, criterion; max_step=max_step)
end # wrapper of `region`

function compute_solution_path(𝐗, 𝚺, test_stat::TestStatistic, alg::AbstractAlgorithm;
    z_min=typemin(Int), z_max=typemax(Int))
    𝐚, 𝐛 = test_stat.𝐚, test_stat.𝐛
    z = float(z_min)
    𝐳 = [z]
    𝐌 = Model[]
    while z < z_max
        𝐲_z = 𝐚 + 𝐛 * z
        model, Ω_z = region(𝐗, 𝐲_z, 𝚺, test_stat, alg)
        for I_z ∈ Ω_z
            z ∈ I_z && (z = I_z.right + ε; break)
        end
        z < z_max ? push!(𝐳, z - ε) : push!(𝐳, z_max)
        push!(𝐌, model)
    end
    return 𝐳, 𝐌
end

function truncated_interval(model_obs, params...; kwargs...)
    𝐳, 𝐌 = compute_solution_path(params...; kwargs...)
        Z = ClosedInterval{eltype(𝐳)}[]
    for i in eachindex(𝐌)
        model_obs == 𝐌[i] && push!(Z, (𝐳[i]..(𝐳[i + 1])))
    end
    return Z
end

function selective_p(test_prob::SelectiveHypothesisTest)
    @unpack alg, alg_result, test_stats, 𝐗, 𝐲_obs, 𝚺 = test_prob
    model_obs = alg_result.model
    function p_value(test_stat::TestStatistic)
        z_obs, σ²_z = test_stat.z_obs, test_stat.σ²_z

        # compute Z = {z | selection_event}
        Z = truncated_interval(model_obs, 𝐗, 𝚺, test_stat, alg;
            z_min=-abs(z_obs) - 10 * √(σ²_z), z_max=abs(z_obs) + 10 * √(σ²_z))

        𝜋 = cdf(TruncatedDistribution(Normal(0, √(σ²_z)), Z), z_obs)
        return 2 * min(𝜋, 1 - 𝜋)
    end
    p_values = p_value.(test_stats)
    return p_values
end
    
function selective_CI(test_prob::SelectiveHypothesisTest; α=0.05)
    @unpack alg, alg_result, test_stats, 𝐗, 𝐲_obs, 𝚺 = test_prob
    model_obs = alg_result.model
    function CI(test_stat::TestStatistic)
        z_obs, σ²_z = test_stat.z_obs, test_stat.σ²_z

        # compute Z = {z | selection_event}
        Z = truncated_interval(model_obs, 𝐗, 𝚺, test_stat, alg;
            z_min=-abs(z_obs) - 10 * √(σ²_z), z_max=abs(z_obs) + 10 * √(σ²_z)) 

        return confidence_interval(z_obs, √(σ²_z), Z; α=α)
    end
    CIs = CI.(test_stats)
    return CIs
end

end # module