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

const ฮต = 1e-4

struct TestStatistic{A,B,C,D}
    z_obs::A
    ฯยฒ_z::B
    ๐::C
    ๐::D
end
function parametric_representation(๐, ๐ฒ, ๐บ)
    z = ๐' * ๐ฒ
    ฯยฒ_z = ๐' * ๐บ * ๐ 
    ๐ = ๐บ * ๐ / ฯยฒ_z
    ๐ = ๐ฒ - ๐ * ๐' * ๐ฒ
    @assert ๐ + ๐ * z โ ๐ฒ
    return TestStatistic(z, ฯยฒ_z, ๐, ๐)
end
"Make the hypothesis test problem: ฮฒโฑผ = 0 or not for all j โ `alg_result.model.M`, where ฮฒโฑผ is the coefficient of the j-th feature."
struct SelectiveHypothesisTest{A <: AbstractAlgorithm,B <: AbstractAlgResult,C,D,E,F}
    alg::A # used feature selection algorithm
    alg_result::B # the result of the algorithm `alg` for `๐`, `๐ฒ_obs`, `๐บ`
    test_stats::C # the information of test statistic for all hypotheses
    ๐::D # used covariate matrix
    ๐ฒ_obs::E # used response vector
    ๐บ::F # used covariance matrix for response vector

    function SelectiveHypothesisTest(alg::S, alg_result::T, ๐::U, ๐ฒ_obs::V, ๐บ::W) where {S,T,U,V,W}
        M_obs = alg_result.model.M
        ๐_obs = ๐[:, M_obs]
        ๐ฎ = ๐_obs / (๐_obs' * ๐_obs)
        test_stats = [parametric_representation(๐, ๐ฒ_obs, ๐บ) for ๐ in eachcol(๐ฎ)]
        new{S,T,typeof(test_stats),U,V,W}(alg, alg_result, test_stats, ๐, ๐ฒ_obs, ๐บ)
    end
end

function (::AIC)(๐, ๐บโปยน, test_stat::TestStatistic)
    ๐, ๐ = test_stat.๐, test_stat.๐
    ๐ = ๐บโปยน - ๐บโปยน * ๐ * pinv(๐' * ๐บโปยน * ๐) * ๐' * ๐บโปยน
    return quadratic(๐' * ๐ * ๐, 2 * (๐' * ๐ * ๐), ๐' * ๐ * ๐ + 2 * size(๐, 2))
end
function (::BIC)(๐, ๐บโปยน, test_stat::TestStatistic)
    ๐, ๐ = test_stat.๐, test_stat.๐
    ๐ = ๐บโปยน - ๐บโปยน * ๐ * pinv(๐' * ๐บโปยน * ๐) * ๐' * ๐บโปยน
    return quadratic(๐' * ๐ * ๐, 2 * (๐' * ๐ * ๐), ๐' * ๐ * ๐ + log(size(๐, 1)) * size(๐, 2))
end

# compute the region correspond to the selected model at `๐ฒ_z`
function region(๐, ๐ฒ_z, ๐บ,
    test_stat::TestStatistic,
    direction::Bidirection,
    criterion::AbstractCriterion;
    max_step=typemax(Int))
    alg_result = stepwise(๐, ๐ฒ_z, ๐บ, direction, criterion; max_step=max_step)
    history = alg_result.history

    d = size(๐, 2)
    model = Model(direction, d)
    ๐บโปยน = I / ๐บ
    ฮฉ_z = [(-Inf)..Inf]

    for i in eachindex(history)
        @unpack M, Mแถ = model
        # i-th selected feature
        idx = history[i]
        ๐แตข = idx > 0 ? ๐[:, union(M, idx)] : ๐[:, setdiff(M, -idx)]
        โแตข = criterion(๐แตข, ๐บโปยน, test_stat)

        # โแตข is smaller than the value of previous model M
        i != 1 && (ฮฉ_z = ฮฉ_z โฉ (โแตข โค criterion(๐[:, M], ๐บโปยน, test_stat)))
            # delete
        for j โ M
            ฮฉ_z = ฮฉ_z โฉ (โแตข โค criterion(๐[:, setdiff(M, j)], ๐บโปยน, test_stat))
        end
        # add
        for jโฒ โ Mแถ
            ฮฉ_z = ฮฉ_z โฉ (โแตข โค criterion(๐[:, union(M, jโฒ)], ๐บโปยน, test_stat))
        end

        update!(model, idx)
    end; @assert model == alg_result.model

    # stopping rule
    length(history) == max_step && return model, ฮฉ_z
    @unpack M, Mแถ = model
    minโ = criterion(๐[:, M], ๐บโปยน, test_stat)
    ## delete
    for j โ M
        ฮฉ_z = ฮฉ_z โฉ (minโ โค criterion(๐[:, setdiff(M, j)], ๐บโปยน, test_stat))
    end
    ## add
    for jโฒ โ Mแถ
        ฮฉ_z = ฮฉ_z โฉ (minโ โค criterion(๐[:, union(M, jโฒ)], ๐บโปยน, test_stat))
    end

    return model, ฮฉ_z
end
function region(๐, ๐ฒ_z, ๐บ,
    test_stat::TestStatistic,
    direction::Forward,
    criterion::AbstractCriterion;
    max_step=typemax(Int))
    alg_result = stepwise(๐, ๐ฒ_z, ๐บ, direction, criterion; max_step=max_step)
    history = alg_result.history

    d = size(๐, 2)
    model = Model(direction, d)
    ๐บโปยน = I / ๐บ
    ฮฉ_z = [(-Inf)..Inf]
        
    for i in eachindex(history)
        @unpack M, Mแถ = model
        # i-th selected feature
        idx = history[i]
        ๐แตข = ๐[:, union(M, idx)]
        โแตข = criterion(๐แตข, ๐บโปยน, test_stat)

        # โแตข is smaller than the value of previous model M
        i != 1 && (ฮฉ_z = ฮฉ_z โฉ (โแตข โค criterion(๐[:, M], ๐บโปยน, test_stat)))
        # add
        for jโฒ โ Mแถ
            ฮฉ_z = ฮฉ_z โฉ (โแตข โค criterion(๐[:, union(M, jโฒ)], ๐บโปยน, test_stat))
        end

        update!(model, idx)
    end; @assert model == alg_result.model

    # stopping rule
    length(history) == max_step && return model, ฮฉ_z
    @unpack M, Mแถ = model
    minโ = criterion(๐[:, M], ๐บโปยน, test_stat)
    ## add
    for jโฒ โ Mแถ
        ฮฉ_z = ฮฉ_z โฉ (minโ โค criterion(๐[:, union(M, jโฒ)], ๐บโปยน, test_stat))
    end

    return model, ฮฉ_z
end
function region(๐, ๐ฒ_z, ๐บ,
    test_stat::TestStatistic,
    direction::Backward,
    criterion::AbstractCriterion;
    max_step=typemax(Int))
    alg_result = stepwise(๐, ๐ฒ_z, ๐บ, direction, criterion; max_step=max_step)
    history = alg_result.history

    d = size(๐, 2)
    model = Model(direction, d)
    ๐บโปยน = I / ๐บ
    ฮฉ_z = [(-Inf)..Inf]
        
    for i in eachindex(history)
        M = model.M
        # i-th selected feature
        idx = history[i]
        ๐แตข = ๐[:, setdiff(M, -idx)]
        โแตข = criterion(๐แตข, ๐บโปยน, test_stat)

        # โแตข is smaller than the value of previous model M
        i != 1 && (ฮฉ_z = ฮฉ_z โฉ (โแตข โค criterion(๐[:, M], ๐บโปยน, test_stat)))
            # delete
        for j โ M
            ฮฉ_z = ฮฉ_z โฉ (โแตข โค criterion(๐[:, setdiff(M, j)], ๐บโปยน, test_stat))
        end

        update!(model, idx)
    end; @assert model == alg_result.model

    # stopping rule
    length(history) == max_step && return model, ฮฉ_z
    M = model.M
    minโ = criterion(๐[:, M], ๐บโปยน, test_stat)
    ## delete
    for j โ M
        ฮฉ_z = ฮฉ_z โฉ (minโ โค criterion(๐[:, setdiff(M, j)], ๐บโปยน, test_stat))
    end

    return model, ฮฉ_z
end
function region(๐, ๐ฒ_z, ๐บ, test_stat::TestStatistic, alg::StepwiseFeatureSelection) 
    @unpack direction, criterion, max_step = alg
    region(๐, ๐ฒ_z, ๐บ, test_stat, direction, criterion; max_step=max_step)
end # wrapper of `region`

function compute_solution_path(๐, ๐บ, test_stat::TestStatistic, alg::AbstractAlgorithm;
    z_min=typemin(Int), z_max=typemax(Int))
    ๐, ๐ = test_stat.๐, test_stat.๐
    z = float(z_min)
    ๐ณ = [z]
    ๐ = Model[]
    while z < z_max
        ๐ฒ_z = ๐ + ๐ * z
        model, ฮฉ_z = region(๐, ๐ฒ_z, ๐บ, test_stat, alg)
        for I_z โ ฮฉ_z
            z โ I_z && (z = I_z.right + ฮต; break)
        end
        z < z_max ? push!(๐ณ, z - ฮต) : push!(๐ณ, z_max)
        push!(๐, model)
    end
    return ๐ณ, ๐
end

function truncated_interval(model_obs, params...; kwargs...)
    ๐ณ, ๐ = compute_solution_path(params...; kwargs...)
        Z = ClosedInterval{eltype(๐ณ)}[]
    for i in eachindex(๐)
        model_obs == ๐[i] && push!(Z, (๐ณ[i]..(๐ณ[i + 1])))
    end
    return Z
end

function selective_p(test_prob::SelectiveHypothesisTest)
    @unpack alg, alg_result, test_stats, ๐, ๐ฒ_obs, ๐บ = test_prob
    model_obs = alg_result.model
    function p_value(test_stat::TestStatistic)
        z_obs, ฯยฒ_z = test_stat.z_obs, test_stat.ฯยฒ_z

        # compute Z = {z | selection_event}
        Z = truncated_interval(model_obs, ๐, ๐บ, test_stat, alg;
            z_min=-abs(z_obs) - 10 * โ(ฯยฒ_z), z_max=abs(z_obs) + 10 * โ(ฯยฒ_z))

        ๐ = cdf(TruncatedDistribution(Normal(0, โ(ฯยฒ_z)), Z), z_obs)
        return 2 * min(๐, 1 - ๐)
    end
    p_values = p_value.(test_stats)
    return p_values
end
    
function selective_CI(test_prob::SelectiveHypothesisTest; ฮฑ=0.05)
    @unpack alg, alg_result, test_stats, ๐, ๐ฒ_obs, ๐บ = test_prob
    model_obs = alg_result.model
    function CI(test_stat::TestStatistic)
        z_obs, ฯยฒ_z = test_stat.z_obs, test_stat.ฯยฒ_z

        # compute Z = {z | selection_event}
        Z = truncated_interval(model_obs, ๐, ๐บ, test_stat, alg;
            z_min=-abs(z_obs) - 10 * โ(ฯยฒ_z), z_max=abs(z_obs) + 10 * โ(ฯยฒ_z)) 

        return confidence_interval(z_obs, โ(ฯยฒ_z), Z; ฮฑ=ฮฑ)
    end
    CIs = CI.(test_stats)
    return CIs
end

end # module