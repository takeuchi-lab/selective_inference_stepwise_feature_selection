module Stepwise

using Statistics, LinearAlgebra
using Parameters
using Base: ==

export AbstractDirection, Bidirection, Unidirection, ForwardBackward, BackwardForward, Forward, Backward,
    AbstractCriterion, AIC, BIC,
    AbstractAlgorithm, StepwiseFeatureSelection,
    AbstractAlgResult, SFSResult,
    Model, Null, Full, null_or_full, ==, update!,
    stepwise

abstract type AbstractDirection end
abstract type Bidirection <: AbstractDirection end
abstract type Unidirection <: AbstractDirection end
struct ForwardBackward <: Bidirection end
struct BackwardForward <: Bidirection end
struct Forward <: Unidirection end
struct Backward <: Unidirection end

abstract type AbstractCriterion end
struct AIC <: AbstractCriterion end
struct BIC <: AbstractCriterion end
(::AIC)(𝐗, 𝐲, 𝚺⁻¹) = 𝐲' * (𝚺⁻¹ - 𝚺⁻¹ * 𝐗 * pinv(𝐗' * 𝚺⁻¹ * 𝐗) * 𝐗' * 𝚺⁻¹) * 𝐲 + 2 * size(𝐗, 2)
(::BIC)(𝐗, 𝐲, 𝚺⁻¹) = 𝐲' * (𝚺⁻¹ - 𝚺⁻¹ * 𝐗 * pinv(𝐗' * 𝚺⁻¹ * 𝐗) * 𝐗' * 𝚺⁻¹) * 𝐲 + log(length(𝐲)) * size(𝐗, 2)

abstract type AbstractAlgorithm end
@with_kw struct StepwiseFeatureSelection{S <: AbstractDirection,T <: AbstractCriterion,U <: Integer} <: AbstractAlgorithm
    direction::S = ForwardBackward()
    criterion::T = AIC()
    max_step::U = typemax(Int)
end

# define "model"
struct Null end; struct Full end
null_or_full(::Forward) = Null(); null_or_full(::ForwardBackward) = Null()
null_or_full(::Backward) = Full(); null_or_full(::BackwardForward) = Full()
struct Model{T <: Integer}
    M::Vector{T} # the set of features in this model
    Mᶜ::Vector{T} # the set of features not included in this model
    Model{T}(::Null, dims::T) where {T} = new{T}(T[], collect(T, 1:dims))
    Model{T}(::Full, dims::T) where {T} = new{T}(collect(T, 1:dims), T[])
end
Base.:(==)(m::Model, m′::Model) = issetequal(m.M, m′.M) && issetequal(m.Mᶜ, m′.Mᶜ)
Model(direction::AbstractDirection, dims::T) where {T} = Model{T}(null_or_full(direction), dims)
## update model
function update!(model::Model, idx)
    @unpack M, Mᶜ = model
    if idx > 0
        @assert idx ∈ Mᶜ
        push!(M, idx); setdiff!(Mᶜ, idx)
    else
        @assert -idx ∈ M
        setdiff!(M, -idx); push!(Mᶜ, -idx)
    end
end

abstract type AbstractAlgResult end
struct SFSResult{S,T} <: AbstractAlgResult
    model::Model{S}
    history::T
end
function Base.show(io::IO, ::MIME"text/plain", result::SFSResult{S,T}) where {S,T}
    println(io, "SFSResult{$S,$T}")
    println(io, " selected features: ", result.model.M)
    println(io, " history of this algorithm: ", result.history)
end

# stepwise feature selection
function stepwise(𝐗, 𝐲, 𝚺, direction::Bidirection, criterion::AbstractCriterion; max_step=typemax(Int))
    𝚺⁻¹ = I / 𝚺
    d = size(𝐗, 2)
    model = Model(direction, d)
    history = typeof(d)[]
    L_prev = Inf

    for i = 1:max_step
        @unpack M, Mᶜ = model
        minL = Inf
        idx = 0
        # delete
        for j ∈ M
            L = criterion(𝐗[:, setdiff(M, j)], 𝐲, 𝚺⁻¹)
            L < minL && (minL = L; idx = -j)
        end
        # add
        for j′ ∈ Mᶜ
            L = criterion(𝐗[:, union(M, j′)], 𝐲, 𝚺⁻¹)
            L < minL && (minL = L; idx = +j′)
        end
        
        # if the criterion is not improved, then finish this algorithm
        L_prev ≤ minL && (return SFSResult(model, history)) 

        # update
        L_prev = minL
        push!(history, idx)
        update!(model, idx)
    end
    return SFSResult(model, history)
end
stepwise(𝐗, 𝐲_obs, 𝚺) = stepwise(𝐗, 𝐲_obs, 𝚺, ForwardBackward(), AIC()) # default
function stepwise(𝐗, 𝐲, 𝚺, direction::Forward, criterion::AbstractCriterion; max_step=typemax(Int))
    𝚺⁻¹ = I / 𝚺
    d = size(𝐗, 2)
    model = Model(direction, d)
    history = typeof(d)[]
    L_prev = Inf

    for i = 1:max_step
        @unpack M, Mᶜ = model
        minL = Inf
        idx = 0
        # add
        for j′ ∈ Mᶜ
            L = criterion(𝐗[:, union(M, j′)], 𝐲, 𝚺⁻¹)
            L < minL && (minL = L; idx = +j′)
        end
        
        # if the criterion is not improved, then finish this algorithm
        L_prev ≤ minL && (return SFSResult(model, history)) 

        # update
        L_prev = minL
        push!(history, idx)
        update!(model, idx)
    end
    return SFSResult(model, history)
end
function stepwise(𝐗, 𝐲, 𝚺, direction::Backward, criterion::AbstractCriterion; max_step=typemax(Int))
    𝚺⁻¹ = I / 𝚺
    d = size(𝐗, 2)
    model = Model(direction, d)
    history = typeof(d)[]
    L_prev = Inf

    for i = 1:max_step
        M = model.M
        minL = Inf
        idx = 0
        # delete
        for j ∈ M
            L = criterion(𝐗[:, setdiff(M, j)], 𝐲, 𝚺⁻¹)
            L < minL && (minL = L; idx = -j)
        end
        
        # if the criterion is not improved, then finish this algorithm
        L_prev ≤ minL && (return SFSResult(model, history)) 

        # update
        L_prev = minL
        push!(history, idx)
        update!(model, idx)
    end
    return SFSResult(model, history)
end
function stepwise(𝐗, 𝐲, 𝚺, alg::StepwiseFeatureSelection)
    @unpack direction, criterion, max_step = alg
    stepwise(𝐗, 𝐲, 𝚺, direction, criterion; max_step=max_step)
end # wrapper of stepwise

end # module Stepwise