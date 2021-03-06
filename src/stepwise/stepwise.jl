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
(::AIC)(๐, ๐ฒ, ๐บโปยน) = ๐ฒ' * (๐บโปยน - ๐บโปยน * ๐ * pinv(๐' * ๐บโปยน * ๐) * ๐' * ๐บโปยน) * ๐ฒ + 2 * size(๐, 2)
(::BIC)(๐, ๐ฒ, ๐บโปยน) = ๐ฒ' * (๐บโปยน - ๐บโปยน * ๐ * pinv(๐' * ๐บโปยน * ๐) * ๐' * ๐บโปยน) * ๐ฒ + log(length(๐ฒ)) * size(๐, 2)

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
    Mแถ::Vector{T} # the set of features not included in this model
    Model{T}(::Null, dims::T) where {T} = new{T}(T[], collect(T, 1:dims))
    Model{T}(::Full, dims::T) where {T} = new{T}(collect(T, 1:dims), T[])
end
Base.:(==)(m::Model, mโฒ::Model) = issetequal(m.M, mโฒ.M) && issetequal(m.Mแถ, mโฒ.Mแถ)
Model(direction::AbstractDirection, dims::T) where {T} = Model{T}(null_or_full(direction), dims)
## update model
function update!(model::Model, idx)
    @unpack M, Mแถ = model
    if idx > 0
        @assert idx โ Mแถ
        push!(M, idx); setdiff!(Mแถ, idx)
    else
        @assert -idx โ M
        setdiff!(M, -idx); push!(Mแถ, -idx)
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
function stepwise(๐, ๐ฒ, ๐บ, direction::Bidirection, criterion::AbstractCriterion; max_step=typemax(Int))
    ๐บโปยน = I / ๐บ
    d = size(๐, 2)
    model = Model(direction, d)
    history = typeof(d)[]
    L_prev = Inf

    for i = 1:max_step
        @unpack M, Mแถ = model
        minL = Inf
        idx = 0
        # delete
        for j โ M
            L = criterion(๐[:, setdiff(M, j)], ๐ฒ, ๐บโปยน)
            L < minL && (minL = L; idx = -j)
        end
        # add
        for jโฒ โ Mแถ
            L = criterion(๐[:, union(M, jโฒ)], ๐ฒ, ๐บโปยน)
            L < minL && (minL = L; idx = +jโฒ)
        end
        
        # if the criterion is not improved, then finish this algorithm
        L_prev โค minL && (return SFSResult(model, history)) 

        # update
        L_prev = minL
        push!(history, idx)
        update!(model, idx)
    end
    return SFSResult(model, history)
end
stepwise(๐, ๐ฒ_obs, ๐บ) = stepwise(๐, ๐ฒ_obs, ๐บ, ForwardBackward(), AIC()) # default
function stepwise(๐, ๐ฒ, ๐บ, direction::Forward, criterion::AbstractCriterion; max_step=typemax(Int))
    ๐บโปยน = I / ๐บ
    d = size(๐, 2)
    model = Model(direction, d)
    history = typeof(d)[]
    L_prev = Inf

    for i = 1:max_step
        @unpack M, Mแถ = model
        minL = Inf
        idx = 0
        # add
        for jโฒ โ Mแถ
            L = criterion(๐[:, union(M, jโฒ)], ๐ฒ, ๐บโปยน)
            L < minL && (minL = L; idx = +jโฒ)
        end
        
        # if the criterion is not improved, then finish this algorithm
        L_prev โค minL && (return SFSResult(model, history)) 

        # update
        L_prev = minL
        push!(history, idx)
        update!(model, idx)
    end
    return SFSResult(model, history)
end
function stepwise(๐, ๐ฒ, ๐บ, direction::Backward, criterion::AbstractCriterion; max_step=typemax(Int))
    ๐บโปยน = I / ๐บ
    d = size(๐, 2)
    model = Model(direction, d)
    history = typeof(d)[]
    L_prev = Inf

    for i = 1:max_step
        M = model.M
        minL = Inf
        idx = 0
        # delete
        for j โ M
            L = criterion(๐[:, setdiff(M, j)], ๐ฒ, ๐บโปยน)
            L < minL && (minL = L; idx = -j)
        end
        
        # if the criterion is not improved, then finish this algorithm
        L_prev โค minL && (return SFSResult(model, history)) 

        # update
        L_prev = minL
        push!(history, idx)
        update!(model, idx)
    end
    return SFSResult(model, history)
end
function stepwise(๐, ๐ฒ, ๐บ, alg::StepwiseFeatureSelection)
    @unpack direction, criterion, max_step = alg
    stepwise(๐, ๐ฒ, ๐บ, direction, criterion; max_step=max_step)
end # wrapper of stepwise

end # module Stepwise