module DataUtil

export DataParam, make_dataset, centering, train_val_split, KFold

using Random, Statistics, Distributions
using LinearAlgebra
using InvertedIndices
using Parameters

@with_kw struct DataParam{A,B,C,D,E,F}
    N::A
    β₀::B
    𝛃::C
    𝚺::D = I
    𝛍ₓ::E = zeros(length(𝛃)); @assert length(𝛍ₓ) == length(𝛃)
    𝚺ₓ::F = I
end

# make_dataset
function make_dataset(β₀, 𝛃, N)
    d = length(𝛃)
    𝐗 = randn(N, d)
    𝐲 = β₀ .+ 𝐗 * 𝛃 .+ randn(N)
    return (𝐗, 𝐲)
end

function make_dataset(param::DataParam)
    @unpack N, β₀, 𝛃, 𝚺, 𝛍ₓ, 𝚺ₓ = param
    𝐗 = rand(MvNormal(𝛍ₓ, 𝚺ₓ), N) |> transpose
    𝐲 = rand(MvNormal(β₀ .+ 𝐗 * 𝛃, 𝚺))
    return (𝐗, 𝐲)
end

# centering
function centering(𝐗::AbstractMatrix)
    X̄ = mean(𝐗, dims=1)
    return 𝐗 .- X̄
end

function centering(𝐲::AbstractVector)
    ȳ = mean(𝐲)
    return 𝐲 .- ȳ
end

centering(𝐗, 𝐲) = centering(𝐗), centering(𝐲)

# train_val_split
function train_val_split(𝐗::AbstractMatrix, val_begin, val_end; do_centering=true)
    𝐗_train, 𝐗_val = @views 𝐗[Not(val_begin:val_end), :], 𝐗[val_begin:val_end, :]
    return do_centering ? centering(𝐗_train, 𝐗_val) : (𝐗_train, 𝐗_val)
end

function train_val_split(𝐲::AbstractVector, val_begin, val_end; do_centering=true)
    𝐲_train, 𝐲_val = @views 𝐲[Not(val_begin:val_end)], 𝐲[val_begin:val_end]
    return do_centering ? centering(𝐲_train, 𝐲_val) : (𝐲_train, 𝐲_val)
end

function train_val_split(𝐗::AbstractMatrix; ratio=0.5, do_centering=true)
    N = size(𝐗, 1)
    train_size = round(Int, N * ratio)
    return train_val_split(𝐗, train_size + 1, N; do_centering=do_centering)
end

function train_val_split(𝐲::AbstractVector; ratio=0.5, do_centering=true)
    N = length(𝐲)
    train_size = round(Int, N * ratio)
    return train_val_split(𝐲, train_size + 1, N; do_centering=do_centering)
end

# KFold
struct KFold{S,T <: AbstractArray}
    𝑘::S # 𝑘-fold
    parts::T # partition index list (the size is 𝑘+1, parts[begin]=1, parts[end]=N)
end

KFold(N::Integer, 𝑘::Integer) = KFold(𝑘, [min(1 + i * ceil(Int, N / 𝑘), N + 1) for i = 0:𝑘])

end # module DataUtil