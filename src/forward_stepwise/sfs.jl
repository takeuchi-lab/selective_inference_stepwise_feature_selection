module SFS

export sfs, sfs_TVS, sfs_CV

include("../data.jl")
using .DataUtil
using Statistics, LinearAlgebra
using InvertedIndices

# forward stepwise feature selection with step size k
function sfs(๐, ๐ฒ, k)
    @assert k โค size(๐, 2)

    ๐ด = Array{Int}(undef, k)
    ๐ฌ = Array{Float64}(undef, k)
    for t = 1:k
        ๐ด_t = ๐ด[1:(t - 1)]
        ๐ดแถ_t = collect(1:size(๐, 2))[Not(๐ด_t)]
        ๐_๐ด = ๐[:, ๐ด_t]
        ๐ซ = ๐ฒ - ๐_๐ด / (๐_๐ด' * ๐_๐ด) * (๐_๐ด' * ๐ฒ)
        j_t = ๐ดแถ_t[argmax(abs.(๐[:, ๐ดแถ_t]' * ๐ซ))] # new feature
        ๐ด[t] = j_t
        ๐ฌ[t] = float(sign(๐[:,j_t]' * ๐ซ))
    end
    return (๐ด, ๐ฌ)
end

# train-val-split for determining the hyperparameter "k" in candidate list K
function sfs_TVS(๐, ๐ฒ, K; ratio=0.5) 
    ๐_train, ๐_val = train_val_split(๐; ratio=ratio, do_centering=true)
    ๐ฒ_train, ๐ฒ_val = train_val_split(๐ฒ; ratio=ratio, do_centering=true)
    k = K[argmin((k -> validation_error(k, ๐_train, ๐_val, ๐ฒ_train, ๐ฒ_val)).(K))]
    return (sfs(๐, ๐ฒ, k)..., k)
end

# cross-validation for determining the hyperparameter "k" in candidate list K
function sfs_CV(๐, ๐ฒ, K; ๐=length(๐ฒ)) 
    k = K[argmin((k -> cv_error(k, ๐, ๐ฒ, ๐)).(K))]
    return (sfs(๐, ๐ฒ, k)..., k)
end

function validation_error(k, ๐_train, ๐_val, ๐ฒ_train, ๐ฒ_val)
    ๐ด_k, _ = sfs(๐_train, ๐ฒ_train, k)
    ๐_train_๐ด = ๐_train[:, ๐ด_k] # active train input
    ๐_val_๐ด = ๐_val[:, ๐ด_k] # active validation input
    ๐โ = (๐_train_๐ด' * ๐_train_๐ด) \ ๐_train_๐ด' # pseudo inverse matrix of ๐_train_๐ด
    ๐ = ๐ฒ_val - ๐_val_๐ด * ๐โ * ๐ฒ_train
    return ๐' * ๐
end

function cv_error(k, ๐, ๐ฒ, ๐)
    ๐_fold = KFold(length(๐ฒ), ๐) # ๐-fold Cross Validation
    return mean(
        begin
        ๐_train, ๐_val = train_val_split(๐, ๐_fold.parts[i], ๐_fold.parts[i + 1] - 1)
        ๐ฒ_train, ๐ฒ_val = train_val_split(๐ฒ, ๐_fold.parts[i], ๐_fold.parts[i + 1] - 1)
        validation_error(k, ๐_train, ๐_val, ๐ฒ_train, ๐ฒ_val)
    end for i = 1:๐)
end

end # module SFS