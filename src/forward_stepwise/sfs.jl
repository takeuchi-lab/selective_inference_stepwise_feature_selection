module SFS

export sfs, sfs_TVS, sfs_CV

include("../data.jl")
using .DataUtil
using Statistics, LinearAlgebra
using InvertedIndices

# forward stepwise feature selection with step size k
function sfs(𝐗, 𝐲, k)
    @assert k ≤ size(𝐗, 2)

    𝐴 = Array{Int}(undef, k)
    𝐬 = Array{Float64}(undef, k)
    for t = 1:k
        𝐴_t = 𝐴[1:(t - 1)]
        𝐴ᶜ_t = collect(1:size(𝐗, 2))[Not(𝐴_t)]
        𝐗_𝐴 = 𝐗[:, 𝐴_t]
        𝐫 = 𝐲 - 𝐗_𝐴 / (𝐗_𝐴' * 𝐗_𝐴) * (𝐗_𝐴' * 𝐲)
        j_t = 𝐴ᶜ_t[argmax(abs.(𝐗[:, 𝐴ᶜ_t]' * 𝐫))] # new feature
        𝐴[t] = j_t
        𝐬[t] = float(sign(𝐗[:,j_t]' * 𝐫))
    end
    return (𝐴, 𝐬)
end

# train-val-split for determining the hyperparameter "k" in candidate list K
function sfs_TVS(𝐗, 𝐲, K; ratio=0.5) 
    𝐗_train, 𝐗_val = train_val_split(𝐗; ratio=ratio, do_centering=true)
    𝐲_train, 𝐲_val = train_val_split(𝐲; ratio=ratio, do_centering=true)
    k = K[argmin((k -> validation_error(k, 𝐗_train, 𝐗_val, 𝐲_train, 𝐲_val)).(K))]
    return (sfs(𝐗, 𝐲, k)..., k)
end

# cross-validation for determining the hyperparameter "k" in candidate list K
function sfs_CV(𝐗, 𝐲, K; 𝑘=length(𝐲)) 
    k = K[argmin((k -> cv_error(k, 𝐗, 𝐲, 𝑘)).(K))]
    return (sfs(𝐗, 𝐲, k)..., k)
end

function validation_error(k, 𝐗_train, 𝐗_val, 𝐲_train, 𝐲_val)
    𝐴_k, _ = sfs(𝐗_train, 𝐲_train, k)
    𝐗_train_𝐴 = 𝐗_train[:, 𝐴_k] # active train input
    𝐗_val_𝐴 = 𝐗_val[:, 𝐴_k] # active validation input
    𝐗₊ = (𝐗_train_𝐴' * 𝐗_train_𝐴) \ 𝐗_train_𝐴' # pseudo inverse matrix of 𝐗_train_𝐴
    𝐞 = 𝐲_val - 𝐗_val_𝐴 * 𝐗₊ * 𝐲_train
    return 𝐞' * 𝐞
end

function cv_error(k, 𝐗, 𝐲, 𝑘)
    𝑘_fold = KFold(length(𝐲), 𝑘) # 𝑘-fold Cross Validation
    return mean(
        begin
        𝐗_train, 𝐗_val = train_val_split(𝐗, 𝑘_fold.parts[i], 𝑘_fold.parts[i + 1] - 1)
        𝐲_train, 𝐲_val = train_val_split(𝐲, 𝑘_fold.parts[i], 𝑘_fold.parts[i + 1] - 1)
        validation_error(k, 𝐗_train, 𝐗_val, 𝐲_train, 𝐲_val)
    end for i = 1:𝑘)
end

end # module SFS