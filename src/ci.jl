module ConfidenceInterval

export confidence_interval

using IntervalSets, Distributions
include("selective_p.jl")
using .MultiTruncatedDistributions
using Roots

const ε = 1e-4

function F⁻¹(p, σ, Z::AbstractArray{<:ClosedInterval}, z_obs) # return μ : ℱ_{μ,σ}^{Z}(z_obs) = p
    f(μ) = cdf(TruncatedDistribution(Normal(μ, BigFloat(σ)), Z), z_obs)
    a, b = Z[begin].left, Z[end].right
    fa, fb = f(a), f(b)
    @assert fa > fb # F is monotonically decreasing function

    # adjust a, b s.t. fa > p > fb
    while fb ≥ p
        extent = b - a
        b_tmp = b + extent
        fb_tmp = f(b_tmp)
        while isnan(fb_tmp) # if :fb_tmp is NaN, then :extent must be smaller
            extent /= 10 
            b_tmp = b + extent
            fb_tmp = f(b_tmp)
        end
        b = b_tmp
        fb = fb_tmp
    end
    while p ≥ fa
        extent = b - a
        a_tmp = a - extent
        fa_tmp = f(a_tmp)
        while isnan(fa_tmp) # if :fa_tmp is NaN, then :extent must be smaller
            extent /= 10
            a_tmp = a - extent
            fa_tmp = f(a_tmp)
        end
        a = a_tmp
        fa = fa_tmp
    end
    @assert fa > p > fb

    find_zero(μ -> f(μ) - p, (a, b))
end

F⁻¹(p, σ, Z::ClosedInterval, z_obs) = F⁻¹(p, σ, [Z], z_obs)

function confidence_interval(z_obs, σ, Z; α=0.05)
    l = F⁻¹(1 - α / 2, σ, Z, z_obs) # CCDF = α/2
    u = F⁻¹(α / 2, σ, Z, z_obs) # CDF = α/2
    return l..u
end

end # module ConfidenceInterval