module MultiTruncatedDistributions

using Distributions, IntervalSets
export TruncatedDistribution, cdf

prob(d::UnivariateDistribution, range::AbstractInterval) = cdf(d, range.left) - cdf(d, range.right)

struct TruncatedDistribution{S,T}
    d::S # distribution
    Z::T # truncated region
end

function Distributions.cdf(td::TruncatedDistribution, x)
    d, Z = td.d, td.Z

    length(Z) > 1 && begin
        function modify(Z)
            Z_new = eltype(Z)[Z[begin]]
            for I in Z[2:end]
                if abs(Z_new[end].right - I.left) < 1e-2 * std(d)
                    Z_new[end] = (Z_new[end].left)..(I.right)
                else
                    push!(Z_new, I)
                end
            end
            return Z_new
        end
        Z = modify(Z)
    end

    numerator = zero(prob(d, x..x)); denominator = numerator
    for I in Z
        denominator += prob(d, I)
        if I.right ≤ x
            numerator += prob(d, I)
        elseif x ∈ I
            numerator += prob(d, (I.left)..x)
        end
    end
    return numerator / denominator
end

end # module MultiTruncatedDistributions