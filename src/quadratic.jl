module Quadratic

export quadratic, +, -, *, /, ≤

using IntervalSets
using Base: +, ≤

struct quadratic{S,T,U}
    a::S
    b::T
    c::U
end

(f::quadratic)(x) = f.a * x^2 + f.b * x + f.c

Base.:+(f::quadratic, g::quadratic) = quadratic(f.a + g.a, f.b + g.b, f.c + g.c)
Base.:-(f::quadratic, g::quadratic) = quadratic(f.a - g.a, f.b - g.b, f.c - g.c)
Base.:*(f::quadratic, c) = quadratic(f.a * c, f.b * c, f.c * c)
Base.:/(f::quadratic, c) = quadratic(f.a / c, f.b / c, f.c / c)

# return: region that f is less than g
function Base.:≤(f::quadratic, g::quadratic)
    # coefs of h ≔ f - g
    a = f.a - g.a
    b = f.b - g.b
    c = f.c - g.c

    # if h is linear or constant
    if iszero(a)
        iszero(b) && return c ≤ 0 ? [-Inf..Inf] : ClosedInterval{Float64}[]
        return b > 0 ? [-Inf..(-c / b)] : [(-c / b)..Inf]
    end

    # if h is quadratic
    D = b^2 - 4 * a * c
    if D > 0
        x₁, x₂ = (-b + √D) / (2a), (-b - √D) / (2a)
        if a > 0
            @assert x₁ > x₂
            return [x₂..x₁]
        else
            @assert x₁ < x₂
            return [-Inf..x₁, x₂..Inf]
        end
    else
        return a < 0 ? [-Inf..Inf] : ClosedInterval{Float64}[]
    end
end

end # module Quadratic

# # test
# using Plots
# f = quadratic(2, -2, 1)
# g = quadratic(-4, 5, 100)
# xs = -10:0.1:10
# plot(xs, f.(xs), label="f", c="blue", legend=:inline)
# plot!(xs, g.(xs), label="g", c="green", legend=:inline)
# region = f ≤ g
# for interval in region
#     xs = range(max(interval.left, -10), min(interval.right, 10); length=100)
#     plot!(xs, f.(xs), label=nothing, color="red", lw=3) |> display
# end
