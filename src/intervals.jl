module Intersection

export intersect

using IntervalSets

# A, B is disjoint sets
function IntervalSets.intersect(A::AbstractArray{<:AbstractInterval}, B::AbstractArray{<:AbstractInterval})
    # check order
    @assert all(A[i].right ≤ A[i + 1].left for i in 1:(length(A) - 1))
    @assert all(B[i].right ≤ B[i + 1].left for i in 1:(length(B) - 1))

    return filter(I -> !isempty(I), reshape(A, (1, length(A))) .∩ B)
end

# I is an interval, and A is disjoint sets
function IntervalSets.intersect(I::AbstractInterval, A::AbstractArray{<:AbstractInterval})
    # check order
    @assert all(A[i].right ≤ A[i + 1].left for i in 1:(length(A) - 1))

    return filter(I -> !isempty(I), Ref(I) .∩ A)
end
IntervalSets.intersect(A::AbstractArray{<:AbstractInterval}, I::AbstractInterval) =  I ∩ A

end # module Intersection