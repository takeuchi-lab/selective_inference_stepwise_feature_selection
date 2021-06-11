using Pkg

function main()
    deps = [
        "Random",
        "Statistics",
        "Distributions",
        "LinearAlgebra",
        "InvertedIndices",
        "IntervalSets",
        "Parameters",
        "Roots",
        "Reexport",
    ]

    for pkg in deps
        Pkg.add(pkg)
    end
end