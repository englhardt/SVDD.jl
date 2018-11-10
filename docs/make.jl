using Documenter
using SVDD

makedocs(
    sitename = "SVDD Documentation",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "start.md",
        "Custom Solvers" => [
            "SMO" => "smo.md"
            ]
        ],
    format = :html,
    modules = [SVDD]
)
deploydocs(
    repo = "github.com/englhardt/SVDD.jl.git"
)
