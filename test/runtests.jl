using SVDD
using JuMP, Ipopt
using StatsBase, Distributions
using MLKernels, MLLabelUtils
using Test
using LinearAlgebra, Random

TEST_SOLVER =  with_optimizer(Ipopt.Optimizer, print_level=0)

include("test_utils.jl")

@testset "SVDD" begin
    include("svdd_util_test.jl")

    include("classifiers/classifier_test.jl")
    include("classifiers/classifier_ssad_test.jl")
    include("classifiers/classifier_svdd_test.jl")
    include("classifiers/classifier_svdd_neg_test.jl")
    include("classifiers/classifier_svdd_vanilla_test.jl")

    include("init_strategies/init_strategies_test.jl")
end
