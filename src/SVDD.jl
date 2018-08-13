module SVDD

include("svdd_base.jl")
include("svdd_util.jl")

include("classifiers/classifier.jl")
include("classifiers/classifier_svdd.jl")
include("classifiers/classifier_svdd_vanilla.jl")
include("classifiers/classifier_svdd_neg.jl")
include("classifiers/classifier_ssad.jl")
include("classifiers/classifier_randomOC.jl")

include("init_strategies/strategies_C.jl")
include("init_strategies/strategies_gamma.jl")
include("init_strategies/strategies_combined.jl")

using Memento
using Compat: @__MODULE__

const LOGGER = getlogger(@__MODULE__)

function __init__()
    Memento.register(LOGGER)
end

export
    OCClassifier,
    VanillaSVDD,
    SVDDneg,
    SSAD,

    ModelSolverException, ModelStateException,

    FixedParameterInitialization,
    FixedGammaStrategy,
    FixedCStrategy,
    SimpleCombinedStrategy,
    GammaFirstCombinedStrategy,
    BinarySearchCStrategy,
    TaxErrorEstimate,
    BoundedTaxErrorEstimate,
    RuleOfThumbSilverman,
    RuleOfThumbScott,

    initialize!,
    fit!,
    predict,
    classify,

    instantiate,
    set_adjust_K!,
    set_data!,
    set_pools!,
    set_kappa!,
    merge_pools,
    get_kernel,
    get_model_params,

    adjust_kernel_matrix!
end
