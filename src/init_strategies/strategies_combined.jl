abstract type InitializationStrategyCombined <: InitializationStrategy end

struct SimpleCombinedStrategy <: InitializationStrategyCombined
    gamma_strategy::InitializationStrategyGamma
    C_strategy::InitializationStrategyC
end

FixedParameterInitialization(kernel, C) = SimpleCombinedStrategy(FixedGammaStrategy(kernel), FixedCStrategy(C))

function get_parameters(model, strategy::SimpleCombinedStrategy)
    C = calculate_C(model, strategy.C_strategy)
    gamma = calculate_gamma(model, strategy.gamma_strategy)
    return (C, MLKernels.GaussianKernel(gamma))
end

struct GammaFirstCombinedStrategy <: InitializationStrategyCombined
    gamma_strategy::InitializationStrategyGamma
    C_strategy::InitializationStrategyC
end

function get_parameters(model, strategy::GammaFirstCombinedStrategy)
    gamma = calculate_gamma(model, strategy.gamma_strategy)
    C = calculate_C(model, strategy.C_strategy, MLKernels.GaussianKernel(gamma))
    return (C, MLKernels.GaussianKernel(gamma))
end

struct SimpleSubspaceStrategy <: InitializationStrategyCombined
    gamma_strategy::InitializationStrategyGamma
    C_strategy::InitializationStrategyC
    gamma_scope::Scope

    function SimpleSubspaceStrategy(gamma_strategy, C_strategy; gamma_scope)
        new(gamma_strategy, C_strategy, gamma_scope)
    end
end

function get_parameters(model, strategy::SimpleSubspaceStrategy)
    C = calculate_C(model, strategy.C_strategy)

    if isa(strategy.gamma_scope, Val{:Subspace})
        gamma = @eachsubspace calculate_gamma(model, strategy.gamma_strategy)
    else
        tmp = calculate_gamma(model, strategy.gamma_strategy)
        gamma = fill(tmp, length(model.subspaces))
    end
    return (C, MLKernels.GaussianKernel.(gamma))
end

"""
    Determines C for WangGammaStrategy by using a provided C init strategy
"""
struct WangCombinedInitializationStrategy <: InitializationStrategyCombined
    solver::JuMP.OptimizerFactory
    gamma_search_range
    C_strategy::InitializationStrategyC
end

WangCombinedInitializationStrategy(solver, C_strategy) = WangCombinedInitializationStrategy(solver, 10.0.^range(-2, stop=2, length=50), C_strategy)

function get_parameters(model, strategy::WangCombinedInitializationStrategy)
    C = calculate_C(model, strategy.C_strategy)
    wang_strategy = WangGammaStrategy(strategy.solver, strategy.gamma_search_range, C)
    gamma = calculate_gamma(model, wang_strategy)
    return (C, MLKernels.GaussianKernel(gamma))
end
