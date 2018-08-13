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
    C = calculate_C(model, strategy.C_strategy, gamma)
    return (C, MLKernels.GaussianKernel(gamma))
end
