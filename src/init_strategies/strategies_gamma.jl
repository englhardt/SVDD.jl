abstract type InitializationStrategyGamma <: InitializationStrategy end

struct FixedGammaStrategy <: InitializationStrategyGamma
    kernel::SquaredExponentialKernel
end

calculate_gamma(model, strategy::FixedGammaStrategy) = MLKernels.getvalue(strategy.kernel.alpha)

"""
Original publication:
Silverman, Bernard W. Density estimation for statistics and data analysis. Routledge, 2018.
"""
struct RuleOfThumbSilverman <: InitializationStrategyGamma end

function calculate_gamma(model, strategy::RuleOfThumbSilverman)
    return (size(model.data, 2) * (size(model.data, 1) + 2) / 4.0)^(-1.0 / (size(model.data,1) + 4.0))
end

"""
Original publication:
Scott, David W. Multivariate density estimation: theory, practice, and visualization. John Wiley & Sons, 2015.
"""
struct RuleOfThumbScott <: InitializationStrategyGamma end

function calculate_gamma(model, strategy::RuleOfThumbScott)
    return size(model.data, 2)^(-1.0/(size(model.data,1) + 4))
end
