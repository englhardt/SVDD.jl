abstract type InitializationStrategyC <: InitializationStrategy end

struct FixedCStrategy <: InitializationStrategyC
    C
end

calculate_C(model, strategy::FixedCStrategy) = strategy.C

"""
Original publication:
Tax, David MJ, and Robert PW Duin. "Support vector data description." Machine learning 54.1 (2004): 45-66.
"""
struct TaxErrorEstimate <: InitializationStrategyC
    target_outlier_percentage::Float64
end

calculate_C(model, strategy::TaxErrorEstimate) = 1 / (size(model.data, 2) * strategy.target_outlier_percentage)

struct BoundedTaxErrorEstimate <: InitializationStrategyC
    target_outlier_percentage::Float64
    lower_bound::Float64
    upper_bound::Float64
end

function calculate_C(model, strategy::BoundedTaxErrorEstimate)
    C = calculate_C(model, TaxErrorEstimate(strategy.target_outlier_percentage))
    return min(max(strategy.lower_bound, C), strategy.upper_bound)
end

struct BinarySearchCStrategy <: InitializationStrategyC
    target_outlier_percentage
    C_init
    max_iter
    eps
    solver
end

BinarySearchCStrategy(target_outlier_percentage, solver) = BinarySearchCStrategy(target_outlier_percentage, 0.5, 5, 0.01, solver)

function calculate_C(model, strategy::BinarySearchCStrategy, kernel)
    iteration = 1
    C_current = strategy.C_init
    C_min = 0
    C_max = 1
    debug(LOGGER, "[BINARY_SEARCH] Searching for parameter C with $(typeof(strategy)) with kernel = $(kernel).")
    debug(LOGGER, "[BINARY_SEARCH] Search parameters: C_min = $(C_min), C_max = $(C_max), max_iter = $(strategy.max_iter).")

    m = deepcopy(model)
    init_strategy = SVDD.FixedParameterInitialization(kernel, C_current)
    SVDD.initialize!(m, init_strategy)

    while true
        SVDD.fit!(m, strategy.solver)
        classification = countmap(SVDD.classify.(SVDD.predict(m, m.data)))
        current_outlier_percentage = haskey(classification, :outlier) ? classification[:outlier] / size(m.data,2) : 0.0

        if abs(current_outlier_percentage - strategy.target_outlier_percentage) < strategy.eps || iteration > strategy.max_iter
            debug(LOGGER, "[BINARY_SEARCH] Search result: C = $(C_current)")
            return C_current
        end

        if current_outlier_percentage > strategy.target_outlier_percentage
            C_min = C_current
            C_current = C_current + (C_max - C_current) / 2
        else
            C_max = C_current
            C_current = C_current - (C_current - C_min) / 2
        end
        SVDD.set_C!(m, C_current)
        iteration +=1
    end
end
