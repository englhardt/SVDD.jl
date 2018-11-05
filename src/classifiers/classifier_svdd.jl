
function fit!(model::SVDDClassifier, solver)
    debug(LOGGER, "[FIT] Fitting $(typeof(model)).")
    model.state == model_created && throw(ModelStateException(model.state, model_initialized))
    status = solve!(model, solver)
    model.R, model.const_term = get_R_and_const_term(model)
    model.state = model_fitted
    debug(LOGGER, "[FIT] $(typeof(model)) is now in state $(model.state).")
    return status
end

function predict(model::SVDDClassifier, target::Array{T,2}) where T <: Real
    model.state == model_fitted || throw(ModelStateException(model.state, model_fitted))
    @assert size(model.data, 1) == size(target, 1) "Dimension mismatch between traning data and target.
        target must be a column array, e.g., transpose([1.0 2.0])"

    α = get_alpha_prime(model)
    function predict_observation(z)
        kernel(model.kernel_fct, z, z) -
            2 * sum(α[i] * kernel(model.kernel_fct, model.data[:,i], z) for i in eachindex(α)) +
            model.const_term
    end
    return vec(sqrt.(mapslices(predict_observation, target, dims=1)) .- model.R)
end

function get_R_and_const_term(model::SVDDClassifier)
    @assert length(model.alpha_values) == size(model.K, 1)

    sv_idx = get_support_vectors(model)
    # take the maximum R over all support vectors for numerical reasons
    α = get_alpha_prime(model)

    const_term = sum(α[i] * α[j]*model.K[i,j] for i in eachindex(α) for j in eachindex(α))
    R_squared = 0.0
    tmp = 0.0
    for k in sv_idx
        tmp = model.K[k,k] - 2 * sum(α[i] * model.K[i,k] for i in eachindex(α)) + const_term
        R_squared = tmp > R_squared ? tmp : R_squared
    end
    return (sqrt(R_squared), const_term)
end

function invalidate_solution!(model::SVDDClassifier)
    model.alpha_values = []
    model.const_term = -Inf
    model.R = -Inf
    model.state = model_initialized
    return nothing
end
