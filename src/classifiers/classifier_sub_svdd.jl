
mutable struct SubSVDD <: OCClassifier
    state::ModelState

    # model parameters
    C::Float64
    kernel_fct::Kernel
    subspaces::Vector{Vector{Int}}

    # training data
    data::Array{Float64,2}
    K::Array{Array{Float64,2}}
    adjust_K::Bool
    K_adjusted::Array{Array{Float64,2}}
    pools::Dict{Symbol, Array{Int64,1}}

    # fitted values
    alpha_values::Vector{Vector{Float64}}
    const_term::Vector{Float64}
    R::Vector{Float64}
    C_delta::Vector{Float64}

    function SubSVDD(data, subspaces, pools)
        m = new()
        m.C = 1.0
        m.state = model_created
        m.data = data
        m.adjust_K = false
        m.pools = labelmap(pools)
        m.subspaces = subspaces
        return m
    end
end

macro eachsubspace(expr)
    args = esc.(expr.args)
    sub_idx = :(eachindex($(args[2]).subspaces))
    return :( [$(args[1])($(args[2:end,]...), k) for k in $sub_idx] )
end

function invalidate_solution!(model::SubSVDD)
    model.alpha_values = Vector{Vector{Float64}}()
    model.const_term = Vector{Vector{Float64}}()
    model.R = Vector{Vector{Float64}}()
    model.state = model_initialized
    return nothing
end

function set_C!(model::SubSVDD, C::Real)
    model.C = C
    return nothing
end

function fit!(model::SubSVDD, solver)
    debug(LOGGER, "[FIT] Fitting $(typeof(model)).")
    model.state == model_created && throw(ModelStateException(model.state, model_initialized))
    status = solve!(model, solver)
    r_and_const_term = @eachsubspace get_R_and_const_term(model)
    model.R = map(x -> x[:R], r_and_const_term)
    model.const_term = map(x -> x[:const_term], r_and_const_term)
    model.state = model_fitted
    debug(LOGGER, "[FIT] $(typeof(model)) is now in state $(model.state).")
    return status
end

function solve!(model::SubSVDD, solver::JuMP.OptimizerFactory)
    debug(LOGGER, "[SOLVE] Setting up QP for SubSVDD with $(is_K_adjusted(model) ? "adjusted" : "non-adjusted") kernel matrix.")
    QP = Model(solver)
    K = is_K_adjusted(model) ? model.K_adjusted : model.K

    n = size(model.data, 2)

    @variable(QP, α[1:n, 1:length(K)] >= 0.0)
    @objective(QP, Max, sum(sum(α[i, k]*K[k][i,i] for i in 1:n) -
                            sum(α[i, k] * α[j, k] * K[k][i, j] for i in 1:n for j in 1:n)
                        for k in eachindex(model.subspaces)))
    for i in 1:n
        @constraint(QP, sum(α[i, k] for k in eachindex(model.subspaces)) <= model.C)
    end

    for k in eachindex(model.subspaces)
        @constraint(QP, sum(α[i, k] for i in 1:n) == 1)
    end

    debug(LOGGER, "[SOLVE] Solving QP with $(typeof(solver))...")
    JuMP.optimize!(QP)
    status = JuMP.termination_status(QP)
    debug(LOGGER, "[SOLVE] Finished with status: $(status).")
    model.alpha_values = [JuMP.result_value.(α)[:, k] for k in eachindex(model.subspaces)]
    return status
end

```
    Calculates the upper limits for support vectors in subspace subspace_idx
```
function calculate_upper_limit(α::Vector{Vector{Float64}}, subspace_idx::Int, C::Float64, v::Vector{Float64})
    return v.*C .- sum([α[i] for i in 1:length(α) if i != subspace_idx])
end

calculate_upper_limit(α::Vector{Vector{Float64}}, subspace_idx::Int, C) =
    calculate_upper_limit(α::Vector{Vector{Float64}}, subspace_idx::Int, C, fill(1.0, length(α[1])))

function find_support_vectors(model::SubSVDD, subspace_idx)::Vector{Int}
    upper_limits = SVDD.calculate_upper_limit(model.alpha_values, subspace_idx, model.C)
    return findall((model.alpha_values[subspace_idx] .> SVDD.OPT_PRECISION) .& (model.alpha_values[subspace_idx] .< (upper_limits[subspace_idx] .- SVDD.OPT_PRECISION)))
end

function find_positive_alpha(model::SubSVDD, subspace_idx)
    return findall(model.alpha_values[subspace_idx] .> SVDD.OPT_PRECISION)
end

function get_R_and_const_term(model::SubSVDD, subspace_idx)
    sv_idx = find_support_vectors(model, subspace_idx)
    α = model.alpha_values[subspace_idx]
    K = model.K[subspace_idx]

    const_term = sum(α[i]*α[j]*K[i,j] for i in eachindex(α) for j in eachindex(α))
    R_squared = 0.0
    tmp = 0.0
    for k in sv_idx
        tmp = K[k,k] - 2 * sum(α[i] * K[i,k] for i in eachindex(α)) + const_term
        R_squared = tmp > R_squared ? tmp : R_squared
    end
    return (R=sqrt(R_squared), const_term=const_term)
end

#TODO speed-up for in-sample predict by directly using kernel matrix
predict(model::SubSVDD, subspace_idx) = predict(model, model.data, subspace_idx)

# out of sample predict
function predict(model::SubSVDD, target::Array{T,2}, subspace_idx) where T <: Real
    @assert size(model.data, 1) == size(target, 1) "Dimension mismatch between model data and target.
        You have to provide the full data set, not the subspace projection."

    s = model.subspaces[subspace_idx]
    pos_sv_idx = find_positive_alpha(model, subspace_idx)
    function predict_observation(z)
        kernel(model.kernel_fct, z, z) -
             2 * sum(model.alpha_values[subspace_idx][i] * kernel(model.kernel_fct, model.data[s, i], z) for i in pos_sv_idx) +
             model.const_term[subspace_idx]
    end
    vec(sqrt.(mapslices(predict_observation, target[s, :], dims=1)) .- model.R[subspace_idx])
end
