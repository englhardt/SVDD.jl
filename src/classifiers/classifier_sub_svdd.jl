
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
    const_term::Float64
    R::Float64

    function SubSVDD(data, subspaces, pools)
        m = new()
        m.C = 1.0
        m.state = model_created
        m.data = data
        m.adjust_K = false
        m.pools = labelmap(pools)
        m.subspaces = subspaces
        m.const_term = -Inf
        m.R = -Inf
        return m
    end
end

function invalidate_solution!(model::SubSVDD)
    model.alpha_values = Vector{Vector{Float64}}()
    model.const_term = -Inf
    model.R = -Inf
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
    # model.R, model.const_term = get_R_and_const_term(model)
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
