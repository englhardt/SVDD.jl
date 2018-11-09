
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
    model.R, model.const_term = get_R_and_const_term(model)
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
    sv = findall((model.alpha_values[subspace_idx] .> SVDD.OPT_PRECISION) .& (model.alpha_values[subspace_idx] .< (upper_limits[subspace_idx] .- SVDD.OPT_PRECISION)))
    return sv
end
```
    Returns the indices of support vectors for each subspace
```
function find_support_vectors(model::SubSVDD)::Vector{Vector{Int}}
    [find_support_vectors(model, k) for k in eachindex(model.subspaces)]
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

function get_R_and_const_term(model::SubSVDD)
    R = Vector{Float64}(undef, length(model.subspaces))
    const_term = Vector{Float64}(undef, length(model.subspaces))
    for k in eachindex(model.subspaces)
        R[k], const_term[k] = get_R_and_const_term(model, k)
    end
    return (R=R, const_term=const_term)
end
