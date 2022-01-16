
mutable struct SVDDnegEps <: SVDDClassifier
    state::ModelState

    # model parameters
    C1::Float64
    C2::Float64
    kernel_fct::Kernel
    eps::Float64

    # training data
    data::Array{Float64,2}
    K::Array{Float64,2}
    adjust_K::Bool
    K_adjusted::Array{Float64,2}
    pools::Dict{Symbol, Array{Int,1}}

    # fitted values
    alpha_values::Vector{Float64}
    const_term::Float64
    R::Float64

    function SVDDnegEps(data, pools; eps=0.0)
        m = new()
        m.C1 = 1.0
        m.C2 = 1.0
        m.eps = eps
        m.state = model_created
        m.data = data
        m.adjust_K = false
        m.pools = labelmap(pools)
        m.const_term = -Inf
        m.R = -Inf
        return m
    end
end

get_model_params(model::SVDDnegEps) = Dict(:C1 => model.C1, :C2 => model.C2, :eps => model.eps)

is_valid_param_value(model::SVDDnegEps, x::Type{Val{:C1}}, v) = 0 <= v <= 1
is_valid_param_value(model::SVDDnegEps, x::Type{Val{:C2}}, v) = is_valid_param_value(model, Val{:C1}, v)
is_valid_param_value(model::SVDDnegEps, x::Type{Val{:eps}}, v) = v >= 0

function set_C!(model::SVDDnegEps, C::Tuple{Number, Number})
    @assert 0 <= C[1] <= 1
    @assert 0 <= C[2] <= 1
    model.C1 = C[1]
    model.C2 = C[2]
    return nothing
end

set_C!(model::SVDDnegEps, C::Number) = set_C!(model, (C,C))

function set_eps!(model::SVDDnegEps, eps::Number)
    @assert eps >= 0
    model.eps = eps
    return nothing
end

function solve!(model::SVDDnegEps, solver::SOLVER_TYPE)
    ULin = merge_pools(model.pools, :U, :Lin)
    length(ULin) > 0 || throw(ModelInvariantException("SVDDnegEps requires samples in pool :Lin or :U."))

    debug(LOGGER, "[SOLVE] Setting up QP for SVDDnegEps with $(is_K_adjusted(model) ? "adjusted" : "non-adjusted") kernel matrix.")
    QP = Model(solver)
    K = is_K_adjusted(model) ? model.K_adjusted : model.K

    @variable(QP, α[1:size(K,1)] >= 0)

    if haskey(model.pools, :Lout)
        @objective(QP, Max, sum(α[i] * model.eps for i in ULin) +
                             sum(α[i]*K[i,i] for i in ULin) -
                             sum(α[l]*K[l,l] for l in model.pools[:Lout]) -
                             sum(α[i]*α[j] * K[i,j] for i in ULin for j in ULin) +
                             2 * sum(α[l]*α[j] * K[l,j] for l in model.pools[:Lout] for j in ULin) -
                             sum(α[l]*α[m] * K[l,m] for l in model.pools[:Lout] for m in model.pools[:Lout]))

        @constraint(QP, sum(α[i] for i in ULin) - sum(α[l] for l in model.pools[:Lout]) == 1)
        @constraint(QP, α[ULin] .<= model.C1)
        @constraint(QP, α[model.pools[:Lout]] .<= model.C2)
    else # fall back to standard SVDD
        @objective(QP, Max, sum(α[i]*K[i,i] for i in ULin) -
                             sum(α[i]*α[j] * K[i,j] for i in ULin for j in ULin))
        @constraint(QP, sum(α) == 1)
        @constraint(QP, α[ULin] .<= model.C1)
    end
    debug(LOGGER, "[SOLVE] Solving QP with $(typeof(solver))...")
    JuMP.optimize!(QP)
    status = JuMP.termination_status(QP)
    debug(LOGGER, "[SOLVE] Finished with status: $(status).")
    model.alpha_values = JuMP.value.(α)
    return status
end

function get_alpha_prime(model::SVDDnegEps)
    if haskey(model.pools, :Lout)
        alpha = copy(model.alpha_values)
        alpha[model.pools[:Lout]] = - alpha[model.pools[:Lout]]
        return alpha
    else
        return model.alpha_values
    end
end

function get_support_vectors(model::SVDDnegEps)
    ULin = merge_pools(model.pools, :U, :Lin)
    length(ULin) > 0 || throw(ModelInvariantException("SVDDnegEps requires samples in pool :Lin or :U."))
    sv = filter!(x -> x in ULin, findall((model.alpha_values .> OPT_PRECISION) .& (model.alpha_values .< (model.C1 - OPT_PRECISION))))
    if haskey(model.pools, :Lout)
        sv = append!(sv, filter!(x -> x in model.pools[:Lout],
                                 findall((model.alpha_values .> OPT_PRECISION) .& (model.alpha_values .< (model.C2 - OPT_PRECISION)))))
    end
    return sv
end

function get_R_and_const_term(model::SVDDnegEps)
    @assert length(model.alpha_values) == size(model.K, 1)

    sv_idx = get_support_vectors(model)
    α = get_alpha_prime(model)

    const_term = sum(α[i] * α[j]*model.K[i,j] for i in eachindex(α) for j in eachindex(α))
    R_squared = 0.0
    tmp = 0.0
    # take the maximum R over all support vectors for numerical reasons
    for k in sv_idx
        tmp = model.K[k,k] - 2 * sum(α[i] * model.K[i,k] for i in eachindex(α)) + const_term + model.eps
        R_squared = tmp > R_squared ? tmp : R_squared
    end
    return (sqrt(R_squared), const_term)
end
