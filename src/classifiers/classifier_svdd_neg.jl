"""
Original publication:
Tax, David MJ, and Robert PW Duin. "Support vector data description." Machine learning 54.1 (2004): 45-66.
"""
mutable struct SVDDneg <: SVDDClassifier
    state::ModelState

    # model parameters
    C1::Float64
    C2::Float64
    kernel_fct::Kernel

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

    function SVDDneg(data, pools)
        m = new()
        m.C1 = 1.0
        m.C2 = 1.0
        m.state = model_created
        m.data = data
        m.adjust_K = false
        m.pools = labelmap(pools)
        m.const_term = -Inf
        m.R = -Inf
        return m
    end
end

get_model_params(model::SVDDneg) = Dict(:C1 => model.C1, :C2 => model.C2)

is_valid_param_value(model::SVDDneg, x::Type{Val{:C1}}, v) = 0 <= v <= 1
is_valid_param_value(model::SVDDneg, x::Type{Val{:C2}}, v) = is_valid_param_value(model, Val{:C1}, v)

function set_C!(model::SVDDneg, C::Tuple{Number, Number})
    @assert 0 <= C[1] <= 1
    @assert 0 <= C[2] <= 1
    model.C1 = C[1]
    model.C2 = C[2]
    return nothing
end

set_C!(model::SVDDneg, C::Number) = set_C!(model, (C,C))

function solve!(model::SVDDneg, solver::SOLVER_TYPE)
    ULin = merge_pools(model.pools, :U, :Lin)
    length(ULin) > 0 || throw(ModelInvariantException("SVDDneg requires samples in pool :Lin or :U."))

    debug(LOGGER, "[SOLVE] Setting up QP for SVDDneg with $(is_K_adjusted(model) ? "adjusted" : "non-adjusted") kernel matrix.")
    QP = Model(solver)
    K = is_K_adjusted(model) ? model.K_adjusted : model.K

    @variable(QP, α[1:size(K,1)] >= 0)

    if haskey(model.pools, :Lout)
        @objective(QP, Max, sum(α[i]*K[i,i] for i in ULin) -
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

function get_alpha_prime(model::SVDDneg)
    if haskey(model.pools, :Lout)
        alpha = copy(model.alpha_values)
        alpha[model.pools[:Lout]] = - alpha[model.pools[:Lout]]
        return alpha
    else
        return model.alpha_values
    end
end

function get_support_vectors(model::SVDDneg)
    ULin = merge_pools(model.pools, :U, :Lin)
    length(ULin) > 0 || throw(ModelInvariantException("SVDDneg requires samples in pool :Lin or :U."))
    sv = filter!(x -> x in ULin, findall((model.alpha_values .> OPT_PRECISION) .& (model.alpha_values .< (model.C1 - OPT_PRECISION))))
    if haskey(model.pools, :Lout)
        sv = append!(sv, filter!(x -> x in model.pools[:Lout],
                                 findall((model.alpha_values .> OPT_PRECISION) .& (model.alpha_values .< (model.C2 - OPT_PRECISION)))))
    end
    return sv
end
