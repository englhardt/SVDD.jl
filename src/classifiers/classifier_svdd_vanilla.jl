"""
Original publication:
Tax, David MJ, and Robert PW Duin. "Support vector data description." Machine learning 54.1 (2004): 45-66.
"""
mutable struct VanillaSVDD <: SVDDClassifier
    state::ModelState

    # model parameters
    C::Float64
    kernel_fct::Kernel

    # training data
    data::Array{Float64,2}
    K::Array{Float64,2}
    adjust_K::Bool
    K_adjusted::Array{Float64,2}

    # fitted values
    alpha_values::Vector{Float64}
    const_term::Float64
    R::Float64

    function VanillaSVDD(data)
        m = new()
        m.C = 1.0
        m.state = model_created
        m.data = data
        m.adjust_K = false
        m.const_term = -Inf
        m.R = -Inf
        return m
    end
    VanillaSVDD(data, pools) = VanillaSVDD(data)
end

get_model_params(model::VanillaSVDD) = Dict(:C => model.C)

is_valid_param_value(model::VanillaSVDD, x::Type{Val{:C}}, v) = 0 <= v <= 1

set_pools!(model::VanillaSVDD, pools::Vector{Symbol}) = nothing
set_pools!(model::VanillaSVDD, pools::Dict{Symbol, Vector{Int}}) = nothing

function set_C!(model::VanillaSVDD, C::Number)
    @assert 0 <= C <= 1
    model.C = C
    return nothing
end

function solve!(model::VanillaSVDD, solver::SOLVER_TYPE)
    debug(LOGGER, "[SOLVE] Setting up QP for VanillaSVDD with $(is_K_adjusted(model) ? "adjusted" : "non-adjusted") kernel matrix.")
    QP = Model(solver)
    K = is_K_adjusted(model) ? model.K_adjusted : model.K

    @variable(QP, 0 <= α[1:size(K,1)] <= model.C)
    @objective(QP, Max, sum(α[i]*K[i,i] for i in eachindex(α)) -
                        sum(α[i]*α[j] * K[i,j] for i in eachindex(α) for j in eachindex(α)))
    @constraint(QP, sum(α) == 1)
    debug(LOGGER, "[SOLVE] Solving QP with $(typeof(solver))...")
    JuMP.optimize!(QP)
    status = JuMP.termination_status(QP)
    debug(LOGGER, "[SOLVE] Finished with status: $(status).")
    model.alpha_values = JuMP.value.(α)
    return status
end

function get_support_vectors(model::VanillaSVDD)
    findall((model.alpha_values .> OPT_PRECISION) .& (model.alpha_values .< (model.C - OPT_PRECISION)))
end

get_alpha_prime(model::VanillaSVDD) = model.alpha_values
