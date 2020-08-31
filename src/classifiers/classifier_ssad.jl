"""
Original publication:
Görnitz, Nico, et al. "Toward supervised anomaly detection." Journal of Artificial Intelligence Research 46 (2013): 235-262.
"""
mutable struct SSAD <: OCClassifier
    state::ModelState

    # model parameters
    C1::Float64
    C2::Float64
    kernel_fct::Kernel
    κ::Float64
    kappa_fallback::Bool

    # training data
    data::Array{Float64,2}
    K::Array{Float64,2}
    adjust_K::Bool
    K_adjusted::Array{Float64,2}
    pools::Dict{Symbol, Array{Int64,1}}

    # fitted values
    alpha_values::Vector{Float64}
    ρ::Float64

    function SSAD(data, pools::Vector{Symbol})
        m = new()
        m.C1 = 1.0
        m.C2 = 1.0
        m.κ = 1.0
        m.data = data
        m.adjust_K = false
        m.kappa_fallback = true
        m.pools = labelmap(pools)
        m.state = model_created
        m.ρ = -Inf
        return m
    end
end

get_model_params(model::SSAD) = Dict(:C1 => model.C1, :C2 => model.C2, :κ => model.κ)

is_valid_param_value(model::SSAD, x::Type{Val{:C1}}, v) = 0 <= v <= 1
is_valid_param_value(model::SSAD, x::Type{Val{:C2}}, v) = 0 <= v <= 1
is_valid_param_value(model::SSAD, x::Type{Val{:κ}}, v) = true

function set_C!(model::SSAD, C::Tuple{Number, Number})
    @assert 0 <= C[1] <= 1
    @assert 0 <= C[2] <= 1
    model.C1 = C[1]
    model.C2 = C[2]
    return nothing
end

set_C!(model::SSAD, C::Number) = set_C!(model, (C,C))

function set_kappa!(model::SSAD, κ)
    invalidate_solution!(model)
    model.κ = κ
    return nothing
end

function set_kappa_fallback!(model::SSAD, kappa_fallback::Bool)
    model.kappa_fallback = kappa_fallback
    return nothing
end

function get_cy(model::SSAD)
    cy = ones(size(model.data, 2))
    haskey(model.pools, :Lout) && (cy[model.pools[:Lout]] .= -1)
    return cy
end

function calculate_rho(model::SSAD)
    SV_candidates = findall(model.alpha_values .> OPT_PRECISION)
    SV_candidates_U = haskey(model.pools, :U) ? SV_candidates ∩ model.pools[:U] : Int64[]
    cy = get_cy(model)

    if length(SV_candidates_U) > 0
        scores = (model.alpha_values .* cy)' * model.K[:, SV_candidates_U]
        sv = findall(model.alpha_values[SV_candidates_U] .< model.C1 - OPT_PRECISION)
        ρ = isempty(sv) ? maximum(scores) : minimum(scores[sv])
    else
        scores = model.K'model.alpha_values
        SV_candidates_Lin = haskey(model.pools, :Lin) ? SV_candidates ∩ model.pools[:Lin] : Int64[]
        SV_candidates_Lout = haskey(model.pools, :Lout) ? SV_candidates ∩ model.pools[:Lout] : Int64[]
        if length(SV_candidates_Lout) > 0 && length(SV_candidates_Lin) == 0
            warn(LOGGER, "[CALCULATE_RHO] There are no labeled inlier SV -- check OPT_PRECISION.")
            ρ = maximum(scores[SV_candidates_Lout])
        elseif length(SV_candidates_Lout) == 0
            ρ = minimum(scores[SV_candidates_Lin])
        else
            ρ = (maximum(scores[SV_candidates_Lin]) + minimum(scores[SV_candidates_Lout])) / 2
        end
    end
    return ρ
end

function predict(model::SSAD, target::Array{T,2}) where T <: Real
    model.state == model_fitted || throw(ModelStateException(model.state, model_fitted))
    SV_candidates = findall(model.alpha_values .> OPT_PRECISION)
    function predict_observation(z)
        k = vec(mapslices(x -> kernel(model.kernel_fct, z, x), model.data[:, SV_candidates], dims=1))
        model.alpha_values[SV_candidates]'k
    end
     # this is inverted from Goernitz such that outliers have positive margin to be consistent with SVDD
    return model.ρ .- vec(mapslices(predict_observation, target, dims=1))
end

function fit!(model::SSAD, solver::SOLVER_TYPE)
    debug(LOGGER, "[FIT] Fitting SSAD.")
    model.state == model_created && throw(ModelStateException(model.state, model_initialized))

    # Workaround; Gurobi throws "Gurobi.GurobiError(10005, "Unable to retrieve attribute 'UnbdRay'")" instead of setting solver status
    try
        status = solve!(model, solver)
    catch e
        status = JuMP.MathOptInterface.OTHER_ERROR
    end

    if status != JuMP.MathOptInterface.OPTIMAL && model.kappa_fallback
        debug(LOGGER, "[FIT] Solver returned with status != :Optimal. Retry with kappa = 0.0.")
        κ_orig = model.κ
        model.κ = 0.0
        status = solve!(model, solver)
        model.κ = κ_orig
    end
    model.ρ = calculate_rho(model)
    model.state = model_fitted
    debug(LOGGER, "[FIT] SSAD is now in state $(model.state).")
    return status
end

# see also tilitools https://github.com/nicococo/tilitools/blob/master/tilitools/ssad_convex.py
function solve!(model::SSAD, solver::SOLVER_TYPE)
    debug(LOGGER, "[SOLVE] Setting up QP for SSAD with $(is_K_adjusted(model) ? "adjusted" : "non-adjusted") kernel matrix.")
    QP = Model(solver)
    K = is_K_adjusted(model) ? model.K_adjusted : model.K
    # optimization variables
    @variable(QP, α[1:size(K,1)] >= 0)

    L = merge_pools(model.pools, :Lin, :Lout)
    ULin = merge_pools(model.pools, :U, :Lin)
    cy = get_cy(model)

    # objective function
    @objective(QP, Max, -0.5*sum(α[i]*α[j] * K[i,j] * cy[i] * cy[j] for i in eachindex(α) for j in eachindex(α)))

    # constraints
    haskey(model.pools, :U) && @constraint(QP, α[model.pools[:U]] .<= model.C1)
    if !isempty(L)
        @constraint(QP, α[L] .<= model.C2)
        @constraint(QP, sum(α[i] for i in L) >= model.κ)
    end
    @constraint(QP, sum(α[i] * cy[i] for i in eachindex(α)) == 1)

    debug(LOGGER, "[SOLVE] Solving QP with $(typeof(solver))...")
    JuMP.optimize!(QP)
    status = JuMP.termination_status(QP)
    debug(LOGGER, "[SOLVE] Finished with status: $(status).")
    model.alpha_values = JuMP.value.(α)
    return status
end

function invalidate_solution!(model::SSAD)
    model.alpha_values = []
    model.ρ = -Inf
    model.state = model_initialized
    return nothing
end
