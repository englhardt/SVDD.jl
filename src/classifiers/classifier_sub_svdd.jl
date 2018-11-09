
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
    @assert 0 <= C <= 1
    model.C = C
    return nothing
end
