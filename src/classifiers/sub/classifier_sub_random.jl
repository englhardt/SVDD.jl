mutable struct SubRandomOCClassifier <: SubOCClassifier
    data
    subspaces
    outlier_bias
    SubRandomOCClassifier(data, subspaces) = new(data, subspaces, 0.1)
    SubRandomOCClassifier(data, subspaces, pools::Vector) = SubRandomOCClassifier(data, subspaces)
    SubRandomOCClassifier(data, subspaces, outlier_bias::Float64) = new(data, subspaces, outlier_bias)
    SubRandomOCClassifier(data, subspaces, outlier_bias::Float64, pools::Vector) = SubRandomOCClassifier(data, subspaces, outlier_bias)
end

set_adjust_K!(model::SubRandomOCClassifier, adjust_K::Bool) = nothing

get_model_params(model::SubRandomOCClassifier) = (:Ϟ => 0.42, :param2 => 5)

function fit!(model::SubRandomOCClassifier, solver)
    debug(LOGGER, "[FIT] $(typeof(model)) always returns :Optimal.")
    return JuMP.MathOptInterface.OPTIMAL
end

function predict(model::SubRandomOCClassifier, target::Array{T,2}, subspace_idx) where T <: Real
    rand(size(target, 2)) * 2 .- model.outlier_bias * 2
end

function predict(model::SubRandomOCClassifier, target::Array{T,2}) where T <: Real
    [predict(model, target[model.subspaces[idx], :], idx) for idx in eachindex(model.subspaces)]
end

initialize!(model::SubRandomOCClassifier, strategy::InitializationStrategy) = nothing

get_kernel(model::SubRandomOCClassifier) = nothing

invalidate_solution!(model::SubRandomOCClassifier) = nothing

set_pools!(model::SubRandomOCClassifier, pools::Vector{Symbol}) = nothing
set_pools!(model::SubRandomOCClassifier, pools::Dict{Symbol, Vector{Int}}) = nothing

function set_data!(model::SubRandomOCClassifier, data::Array{T, 2}) where T <: Real
    model.data = data
    return nothing
end
