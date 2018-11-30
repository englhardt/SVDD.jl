
function instantiate(model_type::DataType, data, pools, param::Dict{Symbol, <:Any})
    debug(LOGGER, "Instantiating model of type $(model_type) with params $(param)")
    m = model_type(data, pools)
    set_param!(m, param)
    return m
end

function set_param!(model, param::Dict{Symbol, <:Any})
    has_changed = false
    for (k, v) in param
        k in keys(get_model_params(model)) || throw(ArgumentError("Model $(typeof(model)) has no parameter $k. Valid parameters are $(get_model_params(model))."))
        fieldtype = typeof(getfield(model, k))
        is_valid_param_value(model, Val{k}, v) || throw(ArgumentError("Parameter $k has value $v which is outside the valid range for model $(typeof(model))."))
        if getfield(model,k) != v
            has_changed = true
            setfield!(model, k, v)
        end
    end
    has_changed && invalidate_solution!(model)
    return nothing
end

function set_pools!(model, pools::Vector{Symbol})
    islabelenc(pools, learning_pool_enc) || throw(ArgumentError("Pools must be a LabelEncoding of $(learning_pool_enc.label)."))
    set_pools!(model, labelmap(pools))
    return nothing
end

function set_pools!(model, pools::Dict{Symbol, Vector{Int}})
    if !isdefined(model, :pools) ||model.pools != pools
        model.pools = pools
        invalidate_solution!(model)
    end
    return nothing
end

function set_adjust_K!(model, adjust_K::Bool)
    if adjust_K && !model.adjust_K
        model.adjust_K = adjust_K
        update_K!(model)
        invalidate_solution!(model)
    elseif !adjust_K && model.adjust_K
        model.adjust_K = adjust_K
        invalidate_solution!(model)
    end
    return nothing
end

is_K_adjusted(model)::Bool = model.adjust_K

calculate_kernel_matrix(model) = MLKernels.kernelmatrix(Val(:col), model.kernel_fct, model.data)

function calculate_kernel_matrix(model::SubSVDD)
     map(k -> MLKernels.kernelmatrix(Val(:col), model.kernel_fct[k], model.data[model.subspaces[k],:]), eachindex(model.subspaces))
 end

function update_K!(model)
    updated_K = calculate_kernel_matrix(model)
    if !isdefined(model, :K) || updated_K != model.K || (!isdefined(model, :K_adjusted) && model.adjust_K)
        debug(LOGGER, "[UPDATE_K] Updating Kernel matrix.")
        invalidate_solution!(model)
        model.K = updated_K
        if model.adjust_K
            model.K_adjusted = adjust_kernel_matrix(updated_K)
        end
    else
        debug(LOGGER, "[UPDATE_K] Nothing to update, kernel matrix has not changed.")
    end
    return nothing
end

function set_data!(model, data::Array{T, 2}) where T <: Real
    if model.data != data
        model.data = data
        update_K!(model)
        invalidate_solution!(model)
    end
    return nothing
end

function set_kernel!(model::OCClassifier, kernel_fct)
    if !isdefined(model, :kernel_fct) || model.kernel_fct != kernel_fct
        model.kernel_fct = kernel_fct
        update_K!(model)
        invalidate_solution!(model)
    end
    return nothing
end

function initialize!(model::OCClassifier, strategy::InitializationStrategy)
    debug(LOGGER, "[INITIALIZE] Initializing $(typeof(model)) with strategy: $(typeof(strategy))")
    C, kernel_fct = get_parameters(model, strategy)
    debug(LOGGER, "[INITIALIZE] Fitted parameters are: C = $(C) and kernel_fct = $(kernel_fct)")
    set_C!(model, C)
    set_kernel!(model, kernel_fct)
    model.state = model_initialized
    return nothing
end

get_kernel(model::OCClassifier) = model.kernel_fct

apply_update_strategy!(model, new_pools, query_ids, old_idx_remaining, new_idx_remaining) = nothing

function update_with_feedback!(model, new_data, new_pools::Vector{Symbol}, query_ids::Vector{Int},
    old_idx_remaining::Vector{Int},
    new_idx_remaining::Vector{Int})
    @assert size(new_data, 2) == length(new_pools)
    @assert islabelenc(new_pools, SVDD.learning_pool_enc)

    apply_update_strategy!(model, new_pools, query_ids, old_idx_remaining, new_idx_remaining)

    set_pools!(model, new_pools)
    set_data!(model, new_data)
    invalidate_solution!(model)
    return nothing
end
