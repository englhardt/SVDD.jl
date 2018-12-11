
function merge_pools(pools, names...)
    (names[1] == [] || MLLabelUtils.islabelenc(collect(names), SVDD.learning_pool_enc)) || throw(ArgumentError("$(collect(names)) is not a valid label encoding."))
    return reduce((r, key) -> vcat(r, haskey(pools, key) ? pools[key] : Int64[]), unique(names); init=Int64[])
end

classify(x::Number; opt_precision = OPT_PRECISION) = x > opt_precision ? :outlier : :inlier

function classify(predictions::Vector{Vector{Float64}}, scope::Scope; opt_precision = OPT_PRECISION)
    if isa(scope, Val{:Subspace})
        return map(x -> classify.(x; opt_precision = opt_precision), predictions)
    else
        is_global_outlier = mapreduce(x -> x .> opt_precision, (a,b) -> a .| b, predictions)
        return ifelse.(is_global_outlier, :outlier, :inlier)
    end
end

function adjust_kernel_matrix(K::Array{T, 2}; tolerance = 1e-15, warn_threshold = 1e-8) where T <: Real
    info(LOGGER, "Adjusting Kernel Matrix.")
    F = eigen(K)
    eltype(F.values) <: Complex && throw(ArgumentError("Matrix K has complex eigenvalues."))
    F.values[F.values .< tolerance] .= 0.0
    K_adjusted::Array{Float64, 2} = F.vectors * diagm(0 => F.values) * inv(F.vectors)
    K_diff = abs.(K_adjusted - K)
    sum_adjustment = sum(K_diff)
    max_adjustment = maximum(K_diff)
    info(LOGGER, "[ADJUST KERNEL] Maximum adjustemt of kernel matrix entry is $max_adjustment. The sum of adjustments is $(sum_adjustment)")
    max_adjustment .> warn_threshold && warn(LOGGER, "[ADJUST KERNEL] Maximum adjustment of kernel matrix entry exceeded the threshold of $(warn_threshold)!.")
    return K_adjusted
end

function adjust_kernel_matrix(K::Vector{Array{T, 2}}; tolerance = 1e-15, warn_threshold = 1e-8) where T <: Real
    adjust_kernel_matrix.(K)
end

function min_max_normalize(x)
   min, max = extrema(x)
   (x .- min) ./ (max .- min)
end
