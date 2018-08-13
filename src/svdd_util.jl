
function merge_pools(pools, names...)
    (names[1] == [] || MLLabelUtils.islabelenc(collect(names), SVDD.learning_pool_enc)) || throw(ArgumentError("$(collect(names)) is not a valid label encoding."))
    return reduce((r, key) -> vcat(r, haskey(pools, key) ? pools[key] : Int64[]), Int64[], unique(names))
end

classify(x::Number) = x > 0 ? :outlier : :inlier

function adjust_kernel_matrix(K::Array{T, 2}; tolerance = 1e-15, warn_threshold = 1e-8) where T <: Real
    info(LOGGER, "Adjusting Kernel Matrix.")
    F = eigfact(K)
    eltype(F[:values]) <: Complex && throw(ArgumentError("Matrix K has complex eigenvalues."))
    F[:values][F[:values] .< tolerance] = 0.0
    K_adjusted::Array{Float64, 2} = F[:vectors] * diagm(F[:values]) * inv(F[:vectors])
    K_diff = abs.(K_adjusted - K)
    sum_adjustment = sum(K_diff)
    max_adjustment = maximum(K_diff)
    info(LOGGER, "[ADJUST KERNEL] Maximum adjustemt of kernel matrix entry is $max_adjustment. The sum of adjustments is $(sum_adjustment)")
    max_adjustment .> warn_threshold && warn(LOGGER, "[ADJUST KERNEL] Maximum adjustment of kernel matrix entry exceeded the threshold of $(warn_threshold)!.")
    return K_adjusted
end
