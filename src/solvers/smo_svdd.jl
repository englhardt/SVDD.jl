struct SMOSolver <: SVDDSolver
    opt_precision::Float64
    max_iterations::Int
end

"""
    takeStep!(α, i1, i2, K, C, opt_precision)

    Take optimization step for i1 and i2 and update α.
"""
function takeStep!(α, i1, i2, K, C, opt_precision)
    i1 == i2 && return false

    L = max(0, α[i1] + α[i2] - C)
    H = min(C, α[i1] + α[i2])
    (abs(L - H) < opt_precision) && return false

    Δ = α[i1] + α[i2]
    c(i) = sum(α[j]*K[i,j] for j in eachindex(α) if !(j in [i1, i2]))

    alpha2 = ((2*Δ*(K[i1, i1] - K[i1,i2]) + c(i1) - c(i2) - K[i1, i1] + K[i2, i2]) /
             (2*K[i1,i1] - 4*K[i1,i2] + 2*K[i2,i2]))

    if alpha2 > H
        alpha2 = H
    elseif alpha2 < L
        alpha2 = L
    end

    # see page 10
    # J. Platt, "Sequential minimal optimization: A fast algorithm for training support vector machines," 1998.
    if abs(α[i2] - alpha2) < opt_precision * (alpha2 + α[i2] + opt_precision)
        return false
    else
        alpha1 = Δ - alpha2
        α[i1] = alpha1
        α[i2] = alpha2
        return true
    end
end

# TODO: this is a faster version of predict from classifier_svdd for in-sample predictions
# this should be the default for in-sample predictions
"""
    calculate_predictions(α, K, C, opt_precision)
"""
function calculate_predictions(α, K, C, opt_precision)
    sv_larger_than_zero = α .> opt_precision
    sv_smaller_than_C = α .< (C - opt_precision)

    sv_larger_than_zero_idx = findall(sv_larger_than_zero)
    const_term = sum(α[i] * α[j] * K[i,j] for i in sv_larger_than_zero_idx for j in sv_larger_than_zero_idx)
    distances_to_center = [K[z, z] - 2 * sum(α[i] * K[i, z] for i in sv_larger_than_zero_idx) + const_term for z in eachindex(α)]

    ## see W.-C. Chang, C.-P. Lee, and C.-J. Lin, "A revisit to support vector data description," 2013
    if any(sv_larger_than_zero .& sv_smaller_than_C)
        R = mean(distances_to_center[sv_larger_than_zero .& sv_smaller_than_C])
    else
        R = (minimum(distances_to_center[sv_larger_than_zero]) + maximum(distances_to_center[sv_larger_than_zero])) / 2
    end

    distances_to_decision_boundary = distances_to_center .- R
    return (distances_to_center, distances_to_decision_boundary, R)
end
"""
    violates_KKT_condition(i2, distances_to_decision_boundary, α, C, opt_precision)

"""
function violates_KKT_condition(i2, distances_to_decision_boundary, α, C, opt_precision)
    p1 = (α[i2] > opt_precision) && (distances_to_decision_boundary[i2] < -opt_precision) # inlier, but alpha > 0
    p2 = (α[i2] < C - opt_precision) && (distances_to_decision_boundary[i2] > opt_precision) # outlier, but alpha != C
    return p1 || p2
end

# See Equation (4.8) in
# B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson,
# "Estimating the support of a high-dimensional distribution," 2001
"""
    second_choice_heuristic(i2, α, distances_to_center, C, opt_precision)
"""
function second_choice_heuristic(i2, α, distances_to_center, C, opt_precision)
    SV_nb = (α .> opt_precision) .& (α .< C - opt_precision)
    if !any(SV_nb)
        SV_nb = α .> opt_precision
    end
    findall(SV_nb)[findmax(abs.(distances_to_center[i2] .- distances_to_center[SV_nb]))[2]]
end


"""
    examineExample!(α, i2, distances_to_center, K, C, opt_precision)

    The fallback strategies if second choice heuristic returns false follow recommendations in
    J. Platt, "Sequential minimal optimization: A fast algorithm for training support vector machines," 1998.
"""
function examineExample!(α, i2, distances_to_center, K, C, opt_precision)
    # use the second choice heuristic
    i1 = second_choice_heuristic(i2, α, distances_to_center, C, opt_precision)
    takeStep!(α, i1, i2, K, C, opt_precision) && return true

    # loop over all non-zero and non-C alpha, starting at random position
    candidates = findall((α .> opt_precision) .& (α .< C - opt_precision))
    if !isempty(candidates)
        for i1 in shuffle(candidates)
            takeStep!(α, i1, i2, K, C, opt_precision) && return true
        end
    end

    # loop over all
    for i1 in shuffle(eachindex(α))
        takeStep!(α, i1, i2, K, C, opt_precision) && return true
    end
    return false
end

"""
    initialize_alpha(data, C)
"""
function initialize_alpha(data, C)
    n_init = trunc(Int, 1 / (C)) + 1
    α = fill(0.0, size(data, 2))
    α[sample(1:size(data, 2), n_init, replace=false)] .= 1 / n_init
    @assert sum(α) ≈ 1 && all(α .<= C)
    return α
end

"""
    examine_and_update_predictions!(α, distances_to_center, distances_to_decision_boundary, R,
        KKT_violations, black_list, K, C, opt_precision)
"""
function examine_and_update_predictions!(α, distances_to_center, distances_to_decision_boundary, R,
        KKT_violations, black_list, K, C, opt_precision)
    i2 = sample(KKT_violations)
    if examineExample!(α, i2, distances_to_center, K, C, opt_precision)
        distances_to_center, distances_to_decision_boundary, R = calculate_predictions(α, K, C, opt_precision)
    else
        push!(black_list, i2)
    end
    return distances_to_center, distances_to_decision_boundary, R
end

"""
    smo(α, K, C, opt_precision, max_iterations)

"""
function smo(α, K, C, opt_precision, max_iterations)
    distances_to_center, distances_to_decision_boundary, R = calculate_predictions(α, K, C, opt_precision)

    iter = 0
    while iter < max_iterations
        # Fall back strategy: add indices to black list if examine can not make positive step.
        # See page 9, J. Platt, "Sequential minimal optimization: A fast algorithm for training support vector machines," 1998.
        black_list = Set{Int}()
        iter += 1

        # scan over all data
        KKT_violation_all_idx = filter(i -> violates_KKT_condition(i, distances_to_decision_boundary, α, C, opt_precision) && i ∉ black_list, eachindex(α))
        if isempty(KKT_violation_all_idx)
            return build_result(α, distances_to_decision_boundary, R, K, C, opt_precision, :Optimal, "No more KKT_violations.")
        else
            distances_to_center, distances_to_decision_boundary, R = examine_and_update_predictions!(α, distances_to_center, distances_to_decision_boundary, R, KKT_violation_all_idx, black_list, K, C, opt_precision)
        end

        # scan over SV_nb
        SV_nb = (α .> opt_precision) .& (α .< C - opt_precision)
        KKT_violations_in_SV_nb = filter(i -> violates_KKT_condition(i, distances_to_decision_boundary, α, C, opt_precision) && i ∉ black_list, findall(SV_nb))
        while length(KKT_violations_in_SV_nb) > 0 && iter < max_iterations
            iter += 1
            distances_to_center, distances_to_decision_boundary, R = examine_and_update_predictions!(α, distances_to_center, distances_to_decision_boundary, R, KKT_violations_in_SV_nb, black_list, K, C, opt_precision)
            KKT_violations_in_SV_nb = filter(i -> violates_KKT_condition(i, distances_to_decision_boundary, α, C, opt_precision) && i ∉ black_list, findall(SV_nb))
        end
    end
    return build_result(α, distances_to_decision_boundary, R, K, C, opt_precision, :UserLimit, "Reached max number of iterations.")
end

function calculate_duality_gap(α, distances_to_decision_boundary, R, K, C, opt_precision)
    sv_larger_than_zero_idx = findall(α .> opt_precision)
    primal_obj = R + sum(distances_to_decision_boundary[distances_to_decision_boundary .> opt_precision] * C)
    dual_obj = sum(α[i] * K[i,i] for i in sv_larger_than_zero_idx) - sum(α[i] * α[j] * K[i,j] for i in sv_larger_than_zero_idx for j in sv_larger_than_zero_idx)
    duality_gap = primal_obj - dual_obj
    return primal_obj, dual_obj, duality_gap
end

function build_result(α, distances_to_decision_boundary, R, K, C, opt_precision, status, msg)
    if status == :Optimal
        info(LOGGER, "Exit with status: $status. ($msg)")
    else
        warn(LOGGER, "Exit with status: $status. ($msg)")
    end
    primal_obj, dual_obj, duality_gap = calculate_duality_gap(α, distances_to_decision_boundary, R, K, C, opt_precision)
    info(LOGGER, "duality gap: $duality_gap, primal objective: $primal_obj, dual objective: $dual_obj")
   return α, primal_obj, duality_gap, duality_gap, status
end

"""
    solve!(model::VanillaSVDD, solver::SMOSolver)
"""
function solve!(model::VanillaSVDD, solver::SMOSolver)
    α = initialize_alpha(model.data, model.C)
    model.alpha_values, primal_obj, dual_obj, duality_gap, status = smo(α, model.K, model.C, solver.opt_precision, solver.max_iterations)
    return status
end
