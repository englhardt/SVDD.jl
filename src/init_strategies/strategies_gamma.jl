abstract type InitializationStrategyGamma <: InitializationStrategy end

struct FixedGammaStrategy <: InitializationStrategyGamma
    kernel
end

calculate_gamma(model, strategy::FixedGammaStrategy) = MLKernels.getvalue(strategy.kernel.alpha)
function calculate_gamma(model::SubSVDD, strategy::FixedGammaStrategy, subspace_idx)
    MLKernels.getvalue(strategy.kernel[subspace_idx].alpha)
end

"""
Original publication:
Silverman, Bernard W. Density estimation for statistics and data analysis. Routledge, 2018.
"""
struct RuleOfThumbSilverman <: InitializationStrategyGamma end

function rule_of_thumb_silverman(data::Array{T,2}) where T <: Real
    return (size(data, 2) * (size(data, 1) + 2) / 4.0)^(-1.0 / (size(data,1) + 4.0))
end

calculate_gamma(model, strategy::RuleOfThumbSilverman) = rule_of_thumb_silverman(model.data)

function calculate_gamma(model::SubSVDD, strategy::RuleOfThumbSilverman, subspace_idx)
    return rule_of_thumb_silverman(model.data[model.subspaces[subspace_idx], :])
end

"""
Original publication:
Scott, David W. Multivariate density estimation: theory, practice, and visualization. John Wiley & Sons, 2015.
"""
struct RuleOfThumbScott <: InitializationStrategyGamma end

function rule_of_scott(data::Array{T,2}) where T <: Real
    return size(data, 2)^(-1.0/(size(data, 1) + 4))
end

calculate_gamma(model, strategy::RuleOfThumbScott) = rule_of_scott(model.data)

function calculate_gamma(model::SubSVDD, strategy::RuleOfThumbScott, subspace_idx)
    return rule_of_scott(model.data[model.subspaces[subspace_idx], :])
end

"""
Generate binary data to tune a one class classifier according to the following paper:
Wang, S. et al. 2018. Hyperparameter selection of one-class support vector machine by self-adaptive data
shifting. Pattern Recognition. 74, 2018.
"""
struct WangGammaStrategy <: InitializationStrategyGamma
    solver
    gamma_search_range
    C
    scoring_function
end

WangGammaStrategy(solver) = WangGammaStrategy(solver, 1.0)
WangGammaStrategy(solver, C) = WangGammaStrategy(solver, 10.0.^range(-2, stop=2, length=50), C, f1_scoring)
WangGammaStrategy(solver, gamma_search_range, C) = WangGammaStrategy(solver, gamma_search_range, C, f1_scoring)

function generate_binary_data_for_tuning(data, k=nothing, threshold=0.1)
    if k === nothing
        k = round(Int, ceil(5 * log10(size(data, 2))))
    end
    tree = KDTree(data)
    edge_idx = Int[]
    norm_vec = []
    data_target = []
    l_ns = 0

    for i in 1:size(data, 2)
        idx, dist = knn(tree, data[:, i], k + 1, true)
        v_ij = mapslices(normalize, data[:, i] .- data[:, idx[2:end]], dims=1)
        n_i = sum(v_ij, dims=2)
        θ_ij = sum(v_ij .* n_i, dims=1)
        l_i = 1 / k * sum(θ_ij .>= 0)
        if l_i >= 1 - threshold
            # add new edge
            push!(edge_idx, i)
            push!(norm_vec, n_i)
        end

        # generate pseudo inlier
        n_i = normalize(vec(n_i))
        Λ_i_positive = sum(n_i .* (data[:, idx[2:end]] .- data[:, i]), dims=1)
        if length(Λ_i_positive[Λ_i_positive .> 0]) > 0
            # shift along positive direction of data density gradient
            x_ij_min_positive = minimum(Λ_i_positive[Λ_i_positive .> 0])
            push!(data_target, data[:, i] + x_ij_min_positive * n_i)
        end
        l_ns += 1 / k * sum(dist)
    end

    # compute negative shift amount
    l_ns *= 1 / length(edge_idx)
    # generate pseudo outliers by shifting along negative direction
    data_outliers = data[:, edge_idx] + mapslices(normalize, hcat(norm_vec...), dims=1) * l_ns

    data_target = hcat(data_target...)

    return data_target, data_outliers
end

function f1_scoring(predictions, ground_truth)
    return MLBase.f1score(MLBase.roc(ground_truth .== :outlier, predictions .== :outlier))
end

function calculate_gamma(model::SubSVDD, strategy::WangGammaStrategy, subspace_idx)
    info(LOGGER, "[Gamma Search] Using VanillaSVDD to search for gamma in subspace $subspace_idx")
    calculate_gamma(VanillaSVDD(model.data[model.subspaces[subspace_idx], :]), strategy)
end

function calculate_gamma(model::SubSVDD, strategy::WangGammaStrategy)
    info(LOGGER, "[Gamma Search] Using VanillaSVDD to estimate a global gamma.")
    calculate_gamma(VanillaSVDD(model.data), strategy)
end
function calculate_gamma(model, strategy::WangGammaStrategy)
    m = deepcopy(model)
    data_target, data_outliers = generate_binary_data_for_tuning(m.data)
    ground_truth = vcat(fill(:inlier, size(m.data, 2) + size(data_target, 2)),
                    fill(:outlier, size(data_outliers, 2)))

    debug(LOGGER, "[Gamma Search] Searching for parameter C.")
    best_gamma = 1.0
    best_score = -Inf
    for gamma in strategy.gamma_search_range
        debug(LOGGER, "[Gamma Search] Testing gamma = $gamma")
        init_strategy = FixedParameterInitialization(GaussianKernel(gamma), strategy.C)
        initialize!(m, init_strategy)
        set_adjust_K!(m, true)
        try
            fit!(m, strategy.solver)
        catch e
            debug(LOGGER, "[Gamma Search] Fitting failed for gamma $gamma.")
            println(e)
            continue
        end
        predictions = classify.(predict(m, hcat(m.data, data_target, data_outliers)));
        score = strategy.scoring_function(ground_truth, predictions)
        if score > best_score
            debug(LOGGER, "[Gamma Search] New best found with gamma = $gamma and score = $score.")
            best_gamma = gamma
            best_score = score
        end
    end
    return best_gamma
end
