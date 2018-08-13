
function generate_mvn_with_outliers(n_dim, n_observations,
    seed=123, normalized=true, incl_outliers=true)

    srand(seed)
    norm_distribution = MvNormal(zeros(n_dim), eye(n_dim))
    inliers = rand(norm_distribution, n_observations)
    tmp = [rand(MvNormal([x, y], eye(2)), 2) for x in [4,-4] for y in [4,-4]]
    outliers = vcat(hcat(tmp...), zeros(n_dim - 2, 8))

    if incl_outliers
        x = hcat(inliers, outliers)
        labels = vcat(fill("inlier", n_observations), fill("outlier", 8))
    else
        x = inliers
        labels = vcat(fill("inlier", n_observations))
    end
    if normalized
        x = mapslices(normalize, x, 2)
    end
    return (x, labels)
end
