
@testset "SVDD_util_test" begin

    @testset "merge pool" begin
        pools = labelmap([:U, :Lin, :Lout, :U, :U, :Lin])

        @test [] == SVDD.merge_pools(pools, [])

        @test [1,4,5] == SVDD.merge_pools(pools, :U)
        @test [1,4,5] == sort(SVDD.merge_pools(pools, :U, :U))
        @test [2,6] == SVDD.merge_pools(pools, :Lin)
        @test [3] == SVDD.merge_pools(pools, :Lout)

        @test [2,3,6] == sort(SVDD.merge_pools(pools, :Lin, :Lout))
        @test collect(1:6) == sort(SVDD.merge_pools(pools, :U, :Lin, :Lout))

        @test_throws ArgumentError SVDD.merge_pools(pools, :A)
    end

    @testset "classify" begin
        scores = [2.0, 0.0, 0.01, -0.1, -5.0]
        expected = [:outlier, :inlier, :outlier, :inlier, :inlier]
        actual = SVDD.classify.(scores)
        @test MLLabelUtils.islabelenc(actual, SVDD.class_label_enc)
        @test expected == actual
    end

    @testset "adjust kernel" begin
        K1 = [2 -1 0; -1 2 -1; 0 -1 2]
        @assert all(eig(K1)[1] .> 0)
        @test K1 ≈ SVDD.adjust_kernel_matrix(K1)

        K2 = [1 2; 2 1]
        @assert any(eig(K2)[1] .< 0)
        K2_adjusted = SVDD.adjust_kernel_matrix(K2, warn_threshold = 2)
        @test !(K2 ≈ K2_adjusted)
        @test all(eig(K2_adjusted)[1] .>= 0.0)

        srand(42)
        dummy_data, _ = generate_mvn_with_outliers(2, 100, 42, true, true)
        model = SVDD.VanillaSVDD(dummy_data)
        init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(4), 0.1)
        SVDD.initialize!(model, init_strategy)
        K_old = copy(model.K)
        @assert any(eig(model.K)[1] .< 0)

        @test_throws ArgumentError SVDD.adjust_kernel_matrix([3 -2; 4 -1])
    end
end
