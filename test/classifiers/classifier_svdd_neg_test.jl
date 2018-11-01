
@testset "SVDDneg" begin
    dummy_data, labels = generate_mvn_with_outliers(2, 100, 42, true, true)
    pools = fill(:U, size(dummy_data, 2))
    svdd_neg = SVDD.SVDDneg(dummy_data, pools)

    @testset "create" begin
        @test svdd_neg.state == SVDD.model_created
        @test_throws SVDD.ModelStateException SVDD.predict(svdd_neg, dummy_data)
    end
    C1 = C2 = 0.1
    gamma = 0.5
    init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(gamma), (C1, C2))

    @testset "initialize" begin

        SVDD.initialize!(svdd_neg, init_strategy)
        @test svdd_neg.state == SVDD.model_initialized
        @test svdd_neg.data == dummy_data
        @test svdd_neg.C1 ≈ 0.1
        @test svdd_neg.C2 ≈ 0.1
        @test size(svdd_neg.K) == (size(dummy_data, 2), size(dummy_data, 2))
        @test all(diag(svdd_neg.K) .≈ 1.0)
    end

    @testset "fit" begin
        SVDD.fit!(svdd_neg, TEST_SOLVER)
        @test svdd_neg.state == SVDD.model_fitted
        @test length(svdd_neg.alpha_values) == size(dummy_data, 2)
        @test svdd_neg.R > 0.0
        @test svdd_neg.alpha_values == SVDD.get_alpha_prime(svdd_neg)
    end

    @testset "predict" begin
        vanilla_svdd = SVDD.VanillaSVDD(dummy_data)
        SVDD.initialize!(vanilla_svdd, SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(gamma), C1))
        SVDD.fit!(vanilla_svdd, TEST_SOLVER)
        expected = SVDD.predict(vanilla_svdd, dummy_data)

        actual = SVDD.predict(svdd_neg, dummy_data)

        @test_broken expected ≈ actual
        @test sum(actual .> 0) == sum(labels .== "outlier")
    end

    @testset "with outlier examples" begin
        C1 = C2 = 0.5
        gamma = 0.5
        init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(gamma), (C1, C2))
        SVDD.initialize!(svdd_neg, init_strategy)
        SVDD.fit!(svdd_neg, TEST_SOLVER)
        predictions = SVDD.predict(svdd_neg, dummy_data)
        @test sum(predictions .> 0) == 0
        pools[labels .== "outlier"] .= :Lout

        SVDD.set_pools!(svdd_neg, pools)
        @test_throws SVDD.ModelStateException SVDD.predict(svdd_neg, dummy_data)
        SVDD.fit!(svdd_neg, TEST_SOLVER)
        predictions = SVDD.predict(svdd_neg, dummy_data)
        @test sum(predictions .> 0) == 8

        @test any(SVDD.get_alpha_prime(svdd_neg) .< 0)
        @test all(svdd_neg.alpha_values .>= 0)
    end

    @testset "with inlier examples" begin
        C1 = C2 = 0.05
        gamma = 4.0
        init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(gamma), (C1, C2))
        SVDD.initialize!(svdd_neg, init_strategy)
        SVDD.fit!(svdd_neg, TEST_SOLVER)
        expected = SVDD.predict(svdd_neg, dummy_data)
        @test sum(expected .> 0) > 0

        pools[labels .== "inlier"] .= :Lin
        SVDD.set_pools!(svdd_neg, pools)

        SVDD.fit!(svdd_neg, TEST_SOLVER)
        actual = SVDD.predict(svdd_neg, dummy_data)
        @test expected ≈ actual
        @test svdd_neg.alpha_values ≈ SVDD.get_alpha_prime(svdd_neg)
    end

    @testset "params" begin
        @test all(map(x->SVDD.is_valid_param_value(svdd_neg, Val{:C1}, x), [0.0, 0.1, 1.0]))
        @test all(map(x->SVDD.is_valid_param_value(svdd_neg, Val{:C2}, x), [0.0, 0.1, 1.0]))
        @test !any(map(x->SVDD.is_valid_param_value(svdd_neg, Val{:C1}, x), [-0.1, 1.1]))
        @test !any(map(x->SVDD.is_valid_param_value(svdd_neg, Val{:C2}, x), [-0.1, 1.1]))
    end
end
