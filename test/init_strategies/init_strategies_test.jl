
@testset "initialization strategies" begin
    n = 100
    d = 5
    dummy_data, labels = generate_mvn_with_outliers(d, n, 42, true, false)
    pools = fill(:U, size(dummy_data, 2))
    model = SVDD.VanillaSVDD(dummy_data)

    for s in [:RuleOfThumbSilverman, :RuleOfThumbScott]
        @testset "$s" begin
            @test SVDD.calculate_gamma(model, SVDD.eval(s)()) > 0
        end
    end

    @testset "TaxErrorEstimate" begin
        # see Tax and Duin 2004, Support Vector Data Description
        outlier_percentage = .1
        expected = 1 / (n*outlier_percentage)
        @test expected == SVDD.calculate_C(model, SVDD.TaxErrorEstimate(outlier_percentage))
    end

    @testset "BoundedTaxErrorEstimate-1" begin
        n = 10
        dummy_data_small, _ = generate_mvn_with_outliers(d, n, 42, true, false)
        lower_bound = 0.03
        upper_bound = 0.97
        outlier_percentage = 0.05
        @test upper_bound ≈ SVDD.calculate_C(SVDD.VanillaSVDD(dummy_data_small), SVDD.BoundedTaxErrorEstimate(outlier_percentage, lower_bound, upper_bound))
    end

    @testset "BoundedTaxErrorEstimate-2" begin
        n = 2000
        dummy_data_small, _ = generate_mvn_with_outliers(d, n, 42, true, false)
        lower_bound = 0.03
        upper_bound = 0.97
        outlier_percentage = 0.05
        @test lower_bound ≈ SVDD.calculate_C(SVDD.VanillaSVDD(dummy_data_small), SVDD.BoundedTaxErrorEstimate(outlier_percentage, lower_bound, upper_bound))
    end

    @testset "SimpleCombinedStrategy" begin
        gamma_expected = SVDD.calculate_gamma(model, SVDD.RuleOfThumbScott())
        C_expected = 0.5
        init_strategy = SVDD.SimpleCombinedStrategy(SVDD.RuleOfThumbScott(), SVDD.FixedCStrategy(C_expected))
        C, kernel = SVDD.get_parameters(model, init_strategy)
        @test C == C_expected
        @test kernel == MLKernels.GaussianKernel(gamma_expected)
    end

    @testset "BinarySearchCStrategy" begin
        model = SVDD.VanillaSVDD(dummy_data)

        target_outlier_percentage = 0.1
        C_init = 0.5
        max_iter = 10
        eps = 0.01

        strategy = SVDD.BinarySearchCStrategy(target_outlier_percentage, C_init, max_iter, eps, TEST_SOLVER)
        gamma = 5.0
        C_actual = SVDD.calculate_C(model, strategy, GaussianKernel(gamma))

        init_strategy = SVDD.FixedParameterInitialization(GaussianKernel(gamma), C_actual)

        SVDD.initialize!(model, init_strategy)
        SVDD.fit!(model, TEST_SOLVER)
        actual_outlier_percentage = countmap(SVDD.classify.(SVDD.predict(model, model.data)))[:outlier] / size(model.data, 2)
        @test abs.(actual_outlier_percentage - target_outlier_percentage) < 0.04
    end

    @testset "GammaFirstCombinedStrategy" begin
        model = SVDD.SVDDneg(dummy_data, pools)
        target_outlier_percentage = 0.15

        gamma_strategy = SVDD.RuleOfThumbSilverman()
        C_strategy = SVDD.BinarySearchCStrategy(target_outlier_percentage, 0.5, 10, 0.01, TEST_SOLVER)
        init_strategy = SVDD.GammaFirstCombinedStrategy(gamma_strategy, C_strategy)

        SVDD.initialize!(model, init_strategy)
        SVDD.fit!(model, TEST_SOLVER)
        actual_outlier_percentage = countmap(SVDD.classify.(SVDD.predict(model, model.data)))[:outlier] / size(model.data, 2)
        @test abs.(actual_outlier_percentage - target_outlier_percentage) < 0.04
    end

    @testset "SimpleSubspaceStrategy" begin
        subspaces = [[1, 2], [3, 4, 5]]
        model = SVDD.SubSVDD(dummy_data, subspaces, pools)
        for gamma_strategy in [RuleOfThumbScott(), RuleOfThumbSilverman()]
            @testset "RuleOfThumb-Global-$(typeof(gamma_strategy))" begin
                    strategy = SVDD.SimpleSubspaceStrategy(
                        gamma_strategy,
                        FixedCStrategy(1.0),
                        gamma_scope = Val(:Global))
                    C, kernel = SVDD.get_parameters(model, strategy)
                    @test C ≈ 1.0
                    @test length(kernel) == 2
                    @test all(map(x -> x <: MLKernels.GaussianKernel, typeof.(kernel)))
                    @test kernel[1] == kernel[2]
                end
            @testset "RuleOfThumb-Local-$(typeof(gamma_strategy))" begin
                strategy = SVDD.SimpleSubspaceStrategy(
                    gamma_strategy,
                    FixedCStrategy(1.0),
                    gamma_scope = Val(:Subspace))
                C, kernel = SVDD.get_parameters(model, strategy)
                @test C ≈ 1.0
                @test length(kernel) == 2
                @test all(map(x -> x <: MLKernels.GaussianKernel, typeof.(kernel)))
                @test kernel[1] != kernel[2]
            end
        end
    end


    dummy_data, labels = generate_mvn_with_outliers(2, 50, 42, true, true)
    pools = fill(:Lin, size(dummy_data, 2))

    @testset "WangGammaStrategy" begin
        model = SVDD.VanillaSVDD(dummy_data[:, labels .== :inlier])
        gamma_strategy = SVDD.WangGammaStrategy(TEST_SOLVER, [0.1, 0.5], 1)
        init_strategy = SVDD.SimpleCombinedStrategy(gamma_strategy, SVDD.FixedCStrategy(1))

        SVDD.initialize!(model, init_strategy)
        SVDD.fit!(model, TEST_SOLVER)
        @test all(SVDD.classify.(SVDD.predict(model, dummy_data)) .== labels)
    end

    @testset "WangCombinedInitializationStrategy" begin
        model = SVDD.VanillaSVDD(dummy_data[:, labels .== :inlier])
        fixed_C = 0.95
        init_strategy = SVDD.WangCombinedInitializationStrategy(TEST_SOLVER, [0.1, 0.5], SVDD.FixedCStrategy(fixed_C))

        SVDD.initialize!(model, init_strategy)
        SVDD.fit!(model, TEST_SOLVER)
        @test get_model_params(model)[:C] == fixed_C
        @test all(SVDD.classify.(SVDD.predict(model, dummy_data)) .== labels)
    end
end
