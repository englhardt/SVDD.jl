
@testset "VanillaSVDD" begin

    dummy_data = [1.0 2.0 3.0; 4.0 5.0 6.0]
    vanilla_svdd = SVDD.VanillaSVDD(dummy_data)

    @testset "create" begin
        @test vanilla_svdd.state == SVDD.model_created
        @test_throws SVDD.ModelStateException SVDD.predict(vanilla_svdd, dummy_data)
    end

    @testset "initialize" begin
        init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(0.5), 0.5)

        SVDD.initialize!(vanilla_svdd, init_strategy)
        @test vanilla_svdd.state == SVDD.model_initialized
        @test vanilla_svdd.data == dummy_data
        @test vanilla_svdd.C ≈ 0.5
        @test size(vanilla_svdd.K) == (3,3)
        @test all(diag(vanilla_svdd.K) .≈ 1.0)
    end

    @testset "fit" begin
        SVDD.fit!(vanilla_svdd, TEST_SOLVER)
        @test vanilla_svdd.state == SVDD.model_fitted
        @test length(vanilla_svdd.alpha_values) == size(dummy_data,2)
        @test vanilla_svdd.R > 0.0
        @test vanilla_svdd.alpha_values == SVDD.get_alpha_prime(vanilla_svdd)
    end

    @testset "predict" begin
        predictions = SVDD.predict(vanilla_svdd, dummy_data)
        @test length(predictions) == size(dummy_data, 2)
        @test predictions[1] == SVDD.predict(vanilla_svdd, [1.0 4.0]')[1]
    end

    @testset "params" begin
         @test all(map(x->SVDD.is_valid_param_value(vanilla_svdd, Val{:C}, x), [0.0, 0.1, 1.0]))
         @test !any(map(x->SVDD.is_valid_param_value(vanilla_svdd, Val{:C}, x), [-0.1, 1.1]))
     end
end
