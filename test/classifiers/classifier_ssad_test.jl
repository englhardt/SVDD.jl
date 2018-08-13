
@testset "SSAD" begin

    dummy_data = reshape(collect(1:20), 2, 10)
    pools = fill(:U, 10)
    ssad = SVDD.SSAD(dummy_data, pools)

    @testset "create" begin
        @test ssad.state == SVDD.model_created
        @test_throws SVDD.ModelStateException SVDD.predict(ssad, dummy_data)
    end

    @testset "initialize" begin
        C1 = C2 = 1.0
        gamma = 0.5
        init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(gamma), (C1, C2))

        SVDD.initialize!(ssad, init_strategy)
        @test ssad.state == SVDD.model_initialized
        @test ssad.data == dummy_data
        @test ssad.C1 ≈ 1.0
        @test ssad.C2 ≈ 1.0
        @test size(ssad.K) == (10, 10)
        @test all(diag(ssad.K) .≈ 1.0)
    end

    @testset "fit" begin
        SVDD.fit!(ssad, TEST_SOLVER)
        @test ssad.state == SVDD.model_fitted
        @test length(ssad.alpha_values) == size(dummy_data,2)
        @test ssad.ρ > 0.0
    end

    @testset "change pools 1" begin
        pools[1] = :Lin
        SVDD.set_pools!(ssad, pools)
        @test ssad.state == SVDD.model_initialized
        SVDD.fit!(ssad, TEST_SOLVER)
        @test ssad.state == SVDD.model_fitted
    end

    @testset "change pools 2" begin
        pools = fill(:U, 10)
        pools[1] = :Lout
        SVDD.set_pools!(ssad, pools)
        @test ssad.state == SVDD.model_initialized
        SVDD.fit!(ssad, TEST_SOLVER)
        @test ssad.state == SVDD.model_fitted
    end

    @testset "change pools 3" begin
        pools = fill(:U, 10)
        pools[1] = :Lout
        pools[10] = :Lin
        SVDD.set_pools!(ssad, pools)
        @test ssad.state == SVDD.model_initialized
        SVDD.fit!(ssad, TEST_SOLVER)
        @test ssad.state == SVDD.model_fitted
    end

    @testset "predict" begin
        predictions = SVDD.predict(ssad, dummy_data)
        @test length(predictions) == size(dummy_data, 2)
        @test predictions[1] == SVDD.predict(ssad, [1.0 2.0]')[1]
    end

    @testset "params" begin
        @test all(map(x->SVDD.is_valid_param_value(ssad, Val{:C1}, x), [0.0, 0.1, 1.0]))
        @test all(map(x->SVDD.is_valid_param_value(ssad, Val{:C2}, x), [0.0, 0.1, 1.0]))
        @test !any(map(x->SVDD.is_valid_param_value(ssad, Val{:C1}, x), [-0.1, 1.1]))
        @test !any(map(x->SVDD.is_valid_param_value(ssad, Val{:C2}, x), [-0.1, 1.1]))
    end
end
