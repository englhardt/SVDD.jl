
@testset "SubSVDD" begin
    dummy_data, labels = generate_mvn_with_outliers(4, 100, 42, true, true)
    pools = fill(:U, size(dummy_data, 2))
    subspaces = [[1, 2], [3, 4]]

    model = SubSVDD(dummy_data, subspaces, pools)

    @testset "create" begin
        @test model.state == SVDD.model_created
        # @test_throws SVDD.ModelStateException SVDD.predict(model, dummy_data)
    end
    C = 0.1
    gamma = 0.5
    init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(gamma), C)

    @testset "initialize" begin
        SVDD.initialize!(model, init_strategy)
        @test model.state == SVDD.model_initialized
        @test model.data == dummy_data
        @test model.C ≈ C
        @test length(model.K) == length(subspaces)
        @test size(model.K[1]) == (size(dummy_data, 2), size(dummy_data, 2))
        @test all(map(x -> all(diag(x) .≈ 1.0), model.K))
    end

    @testset "fit" begin
        SVDD.fit!(model, TEST_SOLVER)
        @test model.state == SVDD.model_fitted
        @test length(model.alpha_values) == length(subspaces)
        @test all(length.(model.alpha_values) .== size(model.data, 2))
    end
end
