
@testset "SubSVDD" begin
    dummy_data, labels = generate_mvn_with_outliers(4, 100, 42, true, true)
    pools = fill(:U, size(dummy_data, 2))
    subspaces = [[1, 2], [3, 4]]

    model = SubSVDD(dummy_data, subspaces, pools)

    @testset "create" begin
        @test model.state == SVDD.model_created
        # @test_throws SVDD.ModelStateException SVDD.predict(model, dummy_data)
    end
    C = 1.0
    gamma = 0.5
    init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(gamma), C)

    @testset "initialize" begin
        SVDD.initialize!(model, init_strategy)
        @test !(all(model.K[1] .≈ model.K[2]))
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
        @test all(model.R .> 0.0)
        @test all(model.const_term .> 0.0)
    end

    @testset "calculate_c_delta" begin
        @test SVDD.calculate_upper_limit(model.alpha_values, 1, model.C) ≈ model.C .- model.alpha_values[2]
        @test SVDD.calculate_upper_limit(model.alpha_values, 2, model.C) ≈ model.C .- model.alpha_values[1]
        @test SVDD.calculate_upper_limit(model.alpha_values, 2, model.C, fill(0.5, size(model.data, 2))) ≈ 0.5*model.C .- model.alpha_values[1]
    end

    @testset "@each_subspace" begin

        @testset "generic function" begin
            g(m, subspace_idx) = subspace_idx
            expected = [1, 2]
            actual = @eachsubspace g(model)
            @test expected == actual
        end

        @testset "predict" begin
            expected = @eachsubspace SVDD.predict(model)
            actual = [SVDD.predict(model, model.data, k) for k in eachindex(model.subspaces)]
            @test all(expected .≈ actual)
        end
    end

    @testset "predict" begin
        predictions = @eachsubspace SVDD.predict(model, model.data)
        @test length(predictions) == 2
        @test all(length.(predictions) .== size(model.data, 2))
        @test all(map(x -> all(x .< 1e-5), predictions))
    end


end
