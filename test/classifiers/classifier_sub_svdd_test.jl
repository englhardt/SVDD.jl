
@testset "SubSVDD" begin
    dummy_data, labels = generate_mvn_with_outliers(4, 100, 42, true, true)
    pools = fill(:U, size(dummy_data, 2))
    subspaces = [[1, 2], [3, 4]]

    model = SubSVDD(dummy_data, subspaces, pools)

    @testset "create" begin
        @test model.state == SVDD.model_created
        @test_throws SVDD.ModelStateException SVDD.predict(model, 1)
    end
    C = 1.0
    gamma = 0.5
    gamma_strategy = FixedGammaStrategy(MLKernels.GaussianKernel(gamma))
    C_strategy = FixedCStrategy(C)
    init_strategy = SVDD.SimpleSubspaceStrategy(gamma_strategy, C_strategy, gamma_scope=Val(:Global))

    @testset "initialize" begin
        SVDD.initialize!(model, init_strategy)
        @test length(model.kernel_fct) == 2
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
        @test SVDD.calculate_upper_limit(model.alpha_values, model.C, model.v, 1) ≈ model.C .- model.alpha_values[2]
        @test SVDD.calculate_upper_limit(model.alpha_values, model.C, model.v, 2) ≈ model.C .- model.alpha_values[1]
        model.v .= 0.5
        @test SVDD.calculate_upper_limit(model.alpha_values, model.C, model.v, 2) ≈ 0.5*model.C .- model.alpha_values[1]
    end

    @testset "find_support_vectors" begin
        dummy_model = SubSVDD(dummy_data, [[1,2]], pools)
        dummy_model.alpha_values = [[0.1, 0.1-1e-10, 0.05, 0.0, 1e-10]]
        dummy_model.v = ones(Float64, 5)
        dummy_model.C = 0.1
        @test SVDD.find_support_vectors(dummy_model, 1) == [3]

        dummy_model.C = 0.2
        dummy_model.subspaces = [[1,2], [3,4]]
        push!(dummy_model.alpha_values, [0.09, 0.0, 0.0, 0.0, 0.0])
        @test SVDD.find_support_vectors(dummy_model, 1) == [1,2,3]
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
            actual = [SVDD.predict(model, k) for k in eachindex(model.subspaces)]
            @test all(expected .≈ actual)
        end
    end

    @testset "predict" begin
        predictions = @eachsubspace SVDD.predict(model)
        @test length(predictions) == 2
        @test all(length.(predictions) .== size(model.data, 2))
        @test all(map(x -> all(x .< 1e-5), predictions))
    end

    @testset "update_with_feedback" begin
        model.v .= 1.0
        expected_pools = labelmap2vec(model.pools)
        dummy_remaining_indices = collect(eachindex(expected_pools))

        @assert model.weight_update_strategy === nothing
        @test_throws ErrorException update_with_feedback!(model, model.data, expected_pools, [1], dummy_remaining_indices, dummy_remaining_indices)

        update_strategy = SVDD.FixedWeightStrategy(1.1, 0.9)
        set_param!(model, Dict(:weight_update_strategy => update_strategy))
        @test model.weight_update_strategy == update_strategy

        update_with_feedback!(model, model.data, expected_pools, Int[], dummy_remaining_indices, dummy_remaining_indices)
        @test labelmap2vec(model.pools) == expected_pools

        query_ids = [1,2]
        query_pool_labels = [:Lin, :Lout]
        expected_pools[query_ids] .= query_pool_labels
        update_with_feedback!(model, model.data, expected_pools, query_ids, dummy_remaining_indices, dummy_remaining_indices)
        @test model.v[1] ≈ 1.1
        @test model.v[2] ≈ 0.9
        @test all(model.v[3:end] .≈ 1.0)
        @test labelmap2vec(model.pools) == expected_pools
    end
end
