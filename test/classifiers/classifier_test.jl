using MLKernels, MLLabelUtils

@testset "oc_classifier" begin
    dummy_data = [1.0 2.0 3.0; 4.0 5.0 6.0]
    pools = fill(:U, size(dummy_data, 2))

    init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(0.5), 0.5)

    models = []
    svdd_neg = SVDD.SVDDneg(dummy_data, pools)
    SVDD.initialize!(svdd_neg, init_strategy)
    SVDD.fit!(svdd_neg, TEST_SOLVER)
    push!(models, svdd_neg)

    vanilla_svdd = SVDD.VanillaSVDD(dummy_data)
    SVDD.initialize!(vanilla_svdd, init_strategy)
    SVDD.fit!(vanilla_svdd, TEST_SOLVER)
    push!(models, vanilla_svdd)

    ssad = SVDD.SSAD(dummy_data, pools)
    SVDD.initialize!(ssad, init_strategy)
    SVDD.fit!(ssad, TEST_SOLVER)
    push!(models, ssad)

    update_data = [3.0 2.0 3.0; 4.0 5.0 6.0]
    pools_updated = copy(pools)
    pools_updated[1] = :Lout

    @testset "instantiate" begin
        actual = SVDD.instantiate(SVDD.VanillaSVDD, dummy_data, pools, Dict(:C => 0.5))
        @test actual.C == 0.5
        actual = SVDD.instantiate(SVDD.SVDDneg, dummy_data, pools, Dict(:C1 => 0.5))
        @test actual.C1 == 0.5
        actual = SVDD.instantiate(SVDD.SSAD, dummy_data, pools, Dict(:C1 => 0.5))
        @test actual.C1 == 0.5
    end

    @testset "set_param!" begin
        actual = deepcopy(vanilla_svdd)
        @test_throws ArgumentError SVDD.set_param!(actual, Dict(:C => "0.4", :Ψ => "ipd"))
        @test_throws MethodError SVDD.set_param!(actual, Dict(:C => "0.4"))
        SVDD.set_param!(actual, Dict(:C => 0.5))
        @test actual.state == SVDD.model_fitted
        SVDD.set_param!(actual, Dict(:C => 0.4))
        @test actual.state == SVDD.model_initialized
        @test actual.C == 0.4
        @test_throws ArgumentError SVDD.set_param!(actual, Dict(:C => 1.1))
    end

    for m in models
        @testset "get_kernel $(typeof(m))" begin
            actual = SVDD.get_kernel(m)
            if isdefined(m, :kernel_fct)
                @test m.kernel_fct == actual
            end
        end
    end

    for m in models
        @testset "set_adjust_kernel $(typeof(m))" begin
            actual = deepcopy(m)
            @assert actual.state == SVDD.model_fitted
            @assert !SVDD.is_K_adjusted(actual)
            SVDD.set_adjust_K!(actual, false)
            @test actual.state == SVDD.model_fitted
            @test !isdefined(actual, :K_adjusted)
            SVDD.set_adjust_K!(actual, true)
            @test isdefined(actual, :K_adjusted)
            @test size(actual.K) == size(actual.K_adjusted)
            @test actual.state == SVDD.model_initialized
        end
    end

    for m in models
        @testset "model updates $(typeof(m))" begin
            @testset "no change" begin
                actual = deepcopy(m)
                @assert actual.state == SVDD.model_fitted
                SVDD.set_pools!(actual, pools)
                SVDD.set_pools!(actual, labelmap(pools))
                @test actual.state == SVDD.model_fitted
                SVDD.update_K!(actual)
                @test actual.state == SVDD.model_fitted
                SVDD.set_data!(actual, actual.data)
                @test actual.state == SVDD.model_fitted
                SVDD.set_kernel!(actual, actual.kernel_fct)
                @test actual.state == SVDD.model_fitted
            end

            @testset "invalidate model after change" begin
                actual = deepcopy(m)
                SVDD.set_data!(actual, update_data)
                @test actual.state == SVDD.model_initialized
                @test actual.data == update_data

                if (typeof(actual) <: SVDD.SSAD || typeof(actual) <: SVDD.SVDDneg)
                    actual = deepcopy(m)
                    SVDD.set_pools!(actual, pools_updated)
                    @test actual.state == SVDD.model_initialized

                    actual = deepcopy(m)
                    SVDD.set_pools!(actual, labelmap(pools_updated))
                    @test actual.state == SVDD.model_initialized
                end

                actual = deepcopy(m)
                SVDD.set_kernel!(actual, LinearKernel())
                @test actual.state == SVDD.model_initialized
                @test actual.K != m.K
                @test actual.kernel_fct == LinearKernel()
            end
        end
    end
end
