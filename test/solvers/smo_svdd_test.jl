@testset "smo svdd" begin
    dummy_data, labels = generate_mvn_with_outliers(2, 1000, 123, true, true)
    pools = fill(:U, size(dummy_data, 2))

    init_strategy = SVDD.FixedParameterInitialization(MLKernels.GaussianKernel(0.5), 0.5)

    model = SVDD.VanillaSVDD(dummy_data)
    SVDD.initialize!(model, init_strategy)
    solver = SMOSolver(1e-4, 10)
    status = SVDD.solve!(model, solver)
    @test status == :UserLimit

    solver = SMOSolver(1e-2, 1000)
    status = SVDD.fit!(model, solver)
    @test status == JuMP.MathOptInterface.OPTIMAL
    @test model.R > 0
end
