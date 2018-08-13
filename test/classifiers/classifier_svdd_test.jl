
@testset "model exceptions" begin
    @test_throws SVDD.ModelStateException throw(SVDD.ModelStateException(SVDD.model_created, SVDD.model_fitted))
    @test_throws SVDD.ModelInvariantException throw(SVDD.ModelInvariantException("invariant violated"))
end
