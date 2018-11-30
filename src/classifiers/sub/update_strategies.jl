abstract type WeightUpdateStrategy end

# cf. https://discourse.julialang.org/t/broadcasting-structs-as-scalars/14310
Base.Broadcast.broadcastable(ws::WeightUpdateStrategy) = Ref(ws)

struct NoUpdateStrategy <: WeightUpdateStrategy end

update_v(v, label::Symbol, update_strategy::NoUpdateStrategy) = v

struct FixedWeightStrategy <: WeightUpdateStrategy
    v_Lin::Float64
    v_Lout::Float64
end

function update_v(v, fb_label::Symbol, update_strategy::FixedWeightStrategy)
    if fb_label == :Lin
        return update_strategy.v_Lin
    elseif fb_label == :Lout
        return update_strategy.v_Lout
    else
        return v
    end
end

get_default_v(::FixedWeightStrategy) = one(Float64)
