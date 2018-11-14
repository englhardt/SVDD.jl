abstract type WeightUpdateStrategy end

# cf. https://discourse.julialang.org/t/broadcasting-structs-as-scalars/14310
Base.Broadcast.broadcastable(ws::WeightUpdateStrategy) = Ref(ws)

struct NoUpdateStrategy <: WeightUpdateStrategy end

update_v(v, label::Symbol, update_strategy::NoUpdateStrategy) = v

struct FixedWeightStrategy <: WeightUpdateStrategy
    v_Lin::Float64
    v_Lout::Float64
end

function update_v(v, label::Symbol, update_strategy::FixedWeightStrategy)
    if label == :Lin
        return update_strategy.v_Lin
    elseif label == :Lout
        return update_strategy.v_Lout
    else
        return v
    end
end

macro eachsubspace(expr)
    args = esc.(expr.args)
    sub_idx = :(eachindex($(args[2]).subspaces))
    return :( [$(args[1])($(args[2:end,]...), k) for k in $sub_idx] )
end
