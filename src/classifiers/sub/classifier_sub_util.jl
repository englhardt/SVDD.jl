
macro eachsubspace(expr)
    args = esc.(expr.args)
    sub_idx = :(eachindex($(args[2]).subspaces))
    return :( [$(args[1])($(args[2:end,]...), k) for k in $sub_idx] )
end
