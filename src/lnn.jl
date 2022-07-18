
struct LagrangianNN{M} <: Lux.AbstractExplicitContainerLayer{(:nn,)}
    nn::M
    LagrangianNN(m::Lux.AbstractExplicitLayer) = new{typeof(m)}(m)
end

using ReverseDiff
function (lnn::LagrangianNN)(x::AbstractVector, ps, st)
    #grad_f(x) = FiniteDifferences.grad(central_fdm(5, 1), y -> sum(lnn.nn(y, ps, st)[1]), x)[1]
    #grad_f(x) = gradient(y -> sum(lnn.nn(y, ps, st)[1]), x)[1]
   # grad = grad_f(x)
    #hess = FiniteDiff.finite_difference_jacobian(grad_f, x)
    #grad = FiniteDiff.finite_difference_jacobian(y -> sum(lnn.nn(y, ps, st)[1]), x)
    grad = gradient(y -> sum(lnn.nn(y, ps, st)[1]), x)[1]
    hess = hessian(y -> sum(lnn.nn(y, ps, st)[1]), x)
    #hess = FiniteDiff.finite_difference_hessian(y -> sum(lnn.nn(y, ps, st)[1]), x).data
    #hess = hessian(y -> sum(lnn.nn(y, ps, st)[1]), x)

    m = size(x, 1) ÷ 2
    q_t = x[(m + 1):end]
    q_tt = hess[m+1:end, m+1:end] * (grad[1:m] .- hess[1:m, (m+1):end] * q_t)

    return [q_t; q_tt], st
end

function (lnn::LagrangianNN)(x, ps, st)
    ps = NamedTuple(ps)

    grad = gradient(x -> sum(lnn.nn(x, ps, st)[1]), x)[1]
    hess = map(x-> hessian(y -> sum(lnn.nn(y, ps, st)[1]), x), eachcol(x))

    m = size(x, 1) ÷ 2
    q_t = x[(m + 1):end,:]
    q_tt = map(hess, eachcol(grad), eachcol(q_t)) do hess, grad, q_t
        hess[m+1:end, m+1:end] * (grad[1:m] .- hess[1:m, (m+1):end] * q_t)
    end
    q_tt = reduce(hcat, q_tt)

    return [q_t; q_tt], st
end

struct NeuralLagrangianDE{M, T, A, K, S} <: NeuralDELayer
    model::LagrangianNN{M}
    tspan::T
    args::A
    kwargs::K
    sensealg:: S

    function NeuralLagrangianDE(model::M, tspan, args...;
                                sensealg = InterpolatingAdjoint(autojacvec = false), kwargs...) where {M<:Lux.AbstractExplicitLayer}
        lnn = LagrangianNN(model)
        new{M, typeof(tspan), typeof(args), typeof(kwargs), typeof(sensealg)}(lnn, tspan, args, kwargs, sensealg)
    end

    function NeuralLagrangianDE(lnn::LagrangianNN{M}, tspan, args...; kwargs...) where {M}
        new{M, typeof(tspan), typeof(args), typeof(kwargs), typeof(sensealg)}(lnn, tspan, args, kwargs, sensealg)
    end
end

function (ngde::NeuralLagrangianDE)(x, ps, st)
    function dudt(u, p, t; st = st)
        u_, st = ngde.model(u, p, st)
        return u_
    end

    ff = ODEFunction{false}(dudt, tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, ngde.tspan, ps)
    solve(prob, ngde.args...; sensealg = ngde.sensealg, ngde.kwargs...), st
end
