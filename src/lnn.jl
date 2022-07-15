
struct LagrangianNN{M} <: NeuralDELayer
    model::M
end

function (lnn::LagrangianNN)(x, p, st)
    function lagrangian(q, q_t)
        lg, st = lnn.model([q; q_t], p, st)
        return lg
    end

    n = size(x, 1) ÷ 2
    q, q_t = x[1:n], x[(n + 1):end]
    q_tt = pinv(hessian(Base.Fix1(lagrangian, q), q_t)) * (gradient(lagrangian, q, q_t)[1] -
            jacobian((q, q_t) -> jacobian(lagrangian, q, q_t)[2], q, q_t)[1] * q_t)

    return [q_t; q_tt]
end

struct NeuralLagrangianDE{M, T, A, K} <: NeuralDELayer
    model::LagrangianNN{M}
    tspan::T
    args::A
    kwargs::K

    function NeuralLagrangianDE(model::Lux.AbstractExplicitLayer, tspan, args...; kwargs...)
        lnn = LagrangianNN(model)
        new{typeof{model}, typeof{tspan}, typeof{args}, typeof{kwargs}}(lnn,
                                                                        tspan,
                                                                        args,
                                                                        kwargs)
    end

    function NeuralLagrangianDE(lnn::LagrangianNN{M}, tspan, args...; kwargs...)
        new{M, typeof{tspan}, typeof{args}, typeof{kwargs}}(lnn, tspan, args, kwargs)
    end
end

function (ngde::NeuralLagrangianDE)(x, p, st)
    function dudt(u, p, t; st = st)
        u_, st = ngde.model(u, p, st)
        return u_
    end

    ff = ODEFunction{false}(dudt_, tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, ngde.tspan, p)
    sense = InterpolatingAdjoint(autojacvec = false)
    solve(prob, ngde.args...; sensealg = sense, ngde.kwargs...)
end
