using DiffEqFlux, OrdinaryDiffEq, ReverseDiff, Test, Lux, Random

# Checks for Shapes and Non-Zero Gradients
u0 = rand(Float32, 6)

lnn = LagrangianNN(Lux.Chain(Lux.Dense(6, 12, softplus), Lux.Dense(12, 1)))
ps, st = Lux.setup(Random.default_rng(), lnn)

@test size(lnn(u0, ps, st)[1]) == (6,)
@test size(lnn(u0, Lux.ComponentArray(ps), st)[1]) == (6,)

@test ! iszero(Zygote.gradient(p -> sum(lnn(u0, p,st)[1]), ps))


# Test Convergence on a toy problem
t = range(0.0f0, 1.0f0, length = 64)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t, dims = 1)
target = cat(dqdt, dpdt, dims = 1)

lnn = LagrangianNN(Lux.Chain(Lux.Dense(2, 16, softplus), Lux.Dense(16, 1)))
ps, st = Lux.setup(Random.default_rng(), lnn)

opt = Lux.Optimisers.Adam(0.01)
st_opt = Lux.Optimisers.setup(opt, ps)
loss(x, y, p) = sum((lnn(x, p,st) .- y) .^ 2)

initial_loss = loss(data, target, ps)

epochs = 100
for epoch in 1:epochs
    gs = ReverseDiff.gradient(p -> loss(data, target, p), ps)
    Flux.Optimise.update!(opt, p, gs)
end

final_loss = loss(data, target, p)

@test initial_loss > final_loss

# Test output and gradient of NeuralHamiltonianDE Layer
tspan = (0.0f0, 1.0f0)

model = NeuralLagrangianDE(
    hnn, tspan, Tsit5(),
    save_everystep = false, save_start = true,
    saveat = range(tspan[1], tspan[2], length=10)
)
sol = Array(model(data[:, 1]))
@test size(sol) == (2, 10)

ps = Flux.params(model)
gs = Flux.gradient(() -> sum(Array(model(data[:, 1]))), ps)

@test ! iszero(gs[model.p])
