using ForwardDiff
using Gridap
using GridapODEs.ODETools
using GridapODEs.TransientFETools

e = VectorValue(1.0)

u(x,t) = 1.0
u(t) = x -> u(x,t)
f(t) = x -> ∂t(u)(x,t) - Δ(u(t))(x) - e⋅∇(u(t))(x)

domain = (0,1)
cells = (10,)
model = CartesianDiscreteModel(domain, cells)

order = 1
V = TestFESpace(model,
		ReferenceFE(lagrangian, Float64, order),
		conformity=:H1,
		dirichlet_tags=1)
U = TransientTrialFESpace(V, u)

degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

a(t,u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫((e⊙u)⋅∇(v))dΩ
b(t,v) = ∫(v⋅f(t))dΩ
m(t,ut,v) = ∫(ut⋅v)dΩ

op = TransientAffineFEOperator(m, a, b, U, V)

t₀ = 0.0
t₁ = 1.0
δt = 0.1

u₀(x) = 0.0
U₀ = U(0.0)
uh₀ = interpolate_everywhere(u₀, U₀)

ls = LUSolver()
θ = 1.0
ode_solver = ThetaMethod(ls, δt, θ)

sol_t = solve(ode_solver, op, uh₀, t₀, t₁)
using Plots
using Printf
plt = plot()
for (uh_tn, tn) in sol_t
	plot!(plt, uh_tn.free_values[1:10], label = @sprintf "t = %1.1f" tn)
end
plot(plt)
