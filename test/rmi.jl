using ForwardDiff
using Gridap
using GridapODEs.ODETools
using GridapODEs.TransientFETools

Ks = 0.001
α = 0.01
m = 0.5
θs = 0.5
θr = 0.1

e = VectorValue(1.0)

u(x,t) = θs
u(t) = x -> u(x,t)
f(t) = x -> ∂t(u)(x,t) - D(u(t,x))*Δ(u(t))(x) - K(u(t,x))*∇(u(t))(x)⋅e
D(u) = (1-m)*Ks/(α*m)/(θs-θr) * u^(1/2-1/m) * ((1-u^(1/m))^m + (1-u^(1/m))^(-m) - 2)
K(u) = Ks * u^(1/2) * (1 - (1 - u^(1/m))^m)^2

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

a(u,v) = ∫((D∘u)*∇(u)⋅∇(v))dΩ + ∫((e⊙(K∘u))⋅∇(v))dΩ
b(v,t) = ∫(v⋅f(t))dΩ

res(t,u,v) = a(u,v) + ∫(∂t(u)*v)dΩ - b(v,t)
jac(t,u,du,v) = a(du,v)
jac_t(t,u,dut,v) = ∫(dut*v)dΩ

op = TransientFEOperator(res, jac, jac_t, U, V)

t₀ = 0.0
t₁ = 10.0
δt = 1.0

u₀(x) = θr
U₀ = U(0.0)
uh₀ = interpolate_everywhere(u₀, U₀)

nls = NLSolver()
θ = 1.0
ode_solver = ThetaMethod(nls, δt, θ)

sol_t = solve(ode_solver, op, uh₀, t₀, t₁)
using Plots
using Printf
plt = plot()
for (uh_tn, tn) in sol_t
	plot!(plt, uh_tn.free_values[1:10], label = @sprintf "t = %1.1f" tn)
end
plot(plt)
