using ForwardDiff
using Gridap
using GridapODEs.ODETools
using GridapODEs.TransientFETools

f(t) = 1.0

u₀(x) = 0.0

domain = (0,1)
cells = (10,)
model = CartesianDiscreteModel(domain, cells)

δ = DiracDelta{0}(model, tags=1)

order = 2
V = TestFESpace(model,
		ReferenceFE(lagrangian, Float64, order),
		conformity=:H1)
U = TransientTrialFESpace(V)

degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

a(u,v) = ∫(∇(v)⋅∇(u))dΩ
b(v,t) = δ(f(t)*v)

res(t,u,v) = a(u,v) + ∫(∂t(u)*v)dΩ - b(v,t)
jac(t,u,du,v) = a(du,v)
jac_t(t,u,dut,v) = ∫(dut*v)dΩ

op = TransientFEOperator(res, jac, jac_t, U, V)

t₀ = 0.0
t₁ = 1.0
δt = 0.1

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
	plot!(plt, uh_tn.free_values[1:9], label = @sprintf "t = %1.1f" tn)
end
plot(plt)
