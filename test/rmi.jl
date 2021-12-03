module RMI

using Plots
using ModelingToolkit
using DomainSets
using DiffEqOperators
using OrdinaryDiffEq

@variables t x
@variables u(..)
@parameters β
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# van Genuchten model
# S(t,x) = (1 + (-α*u(t,x))^n)^(-m)
# K(t,x) = Ks * S(t,x)^l * (1 - (1 - S(t,x)^(1/m))^m)^2
K(t,x) = β * u(t,x)

# Parameters
α = 14.5
Ks = 0.297
m = 0.63
n = 1 / (1-m)
l = 1/2

# Richards equation
# eqs = [ Dt(S(t,x)) ~ Dx( K(t,x) * Dx(u(t,x)) ) + Dx(K(t,x)) ]
eqs = [ Dt(u(t,x)) ~ Dxx(u(t,x)) + Dx(K(t,x)) ]
bcs = [ u(0,x) ~ -1.0,
        u(t,0) ~ 0.0,
	Dx(u(t,1)) ~ 0.0 ]

domains = [ t ∈ Interval(0.0, 1.0),
	    x ∈ Interval(0.0, 1.0) ]

@named pdesys = PDESystem(eqs, bcs, domains, [t,x], [u(t,x)], [β => 1.0])

dx = 0.1
discretization = MOLFiniteDifference([x => dx], t)

prob = discretize(pdesys, discretization)

sol = solve(prob, Tsit5(), saveat=0.1)

plot(sol[end])

end
