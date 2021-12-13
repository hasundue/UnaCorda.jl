# Parameters
θr = 0
θs = 1
α = 0.1
n = 2
m = 1 - 1/n
ks = 1

θw(u) = θr + (θs - θr) / (1 + (α*u)^n)^m
Cw(u) = ForwardDiff.derivative(θw, u)
kw(u) = ks * ((1 - (α*u)^(n-1)*(1+(α*u)^n)^(-m)) / (1 + (α*u)^n)^(m/2))^2

# plot(θw, xscale = :log10)
