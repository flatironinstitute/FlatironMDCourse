using LinearAlgebra
using StaticArrays
using ForwardDiff
using Plots
using Printf

# generate n points in a unit square centered at c
function points_in_unit_square(c, n)
	cs = SVector{2, Float64}(c) .- 0.5
	return rand(SVector{2, Float64}, n) .+ Ref(cs)
end
# poisson kernel evaluation
@inline function poisson_greens(r)
	return -log(norm(r)) / (2π)
end
function poisson_kernel(src, trg, σ)
	u = zeros(eltype(σ), length(trg))
	for i in eachindex(trg)
		ui = zero(eltype(σ))
		for j in eachindex(src, σ)
			r = norm(trg[i] - src[j])
			ui += poisson_greens(r) * σ[j]
		end
		u[i] = ui
	end
	return u
end

# We start by looking at the rank of the interaction between
# two well-separated sets of points
N = 1000
src = points_in_unit_square([-5.0, 0.0], N);
trg = points_in_unit_square([ 5.0, 0.0], N);
σ = randn(N);
# these are linear functions; using ForwardDiff
# is just a convenient way to construct the operators
J = ForwardDiff.jacobian(x -> poisson_kernel(src, trg, x), σ);
sv = svdvals(J);
plt = plot(sv, yscale=:log10, color=:black, linewidth=2, xlabel="Singular value index", ylabel="Singular value", label=:none)
hline!([eps(Float64)*sv[1]], label="ε * max singular value", color=:gray, linewidth=2)

# okay, now we'll try to exploit this low-rank structure in some analytical way
function circle(c, r, n)
	θ = LinRange(0, 2π, n+1)[1:end-1]
	return [SVector{2, Float64}(c[1] + r*cos(θ[i]), c[2] + r*sin(θ[i])) for i in 1:n]
end
Ncirc = 36
δfac = 0.0 # (needs to be in [0, 1))
δ = δfac * (4 - sqrt(2.0)) / 3
inner_radius = sqrt(2.0) + δ
outer_radius = 4 - sqrt(2.0) - 2δ
src_inner_circle1 = circle([-5.0, -2.5], inner_radius, Ncirc);
src_outer_circle1 = circle([-5.0, -2.5], outer_radius, Ncirc);
src_inner_circle2 = circle([-5.0,  2.5], inner_radius, Ncirc);
src_outer_circle2 = circle([-5.0,  2.5], outer_radius, Ncirc);
trg_inner_circle = circle([ 5.0, 0.0], inner_radius, Ncirc);
trg_outer_circle = circle([ 5.0, 0.0], outer_radius, Ncirc);
src1 = points_in_unit_square([-5.0, -2.5], N);
src2 = points_in_unit_square([-5.0,  2.5], N);
σ1 = randn(N);
σ2 = randn(N);

# plot this so we can see what this will look like
plt = scatter(first.(src1), last.(src1), color=:blue, xlims=(-8, 8), ylims=(-8, 8), aspect_ratio=:equal, label=:none)
scatter!(plt, first.(src2), last.(src2), color=:blue, label=:none)
scatter!(plt, first.(trg), last.(trg), color=:orange, label=:none)
scatter!(plt, first.(src_outer_circle1), last.(src_outer_circle1), color=:green, label=:none)
scatter!(plt, first.(src_inner_circle1), last.(src_inner_circle1), color=:red, label=:none)
scatter!(plt, first.(src_outer_circle2), last.(src_outer_circle2), color=:green, label=:none)
scatter!(plt, first.(src_inner_circle2), last.(src_inner_circle2), color=:red, label=:none)
scatter!(plt, first.(trg_outer_circle), last.(trg_outer_circle), color=:green, label=:none)
scatter!(plt, first.(trg_inner_circle), last.(trg_inner_circle), color=:red, label=:none)

# precompute some operators and their LU decompositions
SRC_INNER_TO_OUTER = ForwardDiff.jacobian(x -> poisson_kernel(src_inner_circle1, src_outer_circle1, x), zeros(Ncirc));
TRG_OUTER_TO_INNER = ForwardDiff.jacobian(x -> poisson_kernel(trg_outer_circle, trg_inner_circle, x), zeros(Ncirc));
LU_SRC_INNER_TO_OUTER = lu(SRC_INNER_TO_OUTER);
LU_TRG_OUTER_TO_INNER = lu(TRG_OUTER_TO_INNER);

# compute the interaction between the source points and the outer source circle
u_outer_src_circle1 = poisson_kernel(src1, src_outer_circle1, σ1); # N*Ncirc
u_outer_src_circle2 = poisson_kernel(src2, src_outer_circle2, σ2); # N*Ncirc
# get a density on the inner source circle that represents this
σ_inner_src_circle1 = LU_SRC_INNER_TO_OUTER \ u_outer_src_circle1; # Ncirc^2
σ_inner_src_circle2 = LU_SRC_INNER_TO_OUTER \ u_outer_src_circle2; # Ncirc^2
# evaluate this density on the inner target circle
u_inner_trg_circle = poisson_kernel(src_inner_circle1, trg_inner_circle, σ_inner_src_circle1); # Ncirc^2
u_inner_trg_circle .+= poisson_kernel(src_inner_circle2, trg_inner_circle, σ_inner_src_circle2); # Ncirc^2
# get a density on the outer target cirle that represents this
σ_outer_trg_circle = LU_TRG_OUTER_TO_INNER \ u_inner_trg_circle; # Ncirc^2
# now evaluate this onto the target points
u_trg_indirect = poisson_kernel(trg_outer_circle, trg, σ_outer_trg_circle); # Ncirc*N

# for comparison, get the direct interaction between the source and target points
u_trg_direct = poisson_kernel(src1, trg, σ1) + poisson_kernel(src2, trg, σ2); # N^2

# get the difference between direct and indirect evaluations
error = norm(u_trg_direct - u_trg_indirect, Inf) / norm(u_trg_direct, Inf);

@printf "Relative error: %0.3e\n" error

