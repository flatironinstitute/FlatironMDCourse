using LinearAlgebra
using StaticArrays
using Plots
using Printf

# generate n points in a unit square centered at c
function points_in_unit_square(c, n)
	cs = SVector{2, Float64}(c) .- 0.5
	return rand(SVector{2, Float64}, n) .+ Ref(cs)
end
# poisson kernel evaluation
@inline poisson_greens(r) = -log(r) / (2π)
@inline poisson_greens(src, trg) = poisson_greens(norm(trg - src))
function poisson_kernel(src, trg, σ)
	u = zeros(eltype(σ), length(trg))
	Threads.@threads for i in eachindex(trg, u)
		ui = zero(eltype(σ))
		for j in eachindex(src, σ)
			ui += poisson_greens(src[j], trg[i]) * σ[j]
		end
		u[i] = ui
	end
	return u
end
# poisson matrix construction, i.e. if M = poisson_matrix(src, trg),
# then M*σ = poisson_kernel(src, trg, σ)
function poisson_matrix(src, trg)
	M = zeros(Float64, length(trg), length(src))
	Threads.@threads for i in eachindex(trg)	
		for j in eachindex(src)
			@inbounds M[i, j] = poisson_greens(src[j], trg[i])
		end
	end
	return M
end

################################################################################
# check out low-rank interaction structure

# We start by looking at the rank of the interaction between
# two well-separated sets of points
N = 2_000
src = points_in_unit_square([-5.0, 0.0], N);
trg = points_in_unit_square([ 5.0, 0.0], N);
σ = randn(N);
# construct src-to-trg operator
M = poisson_matrix(src, trg);
S = svd(M);
sv = S.S;
plt = plot(sv/sv[1], yscale=:log10, color=:black, linewidth=2, xlabel="Singular value index", ylabel="Singular value / Max Singular Value", label=:none)
hline!([eps(Float64)],   label="ε", color=:gray, linewidth=2, linestyle=:dash)
display(plt)

# zoom in on the first singular values
xlims!(plt, 1, 40)
display(plt)

GC.enable(false)
# time direct evaluation (two ways)
u_direct1, t_direct1 = @timed poisson_kernel(src, trg, σ);
u_direct2, t_direct2 = @timed M*σ;
# time svd based evaluation
ε = 1e-10;
NS = sum(S.S .> ε);
U = S.U[:, 1:NS];
Vt = S.Vt[1:NS, :];
Σ = Diagonal(S.S[1:NS]);
u_indirect, t_indirect = @timed U*Σ*Vt*σ;
GC.enable(true)

error = norm(u_direct1 - u_indirect, Inf) / norm(u_direct1, Inf);
@printf "Relative error:                %0.3e\n" error
@printf "Direct eval (function) took:   %0.3f ms\n" t_direct1*1000
@printf "Direct eval (matrix)   took:   %0.3f ms\n" t_direct2*1000
@printf "Indirect eval took:            %0.3f ms\n" t_indirect*1000

################################################################################
# "FMM" demonstration

N = 50_000

# okay, now we'll try to exploit this low-rank structure in some analytical way
function circle(c, r, n)
	θ = LinRange(0, 2π, n+1)[1:end-1]
	return [SVector{2, Float64}(c[1] + r*cos(θ[i]), c[2] + r*sin(θ[i])) for i in 1:n]
end
src1 = points_in_unit_square([-5.0, -2.5], N);
src2 = points_in_unit_square([-5.0,  2.5], N);
trg = points_in_unit_square([ 5.0, 0.0], N);
σ1 = randn(N);
σ2 = randn(N);

################################################################################
# indirect method, setup
# O(Ncirc^3)

Ncirc = 16
# definition of inner/outer radius according to Ying/Biros/Zorin
δfac = 0.0 # (needs to be in [0, 1)) 
δ = δfac * (4 - sqrt(2.0)) / 3
inner_radius = sqrt(2.0) + δ
outer_radius = 4 - sqrt(2.0) - 2δ
# modification for demonstration purposes
outer_radius *= 1.0
src_inner_circle1 = circle([-5.0, -2.5], inner_radius, Ncirc);
src_outer_circle1 = circle([-5.0, -2.5], outer_radius, Ncirc);
src_inner_circle2 = circle([-5.0,  2.5], inner_radius, Ncirc);
src_outer_circle2 = circle([-5.0,  2.5], outer_radius, Ncirc);
trg_inner_circle =  circle([ 5.0,  0.0], inner_radius, Ncirc);
trg_outer_circle =  circle([ 5.0,  0.0], outer_radius, Ncirc);

# plot this so we can see what this will look like
max_plt = 1_000
function sfl(x)
	w = x[1:min(length(x), max_plt)]
	return (first.(w), last.(w))
end
wrap(x) = push!(copy(x), x[1])
plt = plot(xlims=(-8, 8), ylims=(-8, 8), aspect_ratio=:equal, label=:none)
series = [src1,  src2,  trg,    ];
colors = [:blue, :blue, :orange ];
add_scatter(x, col) = scatter!(plt, sfl(x)..., label=:none, color=col)
add_scatter.(series, colors);
series = [src_outer_circle1, src_inner_circle1, src_outer_circle2, src_inner_circle2, trg_outer_circle, trg_inner_circle ];
colors = [:green,            :red,              :green,            :red,              :green,           :red             ];
add_plot(x, col) = plot!(plt, sfl(wrap(x))..., label=:none, color=col, marker=:x)
add_plot.(series, colors);
display(plt)

# precompute some operators and their LU decompositions
SRC_INNER_TO_OUTER = poisson_matrix(src_inner_circle1, src_outer_circle1); # Ncirc^2
TRG_OUTER_TO_INNER = poisson_matrix(trg_outer_circle, trg_inner_circle); # Ncirc^2
LU_SRC_INNER_TO_OUTER = lu(SRC_INNER_TO_OUTER); # Ncirc^3
LU_TRG_OUTER_TO_INNER = lu(TRG_OUTER_TO_INNER); # Ncirc^3

################################################################################
# indirect method, after setup
# O(N*Ncirc + Ncirc^2)

GC.enable(false)
u_trg_indirect, t_indirect = @timed begin

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

	u_trg_indirect
end;

################################################################################
# direct method
# O(N^2)

# for comparison, get the direct interaction between the source and target points
u_trg_direct, t_direct = @timed poisson_kernel(src1, trg, σ1) + poisson_kernel(src2, trg, σ2); # N^2
GC.enable(true)

################################################################################
# comparison

# get the difference between direct and indirect evaluations
error = norm(u_trg_direct - u_trg_indirect, Inf) / norm(u_trg_direct, Inf);
@printf "Relative error:     %0.3e\n" error
@printf "Direct eval took:   %0.3f s\n" t_direct
@printf "Indirect eval took: %0.3f s\n" t_indirect
