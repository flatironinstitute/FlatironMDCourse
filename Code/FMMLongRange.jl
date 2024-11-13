using Revise;
using ForwardDiff;
using Plots;
using LaTeXStrings;
using LinearAlgebra;
using StaticArrays;
using Random;
using Distributions;
using SpecialFunctions;
using BenchmarkTools;
using Chairmarks;
Random.seed!(1234);

# We are going to do the example where we have two boxes separated by some great distance
# to evaluate source and target points on in 2D, such that we can then see how an FMM
# might work.

# Coulomb potential in 2D
@inline function Coulomb2D(x, y)
    return -log(norm(x-y))/(2π)
end

@inline function Coulomb2D(x::ComplexF64, y::ComplexF64)
    return -log(x-y)/(2π)
end

# Set up the number of particles, and where the two boxes are located
N = 1;
L = 100.0;
# Make the box separation some large number
Lsep = 5.0*L;

# Set the centers for omega_sigma and omega_tau
omega_sigma_center = SVector{2,Float64}(0.0, 0.0);
omega_tau_center = SVector{2,Float64}(Lsep, 0.0);

# Loop over the number of particels and set their positions.
Xsources = Vector{SVector{2,Float64}}(undef, N);
Xtargets = Vector{SVector{2,Float64}}(undef, N);
qsources = Vector{Float64}(undef, N);
qtargets = Vector{Float64}(undef, N);
for idx in 1:N
    Xsources[idx] = SVector{2,Float64}(rand(Uniform(-L/2, L/2)), rand(Uniform(-L/2, L/2))) + omega_sigma_center;
    Xtargets[idx] = SVector{2,Float64}(rand(Uniform(-L/2, L/2)), rand(Uniform(-L/2, L/2))) + omega_tau_center;
    qsources[idx] = idx%2 == 0 ? 1.0 : -1.0;
    qtargets[idx] = idx%2 == 0 ? -1.0 : 1.0;
end

# Create a complex version of the sources and targets
omega_sigma_center_complex = ComplexF64(omega_sigma_center[1], omega_sigma_center[2]);
omega_tau_center_complex = ComplexF64(omega_tau_center[1], omega_tau_center[2]);
Xsources_complex = [ComplexF64(Xsources[idx][1], Xsources[idx][2]) for idx in 1:N];
Xtargets_complex = [ComplexF64(Xtargets[idx][1], Xtargets[idx][2]) for idx in 1:N];

# We can directly construct the potentials uvec such that it is a matrix multiply
vtau = zeros(ComplexF64, N);
Amat = zeros(ComplexF64, N, N);
qvec = zeros(ComplexF64, N);
# qvec is the source strengths
for jdx in 1:N
    qvec[jdx] = qsources[jdx];
end
for idx in 1:N
    for jdx in 1:N
        Amat[idx,jdx] = Coulomb2D(Xtargets_complex[idx], Xsources_complex[jdx]);
    end
end

# Now vtau is the matrix vector product
vtau = Amat*qvec;

# Now we can actually do the FMM part. Define the outgoing expansion of Isigma.
P = 2;
qhatsigma = zeros(ComplexF64, P);
# Remember we have 1-based indexing in Julia
for ipdx in 1:P
    pdx = ipdx-1;
    if pdx == 0
        for jdx in 1:N
            qhatsigma[ipdx] += qsources[jdx];
        end
    else
        for jdx in 1:N
            qhatsigma[ipdx] += (-1.0/pdx) * (Xsources_complex[jdx] - omega_sigma_center_complex)^(pdx) * qsources[jdx];
        end
    end
end

# What we actually want is the map that defines this operation, such that
# q_hat_sigma = T_sigma_ofs q(I_sigma)

# Outgoing-from-sources translation operator
T_sigma_ofs = zeros(ComplexF64, P, N);
for ipdx in 1:P
    pdx = ipdx-1;
    for jdx in 1:N
        if pdx == 0
            T_sigma_ofs[ipdx,jdx] = 1.0;
        else
            T_sigma_ofs[ipdx,jdx] = (-1.0/pdx) * (Xsources_complex[jdx] - omega_sigma_center_complex)^(pdx);
        end
    end
end

# Incoming-from-outgoing translation operator
T_tau_sigma_ifo = zeros(ComplexF64, P, P);
for ipdx in 1:P
    i = ipdx-1;

    # Do the zero(i) case
    if i == 0
        for jpdx in 1:P
            j = jpdx-1;
            if j == 0
                T_tau_sigma_ifo[ipdx,jpdx] = log(omega_tau_center_complex - omega_sigma_center_complex);
            else
                T_tau_sigma_ifo[ipdx,jpdx] = (-1.0)^(j) / (omega_sigma_center_complex - omega_tau_center_complex)^(j);
            end
        end
    else
        # Do the (i) case
        for jpdx in 1:P
            j = jpdx-1;
            if j == 0
                T_tau_sigma_ifo[ipdx,jpdx] = -1.0 / (i * (omega_sigma_center_complex - omega_tau_center_complex)^(i));
            else
                T_tau_sigma_ifo[ipdx,jpdx] = (-1.0)^(j) * binomial(i+j-1, j-1) / (omega_sigma_center_complex - omega_tau_center_complex)^(i+j);
            end
        end
    end
end

# Targets-from-incoming translation operator
T_tau_tfi = zeros(ComplexF64, N, P);
for idx in 1:N
    for ipdx in 1:P
        pdx = ipdx-1;
        T_tau_tfi[idx,ipdx] = (Xtargets_complex[idx] - omega_tau_center_complex)^(pdx-1);
    end
end

# Calculate what should be the equivalent of the A matrix
T_tau_tfi * T_tau_sigma_ifo * T_sigma_ofs