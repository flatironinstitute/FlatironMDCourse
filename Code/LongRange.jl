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

# We are going to build out own entire system for MD in the notebook so that we can do whatever we want with it.
# Make it fairly general.

# MD System keeps track of the current system state, and things that don't change per particle.
# Thanks to David Stein for what we do in JellySim so that I can autodiff the potential to get the forces
# if I want to.
struct MDSystem
    N::Int64;
    X::Vector{SVector{3,Float64}};
    P::Vector{SVector{3,Float64}};
    F::Vector{SVector{3,Float64}};
    mass::Vector{Float64};
    charge::Vector{Float64};
    Ekin::Float64;
    Epot::Float64;
    Etot::Float64;
    kT::Float64;
end
# Create a function to create a default version of this structure
function MDSystem(N::Int64)
    X = Vector{SVector{3,Float64}}(undef, N);
    P = Vector{SVector{3,Float64}}(undef, N);
    F = Vector{SVector{3,Float64}}(undef, N);
    # Initialize X and P to zeros (thermalize and set initial positions later)
    for idx in 1:N
        X[idx] = SVector{3,Float64}(0.0, 0.0, 0.0);
        P[idx] = SVector{3,Float64}(0.0, 0.0, 0.0);
        F[idx] = SVector{3,Float64}(0.0, 0.0, 0.0);
    end
    mass = zeros(N);
    charge = zeros(N);
    Ekin = 0.0;
    Epot = 0.0;
    Etot = 0.0;
    kT = 0.0;
    return MDSystem(N, X, P, F, mass, charge, Ekin, Epot, Etot, kT);
end

# UnitCell keeps track of the dimensionality of the system.
struct UnitCell
    L::SVector{3,Float64};
    periodic::SVector{3,Int64};
end
function UnitCell(L::Float64, periodicX::Bool, periodicY::Bool, periodicZ::Bool)
    Lvec = SVector{3,Float64}(L, L, L);
    periodicvec = SVector{3,Int64}(periodicX, periodicY, periodicZ);
    return UnitCell(Lvec, periodicvec);
end

# This will randomize our initial positions inside of a defined unit cell
function RandomInitialPositionsUnitCell!(mdsys::MDSystem,
    unitcell::UnitCell)
    for idx in 1:mdsys.N
        randpos = (rand(3) .- 0.5) .* unitcell.L;
        mdsys.X[idx] = SVector{3,Float64}(randpos[1], randpos[2], randpos[3]);
    end
    return nothing
end
# Create a lattice of initial positions in XYZ, centered at (0, 0, 0)
function LatticeInitialPositionsUnitCell!(mdsys::MDSystem,
    lattice_spacing::Float64, Nx::Int64, Ny::Int64, Nz::Int64)
    for idx in 1:Nx
        for jdx in 1:Ny
            for kdx in 1:Nz
                x = (idx - 1) * lattice_spacing - 0.5 * (Nx - 1) * lattice_spacing;
                y = (jdx - 1) * lattice_spacing - 0.5 * (Ny - 1) * lattice_spacing;
                z = (kdx - 1) * lattice_spacing - 0.5 * (Nz - 1) * lattice_spacing;
                mdsys.X[(idx-1)*Ny*Nz + (jdx-1)*Nz + kdx] = SVector{3,Float64}(x, y, z);
            end
        end
    end
end
# This will thermalize the 3D system (assign velocities in a Maxwellian distribution way)
function ThermalizeMomenta!(mdsys::MDSystem,
        kT::Float64)
    N = mdsys.N;
    for idx in 1:N
        MaxwellDistribution = Normal(0.0, sqrt(kT/mdsys.mass[idx]));
        mdsys.P[idx] = rand(MaxwellDistribution, 3);
    end
    return nothing
end
# This will alternate setting charges for the system
function AlternateCharges!(mdsys::MDSystem)
    for idx in 1:mdsys.N
        if idx % 2 == 0
            mdsys.charge[idx] = 1.0;
        else
            mdsys.charge[idx] = -1.0;
        end
    end
    return nothing
end
# This will set the masses of particles in the system
function SetMasses!(mdsys::MDSystem, m::Float64)
    for idx in 1:mdsys.N
        mdsys.mass[idx] = m;
    end
    return nothing
end

# Calculate the minimum distance in 3D assuming that the box goes from -L/2 to L/2
# This follows the minimum image convention, where if the distance for interaction is greater than half the box length
# it will wrap. Unfortunately, static arrays are not mutable by default, so we are going to do some internal
# manipulations to make sure they work correctly (yuck).
@inline function min_image(r::SVector{3,Float64}, unitcell::UnitCell)
    rprime = r;
    if rprime[1] >= 0.5*unitcell.L[1]
        i = floor(Int64, rprime[1] / unitcell.L[1] + 0.5);
        rprime = setindex(rprime, rprime[1] - Float64(i) * unitcell.L[1], 1);
    elseif rprime[1] < -0.5*unitcell.L[1]
        i = floor(Int64, -rprime[1] / unitcell.L[1] + 0.5);
        rprime = setindex(rprime, rprime[1] + Float64(i) * unitcell.L[1], 1);
    end
    if rprime[2] >= 0.5*unitcell.L[2]
        i = floor(Int64, rprime[2] / unitcell.L[2] + 0.5);
        rprime = setindex(rprime, rprime[2] - Float64(i) * unitcell.L[2], 2);
    elseif rprime[2] < -0.5*unitcell.L[2]
        i = floor(Int64, -rprime[2] / unitcell.L[2] + 0.5);
        rprime = setindex(rprime, rprime[2] + Float64(i) * unitcell.L[2], 2);
    end
    if rprime[3] >= 0.5*unitcell.L[3]
        i = floor(Int64, rprime[3] / unitcell.L[3] + 0.5);
        rprime = setindex(rprime, rprime[3] - Float64(i) * unitcell.L[3], 3);
    elseif rprime[3] < -0.5*unitcell.L[3]
        i = floor(Int64, -rprime[3] / unitcell.L[3] + 0.5);
        rprime = setindex(rprime, rprime[3] + Float64(i) * unitcell.L[3], 3);
    end
    return rprime;
end

# Coulomb potential between two particles, getting rid of the self-interaction term
@inline function CoulombPotential(r::SVector{3,T}, q1::Float64, q2::Float64) where {T}
    return norm(r) == 0 ? 0.0 : q1 * q2 / norm(r);
end

# Build the real space number of periodic boxes to use
function BuildPeriodicBoxes(Nradial::Int64)
    # Take a sphere of boxes to compare to the Ewald sum
    nvectors = Vector{SVector{3,Float64}}();
    for mx in -Nradial:Nradial
        for my in -Nradial:Nradial
            for mz in -Nradial:Nradial
                if norm(SVector{3,Float64}(mx, my, mz)) <= Nradial
                    push!(nvectors, SVector{3,Float64}(mx, my, mz));
                end
            end
        end
    end
    return nvectors;
end

# Direct sum energy of all particles in a periodic system up to Nx, Ny, Nz replicas in each direction.
# Note that we don't have the minimum image here, and construct the lattice vector between the points.
function CoulombDirectSumSerial(mdsys::MDSystem, unitcell::UnitCell, Nradial::Int64)
    N = mdsys.N;
    Epot = Threads.Atomic{Float64}(0.0);
    # Take a sphere of boxes to compare to the Ewald sum
    nvectors = BuildPeriodicBoxes(Nradial);
     # Thread this loop
    Threads.@threads for n in nvectors
        for idx in 1:N
            for jdx in 1:N
                r = mdsys.X[jdx] - mdsys.X[idx] + SVector{3,Float64}(n[1]*unitcell.L[1], n[2]*unitcell.L[2], n[3]*unitcell.L[3]);
                Epot_local = CoulombPotential(r, mdsys.charge[idx], mdsys.charge[jdx]);
                Threads.atomic_add!(Epot, Epot_local);
            end
        end
    end
    # Remember to include the factor of 1/2 to avoid double counting! Also get it out of the Atomic version.
    return 0.5 * Epot[];
end

# Use atomics to calculate the energy
function CoulombDirectSumAtomic(mdsys::MDSystem, unitcell::UnitCell, Nradial::Int64)
    N = mdsys.N;
    Epot = Threads.Atomic{Float64}(0.0);
    # Take a sphere of boxes to compare to the Ewald sum
    nvectors = BuildPeriodicBoxes(Nradial);
     # Thread this loop
    Threads.@threads for n in nvectors
        for idx in 1:N
            for jdx in 1:N
                r = mdsys.X[jdx] - mdsys.X[idx] + SVector{3,Float64}(n[1]*unitcell.L[1], n[2]*unitcell.L[2], n[3]*unitcell.L[3]);
                Epot_local = CoulombPotential(r, mdsys.charge[idx], mdsys.charge[jdx]);
                Threads.atomic_add!(Epot, Epot_local);
            end
        end
    end
    # Remember to include the factor of 1/2 to avoid double counting! Also get it out of the Atomic version.
    return 0.5 * Epot[];
end

# Direct sum energy of all particles in a periodic system using thread local storage and a reduction at the end.
function CoulombDirectSumReduce(mdsys::MDSystem, unitcell::UnitCell, Nradial::Int64)
    N = mdsys.N;
    Epot_local = zeros(Float64, Threads.nthreads());
    # Take a sphere of boxes to compare to the Ewald sum
    nvectors = BuildPeriodicBoxes(Nradial);
     # Thread this loop
    Threads.@threads for n in nvectors
        for idx in 1:N
            for jdx in 1:N
                r = mdsys.X[jdx] - mdsys.X[idx] + SVector{3,Float64}(n[1]*unitcell.L[1], n[2]*unitcell.L[2], n[3]*unitcell.L[3]);
                Epot_local[Threads.threadid()] += CoulombPotential(r, mdsys.charge[idx], mdsys.charge[jdx]);
            end
        end
    end
    # Reduce the local energy to the global energy
    Epot = sum(Epot_local);
    # Remember to include the factor of 1/2 to avoid double counting!
    return 0.5 * Epot;
end

# We will be using Allen for the calculation of the Ewald sum. Frenkel has alpha = = sqrt(alpha_allen)
# Short-range part of the Ewald sum
@inline function EwaldShortRangePotential(r::SVector{3,T}, q1::Float64, q2::Float64, sqrtalpha::Float64) where {T}
    return q1 * q2 * erfc(sqrtalpha * norm(r)) / norm(r);
end

# Do the short-range part of the Ewald energy calculation
function EwaldShortRangeSerial(mdsys::MDSystem, unitcell::UnitCell, sqrtalpha::Float64, rcut::Float64)
    N = mdsys.N;
    Epot = 0.0;
    for idx in 1:N
        for jdx in 1:N
            if idx != jdx
                r = min_image(mdsys.X[jdx] - mdsys.X[idx], unitcell);
                if norm(r) <= rcut
                    Epot += EwaldShortRangePotential(r, mdsys.charge[idx], mdsys.charge[jdx], sqrtalpha);
                end
            end
        end
    end
    return 0.5 * Epot;
end

# Do the short-range part of the Ewald energy calculation using reductions
function EwaldShortRangeReduce(mdsys::MDSystem, unitcell::UnitCell, sqrtalpha::Float64, rcut::Float64)
    N = mdsys.N;
    Epot_local = zeros(Float64, Threads.nthreads());
    Threads.@threads for idx in 1:N
        for jdx in 1:N
            if idx != jdx
                r = min_image(mdsys.X[jdx] - mdsys.X[idx], unitcell);
                if norm(r) <= rcut
                    Epot_local[Threads.threadid()] += EwaldShortRangePotential(r, mdsys.charge[idx], mdsys.charge[jdx], sqrtalpha);
                end
            end
        end
    end
    # Reduce the local energy to the global energy
    Epot = sum(Epot_local);
    return 0.5 * Epot;
end

# Compute the k-vectors that lie within a sphere defined by a cutoff for our system
function EwaldKvectors(unitcell::UnitCell, k_cutoff::Float64)
    kx_max = ceil(Int64, k_cutoff * unitcell.L[1] / (2 * pi)) + 1;
    ky_max = ceil(Int64, k_cutoff * unitcell.L[2] / (2 * pi)) + 1;
    kz_max = ceil(Int64, k_cutoff * unitcell.L[3] / (2 * pi)) + 1;
    kvectors = Vector{SVector{3,Float64}}();

    # Loop over the combinations of kx, ky, kz and find those that are within the cutoff and
    # add them to the kvectors
    for kx in -kx_max:kx_max
        for ky in -ky_max:ky_max
            for kz in -kz_max:kz_max
                k = 2 * pi * SVector{3,Float64}(kx / unitcell.L[1], ky / unitcell.L[2], kz / unitcell.L[3]);
                if 0 < norm(k) <= k_cutoff
                    push!(kvectors, k);
                end
            end
        end
    end
    return kvectors;
end

# Long-range Ewald energy
function EwaldLongRangeSerial(mdsys::MDSystem, unitcell::UnitCell, alpha::Float64, kvectors::Vector{SVector{3,Float64}})
    Epot = 0.0;
    for k in kvectors
        # Compute the fourier transform of the charge density
        rho_k = zero(ComplexF64);
        for idx in 1:mdsys.N
            rho_k += mdsys.charge[idx] * exp(1im * dot(k, mdsys.X[idx]));
        end
        # Now do the energy contribution
        Epot += (4 * pi) / dot(k,k) * (rho_k' * rho_k) * exp(-dot(k,k) / (4 * alpha));
    end
    return Epot / (2.0 * unitcell.L[1] * unitcell.L[2] * unitcell.L[3]);
end

# Long-range Ewald energy with reduction
function EwaldLongRangeReduce(mdsys::MDSystem, unitcell::UnitCell, alpha::Float64, kvectors::Vector{SVector{3,Float64}})
    Epot_local = zeros(Float64, Threads.nthreads());
    Threads.@threads for k in kvectors
        # Compute the fourier transform of the charge density
        rho_k = zero(ComplexF64);
        for idx in 1:mdsys.N
            rho_k += mdsys.charge[idx] * exp(1im * dot(k, mdsys.X[idx]));
        end
        # Now do the energy contribution
        Epot_local[Threads.threadid()] += (4 * pi) / dot(k,k) * (rho_k' * rho_k) * exp(-dot(k,k) / (4 * alpha));
    end
    # Sum the local energies
    Epot = sum(Epot_local);
    return Epot / (2.0 * unitcell.L[1] * unitcell.L[2] * unitcell.L[3]);
end

# Self-correction term in Ewald sum
function EwaldSelfEnergy(mdsys::MDSystem, alpha::Float64)
    Epot = 0.0;
    for idx in 1:mdsys.N
        Epot += mdsys.charge[idx]^2 * sqrt(alpha / pi);
    end
    return Epot;
end

# Ewald boundary correction term
function EwaldBoundaryEnergy(mdsys::MDSystem, unitcell::UnitCell, epsilonprime::Float64)
    rqi = SVector{3,Float64}(0.0, 0.0, 0.0);
    for idx in 1:mdsys.N
        rqi += mdsys.charge[idx] * mdsys.X[idx];
    end
    V = unitcell.L[1] * unitcell.L[2] * unitcell.L[3];
    return 2 * pi / ((2 * epsilonprime + 1)*V) * dot(rqi, rqi);
end

# Combined short-range and long-range portions of the Ewald summation with correction term
function EwaldSumSerial(mdsys::MDSystem, unitcell::UnitCell, alpha::Float64, rcut::Float64, Nk::Int64)
    # Convert to sqrt(alpha) for the short-range portion
    sqrtalpha = sqrt(alpha);
    # Calculate the short-range portion of the Ewald sum
    Eshort = EwaldShortRangeSerial(mdsys, unitcell, sqrtalpha, rcut);
    # Calculate the long-range portion of the Ewald sum
    kvectors = EwaldKvectors(unitcell, 2.0 * pi / unitcell.L[1] * Nk);
    Elong = EwaldLongRangeSerial(mdsys, unitcell, alpha, kvectors);
    # Calculate the self-energy correction term
    Eself = EwaldSelfEnergy(mdsys, alpha);
    # Calculate the boundary term
    Eboundary = EwaldBoundaryEnergy(mdsys, unitcell, 1.0);
    return Eshort + Elong + Eboundary - Eself;
end

# Threaded reduction version of the Ewald sum
function EwaldSumReduce(mdsys::MDSystem, unitcell::UnitCell, alpha::Float64, rcut::Float64, Nk::Int64)
    # Convert to sqrt(alpha) for the short-range portion
    sqrtalpha = sqrt(alpha);
    # Calculate the short-range portion of the Ewald sum
    Eshort = EwaldShortRangeReduce(mdsys, unitcell, sqrtalpha, rcut);
    # Calculate the long-range portion of the Ewald sum
    kvectors = EwaldKvectors(unitcell, 2.0 * pi / unitcell.L[1] * Nk);
    Elong = EwaldLongRangeReduce(mdsys, unitcell, alpha, kvectors);
    # Calculate the self-energy correction term
    Eself = EwaldSelfEnergy(mdsys, alpha);
    # Calculate the boundary term
    Eboundary = EwaldBoundaryEnergy(mdsys, unitcell, 1.0);
    return Eshort + Elong + Eboundary - Eself;
end

# Compute the error in the Ewald sum
function EwaldError(mdsys::MDSystem, unitcell::UnitCell, s::Float64, alpha::Float64)
    Q = sum(mdsys.charge.^2);
    dEshort = Q * sqrt(s / (alpha * unitcell.L[1] * unitcell.L[2] * unitcell.L[3])) * exp(-s^2) / (s^2);
    dElong = Q * sqrt(s / (2.0 * alpha * unitcell.L[1] * unitcell.L[2] * unitcell.L[3])) * exp(-s^2) / (s^2);
    return dEshort, dElong;
end

# Initialize our system
# Nx = 2;
# Ny = 1;
# Nz = 1;
# N = Nx * Ny * Nz;
N = 100;
L = 100.0;
mass = 1.0;
kT = 1.0;
lattice_spacing = 2.0;
mdsys = MDSystem(N);
unitcell = UnitCell(L, true, true, true);
SetMasses!(mdsys, mass);
AlternateCharges!(mdsys);
RandomInitialPositionsUnitCell!(mdsys, unitcell);
# LatticeInitialPositionsUnitCell!(mdsys, lattice_spacing, Nx, Ny, Nz);
ThermalizeMomenta!(mdsys, kT);

println("Using NThreads = $(Threads.nthreads())");

# Various periodic configurations of the box
Nradial_max = 60;
E_direct = zeros(Nradial_max);
E_directatomic = zeros(Nradial_max);
E_directreduce = zeros(Nradial_max);
Nbox_direct = zeros(Nradial_max);

println("Direct Coulomb calculation");
for iradial = 10:10:Nradial_max
    Nbox_direct[iradial] = iradial;
    println("  iradial = $iradial");
    # E_direct[iradial] = CoulombDirectSumSerial(mdsys, unitcell, nradial);
    # E_directatomic[iradial] = CoulombDirectSumAtomic(mdsys, unitcell, nradial);
    E_directreduce[iradial] = CoulombDirectSumReduce(mdsys, unitcell, iradial);
end
# Test the timing of the different direct sum threading levels
# println("Testing Coulomb direct sum speeds");
# for iradial = 1:Nradial_max
#     nradial = iradial - 1;
#     println("DirectSum nradial = $nradial");
#     @btime CoulombDirectSum($mdsys, $unitcell, $nradial)
# end
# Quick and dirty benchmarking
# @b CoulombDirectSum(mdsys, unitcell, 10)
# @b CoulombDirectSumAtomic(mdsys, unitcell, 10)
# @b CoulombDirectSumReduce(mdsys, unitcell, 10)

# plot(Nbox_direct, E_direct, label="Periodic Box Energy", xlabel="Number of Periodic Images", ylabel="Energy", legend=:topleft)

# Now do all of the Ewald sum information
# What do each of the individual portions look like?
alpha_vec = 0:0.1:2.0;
# alpha_vec = 0:0.1:1.0;
s = 3.0;
E_Ewaldserial = zeros(length(alpha_vec));
E_Ewaldreduce = zeros(length(alpha_vec));
dE_short = zeros(length(alpha_vec));
dE_long = zeros(length(alpha_vec));
dE_total = zeros(length(alpha_vec));
for ialpha in eachindex(alpha_vec)
    alpha = alpha_vec[ialpha];
    # For a given choice of alpha and the scale parameter, we can compute the cutoff radius and how many
    # terms in k-space we need to include.
    rcut = s / alpha;
    Nk = 2*ceil(Int64, s * unitcell.L[1] * alpha / pi);
    println("alpha = $alpha, rcut = $rcut, Nk = $Nk");

    # Entire Ewald sum
    # E_Ewaldserial[ialpha] = EwaldSumSerial(mdsys, unitcell, alpha, rcut, Nk);
    E_Ewaldreduce[ialpha] = EwaldSumReduce(mdsys, unitcell, alpha, rcut, Nk);

    # What the error should be in the terms of the Ewald sum
    dE_short[ialpha], dE_long[ialpha] = EwaldError(mdsys, unitcell, s, alpha);
    dE_total[ialpha] = dE_short[ialpha] + dE_long[ialpha];
end

# Calculate the difference between using a lot of periodic images and the Ewald sum
# E_diff = E_Ewald .- E_directreduce[end]

# NOTE: Something is wrong that is giving a very slight systematic error for different values
# of alpha. I'm not sure what it is, but it is very small.
