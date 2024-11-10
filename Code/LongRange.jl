using Revise;
using ForwardDiff;
using Plots;
using LaTeXStrings;
using LinearAlgebra;
using StaticArrays;
using Random;
using Distributions;
using SpecialFunctions;
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

# Direct sum energy of all particles in a periodic system up to Nx, Ny, Nz replicas in each direction.
# Note that we don't have the minimum image here, and construct the lattice vector between the points.
function CoulombDirectSum!(mdsys::MDSystem, unitcell::UnitCell, Nx::Int64, Ny::Int64, Nz::Int64)
    N = mdsys.N;
    Epot = 0.0;
    for idx in 1:N
        for jdx in 1:N
            for mx in -Nx:Nx
                for my in -Ny:Ny
                    for mz in -Nz:Nz
                        # Note that we do NOT use the min image convention here!
                        r = mdsys.X[jdx] - mdsys.X[idx] + SVector{3,Float64}(mx*unitcell.L[1], my*unitcell.L[2], mz*unitcell.L[3]);
                        Epot += CoulombPotential(r, mdsys.charge[idx], mdsys.charge[jdx]);
                    end
                end
            end
        end
    end
    # Remember to include the factor of 1/2 to avoid double counting!
    return 0.5 * Epot;
end

# Short-range part of the Ewald sum
@inline function EwaldShortRangePotential(r::SVector{3,T}, q1::Float64, q2::Float64, alpha::Float64) where {T}
    return q1 * q2 * erfc(alpha * norm(r)) / norm(r);
end

# Do the short-range part of the Ewald energy calculation
function EwaldShortRange!(mdsys::MDSystem, unitcell::UnitCell, alpha::Float64)
    N = mdsys.N;
    Epot = 0.0;
    for idx in 1:N
        for jdx in 1:N
            if idx != jdx
                r = mdsys.X[jdx] - mdsys.X[idx];
                if norm(r) <= unitcell.L[1] / 2.0
                    Epot += EwaldShortRangePotential(r, mdsys.charge[idx], mdsys.charge[jdx], alpha);
                end
                # Epot += EwaldShortRangePotential(r, mdsys.charge[idx], mdsys.charge[jdx], alpha);
            end
        end
    end
    return 0.5 * Epot;
end

# Compute the k-vectors that lie within a sphere defined by a cutoff for our system
function EwaldKvectors!(unitcell::UnitCell, k_cutoff::Float64)
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
function EwaldLongRange!(mdsys::MDSystem, unitcell::UnitCell, alpha::Float64, kvectors::Vector{SVector{3,Float64}})
    Epot = 0.0;
    for k in kvectors
        # Compute the fourier transform of the charge density
        rho_k = zero(ComplexF64);
        for idx in 1:mdsys.N
            rho_k += mdsys.charge[idx] * exp(1im * dot(k, mdsys.X[idx]));
        end
        # Now do the energy contribution
        Epot += (4 * pi) / dot(k,k) * (rho_k' * rho_k) * exp(-dot(k,k) / (4 * alpha^2));
    end
    return Epot / (2.0 * unitcell.L[1] * unitcell.L[2] * unitcell.L[3]);
end

# Self-correction term in Ewald sum
function EwaldSelfEnergy!(mdsys::MDSystem, alpha::Float64)
    Epot = 0.0;
    for idx in 1:mdsys.N
        Epot += mdsys.charge[idx]^2 * alpha / sqrt(pi);
    end
    return Epot;
end

# Combined short-range and long-range portions of the Ewald summation with correction term
function EwaldSum!(mdsys::MDSystem, unitcell::UnitCell, alpha::Float64, Nk::Int64)
    # Calculate the short-range portion of the Ewald sum
    Eshort = EwaldShortRange!(mdsys, unitcell, alpha);
    # Calculate the long-range portion of the Ewald sum
    kvectors = EwaldKvectors!(unitcell, 2.0 * pi / unitcell.L[1] * Nk);
    Elong = EwaldLongRange!(mdsys, unitcell, alpha, kvectors);
    # Calculate the self-energy correction term
    Eself = EwaldSelfEnergy!(mdsys, alpha);
    return Eshort + Elong - Eself;
end

# Initialize our system
Nx = 2;
Ny = 1;
Nz = 1;
N = Nx * Ny * Nz;
# N = 10;
L = 10.0;
mass = 1.0;
kT = 1.0;
lattice_spacing = 2.0;
mdsys = MDSystem(N);
unitcell = UnitCell(L, true, true, true);
SetMasses!(mdsys, mass);
AlternateCharges!(mdsys);
# RandomInitialPositionsUnitCell!(mdsys, unitcell);
LatticeInitialPositionsUnitCell!(mdsys, lattice_spacing, Nx, Ny, Nz);
ThermalizeMomenta!(mdsys, kT);

# Various periodic configurations of the box
Nbox_max = 20+1;
E_direct = zeros(Nbox_max);
Nbox_direct = zeros(Nbox_max);

for ibox = 1:Nbox_max
    nbox = ibox - 1;
    Nbox_direct[ibox] = nbox;
    E_direct[ibox] = CoulombDirectSum!(mdsys, unitcell, nbox, nbox, nbox);
end
# plot(Nbox_direct, E_direct, label="Periodic Box Energy", xlabel="Number of Periodic Images", ylabel="Energy", legend=:topleft)

# Now do all of the Ewald sum information
# What do each of the individual portions look like?
alpha_vec = 0:0.1:2.0;
Nk = 40;
Eshort = zeros(length(alpha_vec));
Elong = zeros(length(alpha_vec));
Eself = zeros(length(alpha_vec));
EwaldComplete = zeros(length(alpha_vec));
for ialpha in eachindex(alpha_vec)
    alpha = alpha_vec[ialpha];
    # Short range portion
    Eshort[ialpha] = EwaldShortRange!(mdsys, unitcell, alpha);
    # Take the first Nk |kvector|'s
    kvectors = EwaldKvectors!(unitcell, 2.0 * pi / unitcell.L[1] * Nk);
    Elong[ialpha] = EwaldLongRange!(mdsys, unitcell, alpha, kvectors);
    # Self interaction
    Eself[ialpha] = EwaldSelfEnergy!(mdsys, alpha);
    # Entire Ewald sum
    EwaldComplete[ialpha] = EwaldSum!(mdsys, unitcell, alpha, Nk);
end

# Calculate the difference between using a lot of periodic images and the Ewald sum
E_diff = EwaldComplete .- E_direct[end]

# NOTE: Something is wrong that is giving a very slight systematic error for different values
# of alpha. I'm not sure what it is, but it is very small.
