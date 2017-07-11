using QuBase
using QuDynamics

import Base.isinteger,
QuBase.spin,
Base.length,
Base.size,
Base.getindex,
Base.promote_rule,
Base.convert

"""
A library of spin values (S = j/2) and gyromagnetic ratios (rad/s).

Expand as needed.
"""
function spin(name::String)
    # for now, we only define a few
    # electron "E"
    if name == "E"
        j = 1
        gamma = -1.760859708e11
    # proton "1H"
    elseif name == "1H"
        j = 1
        gamma = 2.675222005e8
    # deueterium "2H"
    elseif name == "2H"
        j = 2
        gamma = 4.10662791e7
    # carbon-13 "13C"
    elseif name == "13C"
        j = 1
        gamma = 6.728284e7
    # nitrogen-15 "15N"
    elseif name == "15N"
        j = 1
        gamma = -2.71261804e7
    # oxygen-17 "17O"
    elseif name == "17O"
        j = 5
        gamma = -3.62808e7
    # fluorine-19 "19F"
    elseif name == "19F"
        j = 1
        gamma = 2.51848e8
    else
        error("spin not recognized.")
    end
    (j,gamma)
end

immutable Isotope{T<:Real}
    label::String
    spin::QuBase.HalfSpin
    gamma::T
    Isotope(label::String, spin::QuBase.HalfSpin, gamma::T) = new(label, spin, gamma)
end

Isotope{T<:Real}(label::String, spin::QuBase.HalfSpin, gamma::T) = Isotope{T}(label, spin, gamma)
Isotope{T<:Real,V<:Integer}(label::String, j::V, gamma::T) = Isotope{T}(label, QuBase.spin(j//2), gamma)
Isotope(label::String) = Isotope(label, spin(label)...)

function spin(x::Isotope)
    QuBase.spin(x.spin)
end

function mult(x::Isotope)
    Int(2*QuBase.spin_value(spin(x))+1)
end

# a type for a particular Zeeman coupling
# we need it to be mutable so we can change the strength flag
# it contains the label of the spin as well
mutable struct Zeeman{T<:Real}
    spin::Int
    matrix::Array{T,2}          # csa tensor
    strength::String            # flag for keeping labframe, secular, etc.
    Zeeman{T}(spin::Int, matrix::Array{T,2}, strength::String) where {T<:Real} = new(spin, matrix, strength)
    Zeeman{T}(spin::Int, matrix::Array{T,2}) where {T<:Real} = new(spin, matrix, "")
end

function size(x::Zeeman{T} where {T<:Real})
    size(x.matrix)
end

function getindex(x::Zeeman, values...)
    getindex(x.matrix, values...)
end

# ZYZ convention (copied from Spinach)
function euler2dcm{T<:Real}(a::T, b::T, g::T)
    Ra = [cos(a) -sin(a) 0;
          sin(a)  cos(a) 0;
          0       0      1]
    Rb = [cos(b)  0      sin(b);
          0       1      0;
          -sin(b) 0      cos(b)]
    Rg = [cos(g) -sin(g) 0;
          sin(g)  cos(g) 0;
          0       0      1]
    Ra*Rb*Rg
end

function Zeeman{T1<:Real, T2<:Real}(spin::Int, eigs::Vector{T1}, euler::Vector{T2}, strength::String="")
    dcm = euler2dcm(euler...)
    matrix = dcm*diagm(eigs)*transpose(dcm)
    Zeeman{eltype(matrix)}(spin, matrix, strength)
end

function Zeeman{T<:Real}(spin::Int, isotropic_cs::T, strength::String="")
    Zeeman{T}(spin, isotropic_cs*eye(T,3), strength)
end

# Haeberlen convention, for converting from spinev
function Zeeman(spin::Int, iso, aniso, eta, euler::Vector, strength::String="")
    eigs = zeros(3)
    eigs[1] = iso + aniso
    eigs[2] = iso - aniso*(1-eta)/2
    eigs[3] = iso - aniso*(1+eta)/2
    if aniso < 0
        eigs = flipdim(eigs,1)
    end
    Zeeman(spin, eigs, euler, strength)
end

mutable struct Coord{T<:Real}
    spin::Int
    pos::Vector{T}
    function Coord(spin::Int, pos::Vector{T}) where {T<:Real}
        new{T}(spin, pos)
    end
end

function size(x::Coord)
    size(x.pos)
end

function getindex(x::Coord, values...)
    getindex(x.pos, values...)
end

# a coupling should have the spins it involves as part of it.
# the other way is to store matrices for all possible couplings
mutable struct Coupling{T<:Real}
    spins::Tuple{Int, Int}
    matrix::Array{T,2}
    strength::String
    Coupling{T}(spins::Tuple{Int,Int}, matrix::Array{T,2}, strength::String) where {T<:Real} = new(spins, matrix, strength)
    Coupling{T}(spins::Tuple{Int,Int}, matrix::Array{T,2}) where {T<:Real} = new(spins, matrix, "")
end

import Base.show
function Base.show(x::Coupling)
    l = x.spins[1]
    s = x.spins[2]
    mat = x.matrix
    length_skip = length("(I$(l)x  I$(l)y  I$(l)z)")
    @printf "%s  ⎛%8.4f  %8.4f  %8.4f⎞  ⎛I%dx⎞\n" " "^length_skip mat[1,1] mat[1,2] mat[1,3] s
    @printf "(I%dx  I%dy  I%dz)  ⎜%8.4f  %8.4f  %8.4f⎟  ⎜I%dy⎟\n" l l l mat[2,1] mat[2,2] mat[2,3] s
    @printf "%s  ⎝%8.4f  %8.4f  %8.4f⎠  ⎝I%dz⎠\n" " "^length_skip mat[3,1] mat[3,2] mat[3,3] s
end

function Coupling{T1<:Real, T2<:Real}(spins::Tuple{Int,Int}, eigs::Vector{T1}, euler::Vector{T2}, strength::String="")
    dcm = euler2dcm(euler...)
    matrix = dcm*diagm(eigs)*transpose(dcm)
    Coupling{eltype(matrix)}(spins, matrix, strength)
end

function Coupling{T<:Real}(spins::Tuple{Int,Int}, scalar_coupling::T, strength::String="")
    Coupling{T}(spins, scalar_coupling*eye(T,3), strength)
end

mutable struct System{T<:Real}
    magnet::T
    isotopes::Vector{Isotope}
    zeeman::Vector{Zeeman}
    coords::Vector{Coord}
    couplings::Vector{Coupling}
    formalism::String           # "Hilbert" or "Liouville"
    System{T}() where T = new(0,[],[],[],[],"Liouville")
end

# modifies z and returns ddscal
function ddscal!(iso::Isotope, z::Zeeman, magnet::T where {T<:Real})
    freeg = 2.0023193043622
    if iso.label=="E"
        answer = z.matrix / freeg
        z.matrix = z.matrix * (-magnet * iso.gamma) / freeg
    else
        answer = eye(3) + (z.matrix * 1e-6)
        z.matrix = (eye(3) + (z.matrix * 1e-6)) * (-magnet * iso.gamma)
    end
    answer
end

# compute dipolar coupling between two spins
# meant for use as part of another function which will iterate through coords
# include the appropriate ddscal values
function dipolar(iso1::Isotope, iso2::Isotope, c1::Coord, c2::Coord, dscal1::Array{Float64,2}, dscal2::Array{Float64, 2})
    distvect = c2.pos - c1.pos
    distance = norm(distvect)
    ort = distvect/distance
    hbar  = 1.054571628e-34
    mu0   = pi*4e-7
    d = iso1.gamma * iso2.gamma * hbar * mu0 / (4*pi*(dist*1e-10)^3)
    mat = d * [1-3*ort[1]*ort[1]    -3*ort[1]*ort[2]   -3*ort[1]*ort[3];
               -3*ort[2]*ort[1]   1-3*ort[2]*ort[2]   -3*ort[2]*ort[3];
               -3*ort[3]*ort[1]    -3*ort[3]*ort[2]  1-3*ort[3]*ort[3]]
    mat = dscal1' * mat * dscal2
    mat = mat - (eye(3)*trace(mat)/3)
    mat = (mat + mat')/2
    Coupling((c1.spin, c2.spin), mat, "")
end

# construct a label basis for the system.
# in Hilbert space, we'll use a simple basis of z states
# in Liouville space, we'll use irreducible spherical tensors
# if dimensionality is n in Hilbert, it's n^2 in Liouville
function basis(sys::System)
    mults = map(mult, sys.isotopes)
    QuBase.LabelBasis(mults...)
end

function liouv_basis(sys::System)
    mults = map(mult, sys.isotopes)
    QuBase.LabelBasis((mults.^2)...)
end

function hilb2liouv(sys::System, x::QuArray, operator_type::String)
    h = QuBase.rawcoeffs(x)
    lb = liouv_basis(sys)
    unit = speye(size(h)...)
    if operator_type == "comm"
        l = kron(unit, h) - kron(transpose(h), unit)
        btup = (lb,lb)
    elseif operator_type == "acomm"
        l = kron(unit, h) + kron(transpose(h), unit)
        btup = (lb,lb)
    elseif operator_type == "left"
        l = kron(unit, h)
        btup = (lb,lb)
    elseif operator_type == "right"
        l = kron(transpose(h), unit)
        btup = (lb,lb)
    elseif operator_type == "statevec"
        l = h[:]
        btup = (lb)
    end
    QuArray(l, btup)
end

# get the rank k spherical tensor operators corresponding to a
# particular spin multiplicity
# also translated from Spinach
function irr_sph(mult::Int, k::Int)
    if k>0
        p = spinjp((mult-1)/2)
        m = spinjm((mult-1)/2)
        T = Vector{typeof(p)}(2*k+1)
        T[1] = ((-1)^k)*(2^(-k/2))*p^k
        for n=2:(2*k+1)
            q = k-n+2
            T[n] = (1/sqrt((k+q)*(k-q+1)))*(m*T[n-1]-T[n-1]*m)
        end
    elseif k==0
        T = [QuArray(speye(mult))]
    end
    T
end

# get the operator corresponding to some specification
# base form: supply a Vector{Int} with a # for the spherical tensor
# operator on each spin
# 0 = unit, 1 = T(1,1) = -sqrt(2)*I+, 2 = T(1,0) = Iz, 3 = T(1,-1) = sqrt(2)*I-
# and so on
function operator(sys::System, opspec::Vector{Int}, operator_type::String="comm")
    b = basis(sys)
    answer = 1
    for k=1:length(opspec)
        l = trunc(Int, sqrt(opspec[k]))
        m = l^2 + l - opspec[k]
        ist = irr_sph(mult(sys.isotopes[k]), l)
        answer = kron(answer, ist[l-m+1].coeffs)
    end
    @show answer
    @show b
    answer = QuArray(answer, (b,b))
    if sys.formalism == "Liouville"
        answer = hilb2liouv(sys, answer, operator_type)
    end
    answer
end

# get the state with some specification

# compute contribution to hamiltonian from a Zeeman coupling
# return the spinach style 1,9,25 matrices describing behavior of term
# under all kinds of rotations
