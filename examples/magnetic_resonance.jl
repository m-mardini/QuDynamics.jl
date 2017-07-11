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

# a type for a particular Zeeman coupling
# we need it to be mutable so we can change the strength flag
# it contains the label of the spin as well
mutable struct Zeeman{T<:Real} <: AbstractMatrix{T}
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

promote_rule{T<:AbstractMatrix}(::Type{T}, ::Type{Zeeman}) = T
convert{T<:AbstractMatrix}(::Type{T}, x::Zeeman) = convert(T, x.matrix)

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

mutable struct Coord{T<:Real} <: AbstractVector{T}
    spin::Int
    pos::Vector{T}
    function Coord(spin::Int, pos::Vector{T}) where {T<:Real}
        new{T}(spin, pos)
    end
end

promote_rule{T<:AbstractVector}(::Type{T}, ::Type{Coord}) = T
convert{T<:AbstractVector}(::Type{T}, x::Coord) = convert(T, x.pos)

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

# construct a label basis for the system.
# in Hilbert space, we'll use a simple basis of z states
# in Liouville space, we'll use irreducible spherical tensors
# if dimensionality is n in Hilbert, it's n^2 in Liouville
function basis(sys::System)
    mults = map(x->Int(2*QuBase.spin_value(spin(x))+1), sys.isotopes)
    if sys.formalism == "Hilbert"
        b = QuBase.LabelBasis(mults...)
    elseif sys.formalism == "Liouville"
        b = QuBase.LabelBasis((mults.^2)...)
    end
    b
end

# get the operator corresponding to some specification

# get the state with some specification

# compute contribution to hamiltonian from a Zeeman coupling
# return the spinach style 1,9,25 matrices describing behavior of term
# under all kinds of rotations
    
