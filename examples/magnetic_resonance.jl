using QuBase
using QuDynamics

import Base.isinteger,
QuBase.spin,
Base.length,
Base.size,
Base.getindex,
Base.promote_rule,
Base.convert,
Base.zeros

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
    d = iso1.gamma * iso2.gamma * hbar * mu0 / (4*pi*(distance*1e-10)^3)
    mat = d * [1-3*ort[1]*ort[1]    -3*ort[1]*ort[2]   -3*ort[1]*ort[3];
               -3*ort[2]*ort[1]   1-3*ort[2]*ort[2]   -3*ort[2]*ort[3];
               -3*ort[3]*ort[1]    -3*ort[3]*ort[2]  1-3*ort[3]*ort[3]]
    mat = dscal1' * mat * dscal2
    mat = mat - (eye(3)*trace(mat)/3)
    mat = (mat + mat')/2
    Coupling{eltype(mat)}((c1.spin, c2.spin), mat, "")
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

function zeros(sys::System)
    b = basis(sys)
    m = length(b)
    answer = QuArray(spzeros(m,m), (b,b))
    if sys.formalism == "Liouville"
        answer = hilb2liouv(sys, answer, "left")
    end
    answer
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
    answer = QuArray(answer, (b,b))
    if sys.formalism == "Liouville"
        answer = hilb2liouv(sys, answer, operator_type)
    end
    answer
end

# easier specification with labels rather than numbers
function operator(sys::System, operators::Vector{String}, spins::Vector{Int}, operator_type::String="comm")
    opspec = fill(0, length(sys.isotopes))
    coeff = 1
    for idx=1:length(operators)
        if operators[idx] == "L+"
            coeff *= -sqrt(2)
            opspec[spins[idx]] = 1
        elseif operators[idx] == "Lz"
            coeff *= 1
            opspec[spins[idx]] = 2
        elseif operators[idx] == "L-"
            coeff *= sqrt(2)
            opspec[spins[idx]] = 3
        else
            r = match(r"T([\+\-]?\d+),([\+\-]?\d+)", operators[idx])
            if r != nothing
                (l,m) = map(x->parse(Int,x), r.captures)
                opspec[spins[idx]] = l^2 + l - m
                coeff = 1*coeff
            else
                error("unrecognized operator specification: $(operators[idx])")
            end
        end
    end
    coeff * operator(sys, opspec, operator_type)
end

# easy way to specify an operator for all spins of some isotope
function operator(sys::System, op::String, spins::String, operator_type::String="comm")
    matching = find([isotope.label == spins for isotope in sys.isotopes])
    @parallel (+) for n in matching
        operator(sys, [op], [n], operator_type)
    end
end

# generate the unit state
# identity matrix if Hilbert
# stretched identity matrix if Liouville
function unit_state(sys::System)
    b = basis(sys)
    rho = QuArray(speye(length(b)), (b,b))
    if sys.formalism == "Liouville"
        rho = hilb2liouv(sys, rho, "statevec")
        rho = rho/norm(rho.coeffs,2)
    end
    rho
end

# get the state with some specification
# returns a matrix in Hilbert space (density matrix)
# or a vector in Liouville space
# basic method is similar to operator
function state(sys::System, opspec::Vector{Int})
    operator(sys, opspec, "left") * unit_state(sys)
end

function state(sys::System, ops, spins)
    operator(sys, ops, spins, "left") * unit_state(sys)
end

# helper function, decide if something is significant
# user supplies the value and the tolerance to check it against
# list of tolerances defined in the function
function significant(num, tol_string::String)
    if tol_string == "inter_cutoff"
        tol = 1e-10
    end

    if norm(num) >= tol
        return true
    else
        return false
    end
end

# generic descriptor for a single term in the Hamiltonian
# specifies a product operator involving up to two spins
# and the isotropic (rank 0) and anisotropic (rank 2, five components)
immutable Descriptor
    nL::Int
    nS::Int
    opL::String
    opS::String
    isotropic::Float64
    ist_coeff::Vector{Float64}
    irr_comp::Vector{Float64}
end

function hamiltonian(sys::System, d::Descriptor, operator_type::String="comm")

    if d.nS == 0
        oper = operator(sys, [d.opL], [d.nL], operator_type)
    else
        oper = operator(sys, [d.opL,d.opS], [d.nL,d.nS], operator_type)
    end

    H = zeros(sys)
    if significant(d.isotropic, "inter_cutoff")
        H = d.isotropic * oper
    end

    Q1 = fill(zeros(sys), 3,3)
    Q2 = fill(zeros(sys), 5,5)

    for m=1:5, k=1:5
        if significant(d.ist_coeff[k]*d.irr_comp[m], "inter_cutoff")
            Q2[k,m] = d.ist_coeff[k] * d.irr_comp[m] * oper
        end
    end

    (H, [Q1,Q2])

end

# convert dcm to spherical tensor components
function mat2sphten{T<:Real}(M::Array{T,2})
    rank0 = -(1/sqrt(3)) * trace(M)
    rank1 = [-(1/2)*(M[3,1]-M[1,3]-1im*((M[3,2]-M[2,3]))),
             (1im/sqrt(2))*(M[1,2]-M[2,1]),
             -(1/2)*(M[3,1]-M[1,3]+1im*(M[3,2]-M[2,3]))]
    rank2 = [+(1/2)*(M[1,1]-M[2,2]-1im*(M[1,2]+M[2,1])),
             -(1/2)*(M[1,3]+M[3,1]-1im*(M[2,3]+M[3,2])),
             +(1/sqrt(6))*(2*M[3,3]-M[1,1]-M[2,2]),
             +(1/2)*(M[1,3]+M[3,1]+1im*(M[2,3]+M[3,2])),
             +(1/2)*(M[1,1]-M[2,2]+1im*(M[1,2]+M[2,1]))]
    (rank0, rank1, rank2)
end

# compute contribution to hamiltonian from a Zeeman coupling
# write a general descriptor which can be interpreted to the actual Hamiltonian
function hamiltonian(sys::System, z::Zeeman)
    nL = zeros(Int,3)
    opL = fill("E",3)
    nS = zeros(Int,3)
    opS = fill("E",3)
    isotropic = zeros(Float64, 3)
    ist_coeff = zeros(Float64, 3, 5)
    irr_comp = zeros(Float64, 3, 5)

    n = z.spin

    # isotropic part
    if z.strength in ["full", "z_full"]

        # keep the carrier
        zeeman_iso = trace(z.matrix)/3

        if significant(zeeman_iso, "inter_cutoff")
            nL[2] = n; opL[2] = "Lz"
            isotropic[2] = zeeman_iso
        end

    elseif z.strength in ["secular", "z_offs"]

        # subtract the carrier
        zeeman_iso = trace(z.matrix)/3 - (-sys.magnet * (sys.isotopes[z.spin]).gamma)
        if significant(zeeman_iso, "inter_cutoff")
            nL[2] = n; opL[2] = "Lz"
            isotropic[2] = zeeman_iso
        end

    end

    # anisotropic part

    phi_zeeman = mat2sphten(z.matrix)[3]

    if significant(phi_zeeman, "inter_cutoff")

        for k=1:3
            irr_comp[k,:] = phi_zeeman
        end

        if z.strength == "full"

            nL[1] = n; opL[1] = "L+"; ist_coeff[1,2] = -0.5
            nL[2] = n; opL[2] = "Lz"; ist_coeff[2,3] = sqrt(2/3)
            nL[3] = n; opL[3] = "L-"; ist_coeff[3,4] = 0.5

        elseif z.strength in ["secular", "z_full", "z_offs"]

            nL[2] = n; opL[2] = "Lz"; ist_coeff[2,3] = sqrt(2/3)

        elseif z.strength == "+"

            nL[1] = n; opL[1] = "L+"; ist_coeff[1,2] = -0.5

        elseif z.strength == "-"

            nL[3] = n; opL[3] = "L-"; ist_coeff[3,4] = 0.5

        end

    end

    mask = find(nL)

    [Descriptor(nL[x],nS[x],opL[x],opS[x],isotropic[x],ist_coeff[x,:],irr_comp[x,:]) for x in mask]

end

# process a quadrupolar coupling
function quad_ham(sys::System, c::Coupling)
    nL = zeros(Int,5)
    opL = fill("E",5)
    nS = zeros(Int,5)
    opS = fill("E",5)
    isotropic = zeros(Float64, 5)
    ist_coeff = zeros(Float64, 5, 5)
    irr_comp = zeros(Float64, 5, 5)

    n = c.spins[1]

    phi_quad = mat2sphten(c.matrix)[3]

    if significant(phi_quad, "inter_cutoff")

        if c.strength == "strong"

            nL[1] = n; opL[1] = "T2,+2"; ist_coeff[1,1] = 1
            nL[2] = n; opL[2] = "T2,+1"; ist_coeff[2,2] = 1
            nL[3] = n; opL[3] =  "T2,0"; ist_coeff[3,3] = 1
            nL[4] = n; opL[4] = "T2,-1"; ist_coeff[4,4] = 1
            nL[5] = n; opL[5] = "T2,-2"; ist_coeff[5,5] = 1

        elseif c.strength in ["secular", "T2,0"]

            nL[3] = n; opL[3] =  "T2,0"; ist_coeff[3,3] = 1

        elseif c.strength == "T2,+2"

            nL[1] = n; opL[1] = "T2,+2"; ist_coeff[1,1] = 1

        elseif c.strength == "T2,-2"

            nL[5] = n; opL[5] = "T2,-2"; ist_coeff[5,5] = 1

        elseif c.strength == "T2,-1"

            nL[4] = n; opL[4] = "T2,-1"; ist_coeff[4,4] = 1

        elseif c.strength == "T2,+1"

            nL[2] = n; opL[2] = "T2,+1"; ist_coeff[2,2] = 1

        end

    end

    mask = find(nL)

    [Descriptor(nL[x],nS[x],opL[x],opS[x],isotropic[x],ist_coeff[x,:],irr_comp[x,:]) for x in mask]

end

# process a bilinear coupling
function bilin_ham(sys::System, c::Coupling)
    nL = zeros(Int,3,3)
    opL = fill("E",3,3)
    nS = zeros(Int,3,3)
    opS = fill("E",3,3)
    isotropic = zeros(Float64,3,3)
    ist_coeff = zeros(Float64,3,3,5)
    irr_comp = zeros(Float64,3,3,5)

    l = c.spins[1]; s = c.spins[2]

    coupling_iso = trace(c.matrix)/3

    if significant(coupling_iso, "inter_cutoff")

        if c.strength in ["strong", "secular"]

            nL[2,2] = l, nS[2,2] = s; opL[2,2] = "Lz"; opS[2,2] = "Lz"; isotropic[2,2] = coupling_iso
            nL[1,3] = l, nS[1,3] = s; opL[1,3] = "L+"; opS[1,3] = "L-"; isotropic[1,3] = coupling_iso/2
            nL[3,1] = l, nS[3,1] = s; opL[3,1] = "L-"; opS[3,1] = "L+"; isotropic[3,1] = coupling_iso/2

        elseif c.strength in ["weak", "z*", "*z", "zz"]

            nL[2,2] = l, nS[2,2] = s; opL[2,2] = "Lz"; opS[2,2] = "Lz"; isotropic[2,2] = coupling_iso

        elseif c.strength == "+-"

            nL[1,3] = l, nS[1,3] = s; opL[1,3] = "L+"; opS[1,3] = "L-"; isotropic[1,3] = coupling_iso/2

        elseif c.strength == "-+"

            nL[3,1] = l, nS[3,1] = s; opL[3,1] = "L-"; opS[3,1] = "L+"; isotropic[3,1] = coupling_iso/2

        end
    end

    phi_coupling = mat2sphten(c.matrix)[3]

    if significant(phi_coupling, "inter_cutoff")
        for k=1:3, m=1:3
            irr_comp[k,m,:] = phi_coupling
        end

        if c.strength == "strong"

            nL[2,2] = l; nS[2,2] = s; opL[2,2] = "Lz"; opS[2,2] = "Lz"
            ist_coeff[2,2,3] = +sqrt(2/3)

            nL[1,3] = l; nS[1,3] = s; opL[1,3] = "L+"; opS[1,3] = "L-"
            ist_coeff[1,3,3] = -sqrt(2/3)/4

            nL[3,1] = l; nS[3,1] = s; opL[3,1] = "L-"; opS[3,1] = "L+"
            ist_coeff[3,1,3] = -sqrt(2/3)/4

            nL[2,1] = l; nS[2,1] = s; opL[2,1] = "Lz"; opS[2,1] = "L+"
            ist_coeff[2,1,2] = -1/2

            nL[1,2] = l; nS[1,2] = s; opL[1,2] = "L+"; opS[1,2] = "Lz"
            ist_coeff[1,2,2] = -1/2

            nL[2,3] = l; nS[2,3] = s; opL[2,3] = "Lz"; opS[2,3] = "L-"
            ist_coeff[2,3,4] = +1/2

            nL[3,2] = l; nS[3,2] = s; opL[3,2] = "L-"; opS[3,2] = "Lz"
            ist_coeff[3,2,4] = +1/2

            nL[1,1] = l; nS[1,1] = s; opL[1,1] = "L+"; opS[1,1] = "L+"
            ist_coeff[1,1,1] = +1/2

            nL[3,3] = l; nS[3,3] = s; opL[3,3] = "L-"; opS[3,3] = "L-"
            ist_coeff[3,3,3] = +1/2

        elseif c.strength == "z*"

            nL[2,2] = l; nS[2,2] = s; opL[2,2] = "Lz"; opS[2,2] = "Lz"
            ist_coeff[2,2,3] = +sqrt(2/3)

            nL[2,1] = l; nS[2,1] = s; opL[2,1] = "Lz"; opS[2,1] = "L+"
            ist_coeff[2,1,2] = -1/2

            nL[2,3] = l; nS[2,3] = s; opL[2,3] = "Lz"; opS[2,3] = "L-"
            ist_coeff[2,3,4] = +1/2

        elseif c.strength == "*z"

            nL[2,2] = l; nS[2,2] = s; opL[2,2] = "Lz"; opS[2,2] = "Lz"
            ist_coeff[2,2,3] = +sqrt(2/3)

            nL[1,2] = l; nS[1,2] = s; opL[1,2] = "L+"; opS[1,2] = "Lz"
            ist_coeff[1,2,2] = -1/2

            nL[3,2] = l; nS[3,2] = s; opL[3,2] = "L-"; opS[3,2] = "Lz"
            ist_coeff[3,2,4] = +1/2

        elseif c.strength in ["secular", "T2,0"]

            nL[2,2] = l; nS[2,2] = s; opL[2,2] = "Lz"; opS[2,2] = "Lz"
            ist_coeff[2,2,3] = +sqrt(2/3)

            nL[1,3] = l; nS[1,3] = s; opL[1,3] = "L+"; opS[1,3] = "L-"
            ist_coeff[1,3,3] = -sqrt(2/3)/4

            nL[3,1] = l; nS[3,1] = s; opL[3,1] = "L-"; opS[3,1] = "L+"
            ist_coeff[3,1,3] = -sqrt(2/3)/4

        elseif c.strength in ["weak", "zz"]

            nL[2,2] = l; nS[2,2] = s; opL[2,2] = "Lz"; opS[2,2] = "Lz"
            ist_coeff[2,2,3] = +sqrt(2/3)

        elseif c.strength == "z+"

            nL[2,1] = l; nS[2,1] = s; opL[2,1] = "Lz"; opS[2,1] = "L+"
            ist_coeff[2,1,2] = -1/2

        elseif c.strength == "+z"

            nL[1,2] = l; nS[1,2] = s; opL[1,2] = "L+"; opS[1,2] = "Lz"
            ist_coeff[1,2,2] = -1/2

        elseif c.strength == "T2,+1"

            nL[2,1] = l; nS[2,1] = s; opL[2,1] = "Lz"; opS[2,1] = "L+"
            ist_coeff[2,1,2] = -1/2

            nL[1,2] = l; nS[1,2] = s; opL[1,2] = "L+"; opS[1,2] = "Lz"
            ist_coeff[1,2,2] = -1/2

        elseif c.strength == "z-"

            nL[2,3] = l; nS[2,3] = s; opL[2,3] = "Lz"; opS[2,3] = "L-"
            ist_coeff[2,3,4] = +1/2

        elseif c.strength == "-z"

            nL[3,2] = l; nS[3,2] = s; opL[3,2] = "L-"; opS[3,2] = "Lz"
            ist_coeff[3,2,4] = +1/2

        elseif c.strength == "T2,-1"

            nL[2,3] = l; nS[2,3] = s; opL[2,3] = "Lz"; opS[2,3] = "L-"
            ist_coeff[2,3,4] = +1/2

            nL[3,2] = l; nS[3,2] = s; opL[3,2] = "L-"; opS[3,2] = "Lz"
            ist_coeff[3,2,4] = +1/2

        elseif c.strength == "+-"

            nL[1,3] = l; nS[1,3] = s; opL[1,3] = "L+"; opS[1,3] = "L-"
            ist_coeff[1,3,3] = -sqrt(2/3)/4

        elseif c.strength == "-+"

            nL[3,1] = l; nS[3,1] = s; opL[3,1] = "L-"; opS[3,1] = "L+"
            ist_coeff[3,1,3] = -sqrt(2/3)/4

        elseif c.strength in ["++", "T2,+2"]

            nL[1,1] = l; nS[1,1] = s; opL[1,1] = "L+"; opS[1,1] = "L+"
            ist_coeff[1,1,1] = +1/2

        elseif c.strength in ["--", "T2,-2"]

            nL[3,3] = l; nS[3,3] = s; opL[3,3] = "L-"; opS[3,3] = "L-"
            ist_coeff[3,3,3] = +1/2

        end
    end

    # reshape everything so it's easier to sort through
    nL = reshape(nL, 9)
    nS = reshape(nS, 9)
    opL = reshape(opL, 9)
    opS = reshape(opS, 9)
    isotropic = reshape(isotropic, 9)
    ist_coeff = reshape(ist_coeff, 9,5)
    irr_comp = reshape(irr_comp, 9, 5)

    mask = find(nL)

    [Descriptor(nL[x],nS[x],opL[x],opS[x],isotropic[x],ist_coeff[x,:],irr_comp[x,:]) for x in mask]

end

# process a coupling
function hamiltonian(sys::System, c::Coupling)
    # decide if it's a quadrupolar coupling
    if c.spins[1] == c.spins[2]
        answer = quad_ham(sys, c)
    else                        # it's a bilinear coupling
        answer = bilin_ham(sys, c)
    end
    answer
end

function cleanup!(sys::System)

    # easy access
    nspins = length(sys.isotopes)

    # check what the user provided
    provided = [x.spin for x in sys.zeeman]
    # make sure no duplicates
    if length(union(provided)) != length(provided)
        error("multiple zeeman interactions on a single spin.")
    end

    # for spins which haven't been explicitly given a Zeeman
    for n=1:nspins
        if !any(provided .== n)
            push!(sys.zeeman, Zeeman(n,0))
        end
    end

    # scale zeeman terms as needed, and keep terms for scaling dipolar interactions.
    ddscal = fill(eye(3), nspins)
    for i=1:length(sys.zeeman)
        ddscal[i] = ddscal!(sys.isotopes[i], sys.zeeman[i], sys.magnet)
    end

    @show ddscal

    # check provided coordinates
    provided = [x.spin for x in sys.coords]
    # make sure no duplicates
    if length(union(provided)) != length(provided)
        error("multiple coordinates provided for a single spin.")
    end

    for i=1:length(provided)
        for j=(i+1):length(provided)
            push!(sys.couplings, dipolar(sys.isotopes[provided[i]], sys.isotopes[provided[j]], sys.coords[i], sys.coords[j], ddscal[provided[i]], ddscal[provided[j]]))
        end
    end
end

function assume!(sys::System, assumptions::String)
    if assumptions == "nmr"

        # all zeeman interactions should be secular
        for x in sys.zeeman
            x.strength = "secular"
        end

        # couplings
        for x in sys.couplings
            if sys.isotopes[x.spins[1]].label == sys.isotopes[x.spins[2]].label
                x.strength = "secular"
            else                # heteronuclear
                x.strength = "weak"
            end
        end

    elseif assumptions in ["esr", "deer"]

        for x in sys.zeeman

            if sys.isotopes[x.spin].label == "E"
                x.strength = "secular"
            else
                x.strength = "full"
            end

        end

        for x in sys.couplings

            if (sys.isotopes[x.spins[1]] == "E") && (sys.isotopes[x.spins[2]] != "E")

                x.strength = "z*"

            elseif (sys.isotopes[x.spins[1]] != "E") && (sys.isotopes[x.spins[2]] == "E")

                x.strength = "*z"

            elseif (sys.isotopes[x.spins[1]] == "E") && (sys.isotopes[x.spins[2]] == "E")

                x.strength = "secular"

            else

                x.strength = "strong"

            end
        end

    elseif assumptions == "labframe"

        for x in sys.zeeman
            x.strength = "full"
        end
        for x in sys.couplings
            x.strength = "strong"
        end
    end
end

function hamiltonian!(sys::System, assumption::String, operator_type::String="comm")
    assume!(sys, assumption)

    D = Vector{Descriptor}()

    for x in sys.zeeman
        push!(D, hamiltonian(sys, x)...)
    end

    for x in sys.couplings
        push!(D, hamiltonian(sys, x)...)
    end

    if length(D) == 0
        H = zeros(sys)
        Q1 = fill(zeros(sys),3,3)
        Q2 = fill(zeros(sys),5,5)
    else
        local_hamiltonians = pmap(x->hamiltonian(sys, x, operator_type), D)
        (H,(Q1,Q2)) = local_hamiltonians[1]
        for n=2:length(local_hamiltonians)
            H += local_hamiltonians[n][1]
            broadcast!(+,Q1,Q1,local_hamiltonians[n][2][1])
            broadcast!(+,Q2,Q2,local_hamiltonians[n][2][2])
        end

    end

    (H,[Q1,Q2])
end
