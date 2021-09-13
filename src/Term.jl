"""
This type represents an interaction term in the energy function of the form

``β_i (\\frac12 x'Ax + x'y + c) + M_i \\log β_i``

The complete energy function is given by

``∑_i β_i (\\frac12 x' A_i x + x' y_i + c_i) + M_i \\log β_i``

as is represented by an Vector{Term}. Note that c and M are only needed for paramenter learning
"""
mutable struct Term{T <: Real}
    A::Matrix{T}
    y::Vector{T}
    c::T
    β::T
    # for parameter learning
    δβ::T
    M::Int
end

Term(A,y,β = one(eltype(A))) = Term{eltype(A)}(A,y,zero(eltype(A)),eltype(A)(β),zero(eltype(A)),0)

function (t::Term)(v::Vector)
    return v⋅(t.A*v-2*t.y) + t.c
end

function updateβ(t::Term{T}, v) where T
    if t.δβ > 0
        t.β = t.δβ * t.M / t(v) + (1-t.δβ) * t.β
    end
end

function sum!(A::Matrix{T}, y::Vector{T}, H::Vector{Term{T}}) where T <: Real
    fill!(A, zero(T))
    fill!(y, zero(T))
    for i=1:length(H)
        A .+= H[i].β * H[i].A
        y .+= H[i].β * H[i].y
    end
end

struct TermRBM{T<:Real}
    w::Matrix{T}
    y::Vector{T}
    β::T
end

TermRBM(w,y,β=one(eltype(w))) = TermRBM{eltype(w)}(w,y,eltype(w)(β))

function sum!(A::Matrix{T}, y::Vector{T}, H::Vector{TermRBM{T}}) where T <: Real
    fill!(A, zero(T))
    fill!(y, zero(T))
    W = zeros(T,length(y),length(y))
    for i=1:length(H)
        W[1:size(H[i].w,1),size(H[i].w,1)+1:end] = H[i].w
        W[size(H[i].w,1)+1:end,1:size(H[i].w,1)] = H[i].w'
        A .-= H[i].β * W
        y .+= H[i].β * H[i].y
    end
end

