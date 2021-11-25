using Random, LinearAlgebra, ExtractMacro

```@meta
CurrentModule = GaussianEP
```

function update_err!(dst, i, val)
    r=abs(val - dst[i])
    dst[i] = val
    return r
end

"""
    Instantaneous state of an expectation propagation run.
"""
struct EPState{T<:AbstractFloat}
    A::Matrix{T}
    y::Vector{T}
    Σ::Matrix{T}
    v::Vector{T}
    av::Vector{T}
    va::Vector{T}
    a::Vector{T}
    μ::Vector{T}
    b::Vector{T}
    s::Vector{T}
end
EPState{T}(N, Nx = N) where {T <: AbstractFloat} = EPState{T}(Matrix{T}(undef,Nx,Nx), zeros(T,Nx), Matrix{T}(undef,Nx,Nx), zeros(T,Nx),zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N), ones(T,N), ones(T,N))

"""
Output of EP algorithm

"""
mutable struct EPOut{T<:AbstractFloat}
    av::Vector{T}
    va::Vector{T}
    μ::Vector{T}
    s::Vector{T}
    converged::Symbol
    state::EPState{T}
end
function EPOut(s, converged::Symbol) where {T <: AbstractFloat}
    converged ∈ (:converged,:unconverged) || error("$converged is not a valid symbol")
    return EPOut(s.av,s.va, s.μ,s.s,converged,s)
end

"""
    expectation_propagation(H::Vector{Term{T}}, P0::Vector{Prior}, F::AbstractMatrix{T} = zeros(0,length(P0)), d::Vector{T} = zeros(size(F,1));
        maxiter::Int = 2000,
        callback = (x...)->nothing,
        # state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]),
        damp::T = 0.9,
        epsconv::T = 1e-6,
        maxvar::T = 1e50,
        minvar::T = 1e-50,
        inverter::Function = inv) where {T <: Real, P <: Prior}


EP for approximate inference of

``P( \\bf{x} )=\\frac1Z exp(-\\frac12\\bf{x}' A \\bf{x} + \\bf{x'} \\bf{y}))×\\prod_i p_{i}(x_i)``

Arguments:

* `A::Array{Term{T}}`: Gaussian Term (involving only x)
* `P0::Array{Prior}`: Prior terms (involving x and y)
* `F::AbstractMatrix{T}`: If included, the unknown becomes ``(\\bf{x} ,\\bf{y} )^T`` and a term ``\\delta(F \\bf{x}+\\bf{d}-\\bf{y})`` is added.

Optional named arguments:

* `maxiter::Int = 2000`: maximum number of iterations
* `callback = (x...)->nothing`: your own function to report progress, see [`ProgressReporter`](@ref)
* `state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2])`: If supplied, all internal state is updated here
* `damp::T = 0.9`: damping parameter
* `epsconv::T = 1e-6`: convergence criterion
* `maxvar::T = 1e50`: maximum variance
* `minvar::T = 1e-50`: minimum variance
* `inverter = inv`: inverter method

# Example

```jldoctest
julia> t=Term(zeros(2,2),zeros(2),1.0)
Term{Float64}([0.0 0.0; 0.0 0.0], [0.0, 0.0], 0.0, 1.0, 0.0, 0)

julia> P=[IntervalPrior(i...) for i in [(0,1),(0,1),(-2,2)]]
3-element Array{IntervalPrior{Int64},1}:
 IntervalPrior{Int64}(0, 1)
 IntervalPrior{Int64}(0, 1)
 IntervalPrior{Int64}(-2, 2)

julia> F=[1.0 -1.0];

julia> res = expectation_propagation([t], P, F)
GaussianEP.EPOut{Float64}([0.499997, 0.499997, 3.66527e-15], [0.083325, 0.083325, 0.204301], [0.489862, 0.489862, 3.66599e-15], [334.018, 334.018, 0.204341], :converged, EPState{Float64}([9.79055 -0.00299477; -0.00299477 9.79055], [0.0, 0.0], [0.102139 3.12427e-5; 3.12427e-5 0.102139], [0.489862, 0.489862], [0.499997, 0.499997, 3.66527e-15], [0.083325, 0.083325, 0.204301], [0.490876, 0.490876, -1.86785e-17], [0.489862, 0.489862, 3.66599e-15], [0.100288, 0.100288, 403.599], [334.018, 334.018, 0.204341]))
```
"""
function expectation_propagation(H::AbstractVector{Term{T}}, P0::AbstractVector{P};
                     F::AbstractMatrix{T} = zeros(T,0,length(P0)),
                     d::AbstractVector{T} = zeros(T,size(F,1)),
                     maxiter::Int = 2000,
                     callback = (x...)->nothing,
                     state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]),
                     damp::T = T(0.9),
                     epsconv::T = T(1e-6),
                     maxvar::T = T(1e50),
                     minvar::T = T(1e-50),
                     inverter = inv) where {T <: Real, P <: Prior}
    @extract state A y Σ v av va a μ b s
    Ny,Nx = size(F)
    N = Nx + Ny
    @assert size(P0,1) == N
    Fp = copy(F')
    for iter = 1:maxiter
        sum!(A,y,H)
        Δμ, Δs, Δav, Δva = 0.0, 0.0, 0.0, 0.0
        A .+= Diagonal(1 ./ b[1:Nx]) .+ Fp * (Diagonal(1 ./ b[Nx+1:end]) * F)
        Σ .= inverter(A)
        ax, bx, ay, by = (@view a[1:Nx]), (@view b[1:Nx]), (@view a[Nx+1:end]), (@view b[Nx+1:end])
        v .= Σ * (y .+ ax ./ bx .+ (Fp * ((ay-d) ./ by)))
        for i in 1:N
            if i <= Nx
                ss = clamp(Σ[i,i], minvar, maxvar)
                vv = v[i]
            else
                ss = clamp(dot(F[i-Nx,:], Σ*Fp[:,i-Nx]), minvar, maxvar)
                vv = dot(Fp[:,i-Nx], v) + d[i-Nx]
            end

            if ss < b[i]
                Δs = max(Δs, update_err!(s, i, clamp(1/(1/ss - 1/b[i]), minvar, maxvar)))
                Δμ = max(Δμ, update_err!(μ, i, s[i] * (vv/ss - a[i]/b[i])))
            else
                ss == b[i] && @warn "infinite var, ss = $ss"
                Δs = max(Δs, update_err!(s, i, maxvar))
                Δμ = max(Δμ, update_err!(μ, i, 0))
            end
            tav, tva = moments(P0[i], μ[i], sqrt(s[i]));
            Δav = max(Δav, update_err!(av, i, tav))
            Δva = max(Δva, update_err!(va, i, tva))
            (isnan(av[i]) || isnan(va[i])) && @warn "avnew = $(av[i]) varnew = $(va[i])"

            new_b = clamp(1/(1/va[i] - 1/s[i]), minvar, maxvar)
            new_a = av[i] + new_b * (av[i] - μ[i])/s[i]
            a[i] = damp * a[i] + (1 - damp) * new_a
            b[i] = damp * b[i] + (1 - damp) * new_b
        end

        # learn prior's params
        for i in randperm(N)
            gradient(P0[i], μ[i], sqrt(s[i]));
        end
        # learn β params
        for i in 1:length(H)
            updateβ(H[i], av[1:Nx])
        end
        ret = callback(iter,state,Δav,Δva,epsconv,maxiter,H,P0)
        if ret === true || (Δav < epsconv && norm(F*av[1:Nx]+d-av[Nx+1:end]) < 1e-4)
            return EPOut(state, :converged)
        end
    end
    return EPOut(state, :unconverged)
end


"""
    expectation_propagation(H::Vector{TermRBM{T}}, P0::Vector{Prior}, F::AbstractMatrix{T} = zeros(0,length(P0)), d::Vector{T} = zeros(size(F,1));
        maxiter::Int = 2000,
        callback = (x...)->nothing,
        # state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]),
        damp::T = 0.9,
        epsconv::T = 1e-6,
        maxvar::T = 1e50,
        minvar::T = 1e-50,
        inverter::Function = inv) where {T <: Real, P <: Prior}


EP for approximate inference of

``P( \\bf{x} )=\\frac1Z exp(-\\frac12\\bf{x}' W \\bf{x} + \\bf{x'} \\bf{y}))×\\prod_i p_{i}(x_i)``

x=(v,h) overall vector of hidden and visible variables of size respectively Nv, Nh: Nv+Nh=Nx

Arguments:

* `H::Array{TermRBM{T}}`: Gaussian TermRBM (involving only x=(v,h))
* `P0::Array{Prior}`: Prior terms (involving x=(v,h) and possibly y)
* `F::AbstractMatrix{T}`: If included, the unknown becomes ``(\\bf{x} ,\\bf{y} )^T`` and a term ``\\delta(F \\bf{x}+\\bf{d}-\\bf{y})`` is added.

Optional named arguments:

* `maxiter::Int = 2000`: maximum number of iterations
* `callback = (x...)->nothing`: your own function to report progress, see [`ProgressReporter`](@ref)
* `state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2])`: If supplied, all internal state is updated here
* `damp::T = 0.9`: damping parameter
* `epsconv::T = 1e-6`: convergence criterion
* `maxvar::T = 1e50`: maximum variance
* `minvar::T = 1e-50`: minimum variance
* `inverter = block_inv`: inverter method
* `epsgrad::T = 1.0e-2`: tolerance over prior parameters

"""
function expectation_propagation(H::AbstractVector{TermRBM{T}}, P0::AbstractVector{P}; 
                     F::AbstractMatrix{T} = zeros(T,0,length(P0)),
                     d::AbstractVector{T} = zeros(T,size(F,1)),
                     maxiter::Int = 2000,
                     callback = (x...)->nothing,
                     state::Union{EPState{T},Nothing} = nothing,
                     damp::T = T(0.9),
                     epsconv::T = 1.0e-2 * (1.0 - damp),
                     maxvar::T = T(1e50),
                     minvar::T = T(-1e50),
                     nprint::Int = 100,
                     inverter::Symbol = :block_inv,
                     epsgrad::T = 1.0e-2) where {T <: Real, P <: Prior}
    
    Ny,Nx = size(F)
    N = Nx + Ny
    @assert size(P0,1) == N
    Fp = copy(F')
    Nv, Nh = size(H[1].w)
    @assert Nv+Nh == Nx
    
    flag = 0
    c = if state === nothing
        state = EPState{T}(sum(size(F)), size(F)[2])
        flag = 1
        min_diagel(H[1].w,P0[1],P0[Nv+1]) 
    end

    @extract state A y Σ v av va a μ b s

    if flag == 1
        b .= 1 ./ c
    end

    fail = 0

    for iter = 1:maxiter
        sum!(A,y,H)
        Δgrad, Δμ, Δs, Δav, Δva = 0.0, 0.0, 0.0, 0.0, 0.0
        _, C = Diagonal(1 ./b[1:Nv]), Diagonal(1 ./ b[Nv+1:Nx])
        Bm1 = Diagonal(b[1:Nv])
        if inverter == :block_inv
            Σ .= block_inv(A[1:Nv,Nv+1:Nx],Bm1,C)
        else
            A .+= Diagonal(1 ./ b[1:Nx]) .+ Fp * (Diagonal(1 ./ b[Nx+1:end]) * F)
            Σ = inv(Symmetric(A))
            @assert isposdef(Σ)
        end
        ax, bx, ay, by = (@view a[1:Nx]), (@view b[1:Nx]), (@view a[Nx+1:end]), (@view b[Nx+1:end])
        v .= Σ * (y .+ ax ./ bx .+ (Fp * ((ay-d) ./ by)))
        
        for i in 1:N
            if i <= Nx
                ss = clamp(Σ[i,i], minvar, maxvar)
                vv = v[i]
            else
                ss = clamp(dot(F[i-Nx,:], Σ*Fp[:,i-Nx]), minvar, maxvar)
                vv = dot(Fp[:,i-Nx], v) + d[i-Nx]
            end
            Δs = max(Δs, update_err!(s, i, clamp(1/(1/ss - 1/b[i]), minvar, maxvar)))
            Δμ = max(Δμ, update_err!(μ, i, s[i] * (vv/ss - a[i]/b[i])))
            tav, tva = try
                moments(P0[i], μ[i], s[i])
            catch err
                if isa(err,DomainError)
                    println("combined variance must be positive")
                    fail = 1
                    return EPOut(state, :unconverged), fail, iter
                end
            end
            #tav, tva = moments(P0[i], μ[i], s[i])
            Δav = max(Δav, update_err!(av, i, tav))
            Δva = max(Δva, update_err!(va, i, tva))
            (isnan(av[i]) || isnan(va[i])) && @warn "avnew = $(av[i]) varnew = $(va[i])"

            new_b = clamp(1/(1/va[i] - 1/s[i]), minvar, maxvar)
            new_a = av[i] + new_b * (av[i] - μ[i])/s[i]
            a[i] = damp * a[i] + (1 - damp) * new_a
            b[i] = damp * b[i] + (1 - damp) * new_b
        end

        # learn prior's params
        for i in 1:Nv+Nh
            Δgrad = max(Δgrad,gradient(P0[i], μ[i], s[i]));
        end
        # learn β params
        # for i in 1:length(H)
        #     updateβ(H[i], av[1:Nx])
        # end
        ret = callback(iter,state,Δav,Δva,epsconv,maxiter,H)
        if mod(iter, nprint) == 0
            println("it: ", iter, " Δav: ", Δav, " Δgrad: ", Δgrad)
        end
        if ret === true || (Δav < epsconv && norm(F*av[1:Nx]+d-av[Nx+1:end]) < 1e-4 && Δgrad < epsgrad)
            #println("it: ", iter, " Δav: ", Δav)
            return EPOut(state, :converged), fail, iter
        end
    end
    return EPOut(state, :unconverged), fail, maxiter
end

function block_inv(w::Matrix{T}, Bm1::Diagonal{T,Array{T,1}}, C::Diagonal{T,Array{T,1}}) where {T <: AbstractFloat}

    @assert size(w,1) == size(Bm1,1) && size(w,2) == size(C,1)

    N,M = size(w)
    Σ = zeros(T,N+M,N+M)
    D = inv(C - w' * Bm1 * w )

    Σ[1:N,1:N] .= Bm1 + Bm1 * w * D * w' * Bm1
    Σ[N+1:N+M,N+1:N+M] .= D
    Σ[1:N,N+1:N+M] .= -Bm1 * w * D 
    Σ[N+1:N+M,1:N] .= Σ[1:N,N+1:N+M]'

    #return Symmetric(Σ)
    return Σ
end

function min_diagel(w::Matrix{Float64}, Pv::P1, Ph::P2; ϵ::Float64=0.5) where {P1 <: Prior, P2 <: Prior}

    N = size(w,1)
    M = size(w,2)

    c = zeros(Float64,N+M)
    W = zeros(Float64,N,N)
    W = w*w'

    λ_max = eigmax(W)
    c .= sqrt(λ_max)+ϵ

    return c

end

function min_diagel(w::Matrix{Float64}, Pv::BinaryPrior, Ph::ReLUPrior; ϵ::Float64=5.0)

    N = size(w,1)
    M = size(w,2)

    c = zeros(N+M)
    W = zeros(Float64,N,N)
    W = w*w'

    λ_max = eigmax(W)

    b = 1 / (Pv.ρ * (1 - Pv.ρ))
    d = λ_max / b

    c[1:N] .= b
    c[N+1:N+M] .= d+ϵ

    return c

end

function min_diagel(w::Matrix{Float64}, Pv::GaussianPrior, Ph::ReLUPrior; ϵ::Float64=0.5)

    N = size(w,1)
    M = size(w,2)

    c = zeros(N+M)
    W = zeros(Float64,N,N)
    W = w*w'

    λ_max = eigmax(W)

    b = Pv.β    
    d = λ_max / b

    c[1:N] .= b
    c[N+1:N+M] .= d+ϵ

    return c

end
