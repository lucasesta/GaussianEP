"""

    In this file we define some functions to numerically compute the moments
    for some case of interest to the application of EP to RBM. 

"""


# Binary visible-Gaussian hidden: Bernoulli visible variables
# defined by parameter θ_B ϕ(v)=exp[-θ_B]/(1+exp[-θ_B]).
# Gaussian hidden units of distribution ψ(h)∝exp[-γ*h^2/2+θ*h] 


#Function defining fields and coupling from 
function CoupField(w::Matrix{T}, Pv::Vector{BinaryPrior{T}}, Ph::Vector{GaussianPrior{T}}) where {T<:Real}

    N, M = size(w)

    ρ = Pv[1].ρ
    θ = zeros(T,M)
    γ = zeros(T,M)

    # Initialize vectors of mean and inverse variance for hidden units
    for m=1:M
        γ[m] = Ph[m].β
        θ[m] = Ph[m].μ*Ph[m].β
    end

    h = zeros(T,N)
    J = zeros(T,N,N)

    # Bernoulli exponent of visible units
    θ_B = log(ρ/(1-ρ))

    # Field definition
    for i=1:N
        h[i] -= θ_B
        for m=1:M
            h[i] -= w[i,m]*(θ[m]-0.5*w[i,m])/γ[m]
        end
    end

    #Coupling definition
    for i=1:N-1
        for j=i+1:N
            for m=1:M
                J[i,j] += w[i,m]*w[j,m]/γ[m] 
            end
            J[j,i] = J[i,j]'
        end
    end

    return h,J

end

#Function to compute the normalization factor
function comp_z(h::Vector{T}, J::Matrix{T}) where {T<:Real}

    Z = 0.0
    N = length(h)
    space = [0:1 for i=1:N]
    conf = collect(Iterators.product(space...))

    for c in conf[:]
        v = [c...]
        E = 0.0
        for i=1:N-1
            E += v[i]*h[i]
            for j=i+1:N
                E += v[i]*v[j]*J[i,j]
            end
        end
        E += v[end]*h[end]
        Z += exp(E)
    end

    return Z

end


#Function to compute the average value of vᵢ (i=idx)
function average_v(h::Vector{T}, J::Matrix{T}, idx::Int64) where {T<:Real}

    N = length(h)
    av = 0.0
    Z = comp_z(h,J)

    space = [0:1 for i=1:N]
    conf = collect(Iterators.product(space...))
    idx_av = findall(x->x[idx]==1,conf[:])

    for c in conf[idx_av]
        E = 0.0
        v = [c...]
        for i=1:N
            E += h[i]*v[i]
            if i<N
                for j=i+1:N
                    E += J[i,j]*v[i]*v[j]
                end
            end
        end
        av += exp(E)
    end

    av /= Z

    return av

end

# Function to compute the cross correlation ⟨vᵢ*hₘ⟩

function average_vh(w::Matrix{T}, Pv::Vector{BinaryPrior{T}}, Ph::Vector{GaussianPrior{T}}, idx_v::Int64, idx_m::Int64) where {T<:Real}

    N = length(Pv)
    xc = 0.0

    θ = Ph[idx_m].μ
    γ = Ph[idx_m].β

    h, J = CoupField(w,Pv,Ph)

    av_v = average_v(h,J,idx_v)

    xc += av_v*(θ-w[idx_v,idx_m])

    for i in setdiff(1:N,idx_v)
        xc += w[i,idx_m]*average_vv(h,J,idx_v,i)
    end
    
    xc /= γ

    return xc

end

# Function to compute visibile-visible correlation ⟨vᵢvⱼ⟩

function average_vv(h::Vector{T}, J::Matrix{T}, idx1::Int64, idx2::Int64) where {T<:Real}

    N = length(h)
    av_v1v2 = 0.0
    Z = comp_z(h,J)

    space = [0:1 for i=1:N]
    conf = collect(Iterators.product(space...))
    idx_av = findall(x->x[idx1]==1 && x[idx2]==1,conf[:])

    for c in conf[idx_av]
        E = 0.0
        v = [c...]
        for i=1:N
            E += h[i]*v[i]
            if i<N
                for j=i+1:N
                    E += J[i,j]*v[i]*v[j]
                end
            end
        end
        av_v1v2 += exp(E)
    end

    av_v1v2 /= Z

    return av_v1v2

end

# Function to compute hidden average value ⟨hₘ⟩

function average_h(h::Vector{T}, J::Matrix{T}, Ph::GaussianPrior{T}, w::Matrix{T}, idx::Int64) where {T<:Real}

    N = length(h)
    av_h = 0.0
    γ = Ph.β
    θ = Ph.μ*Ph.β

    av_h += θ

    for i=1:N
        av_h -= w[i,idx]*average_v(h,J,i)
    end

    av_h /= γ

    return av_h

end

function average_h2(h::Vector{T}, J::Matrix{T}, Ph::GaussianPrior,w::Matrix{T},idx::Int64) where {T<:Real}

    N = length(h)
    av_h2 = 0.0
    γ = Ph.β
    θ = Ph.μ*Ph.β

    av_h2 += θ^2

    for i=1:N
        av_h2 -= 2 * θ * w[i,idx]*average_v(h,J,i)
        for j=1:N
            av_h2 += w[i,idx]*w[j,idx]*average_vv(h,J,i,j)
        end
    end

    av_h2 /= γ^2
    av_h2 += (1/γ)

    return av_h2

end