

"""

    Function realizing the move to be accepted

"""

function Move(x::Float64, Δ::Float64)

    x += 2.0*Δ*(rand()-0.5)
    return x
end

function MoveBin(x::Float64)
   
    if x == 0
        return 1
    else
        return 0
    end
end

function MoveSpAndSlab(x::Float64, Δ::Float64, ρ::Float64)

    r = rand()
    if r < 1.0-ρ
        x = 0.0
    else
        x += 2.0*Δ*(rand()-0.5)
    end

    return x
end

"""

    Functions determining whether the move is accepted
    or not. Different methods according to the prior type

"""

function Accept(y::Array{Float64,1},w::Array{Float64,1},P::Prior, x::Float64, x_old::Float64)

    ΔE = EnerVar(y,w,P,x,x_old)
    r = rand()
    a = exp(-ΔE)


    if r<1 && r > a
        x = x_old
        return x, 0.0
    else
        return x, 1.0
    end

end

"""

    Function computing energy variation between old
    and proposed state

"""

function EnerVar(y::Array{Float64,1},w::Array{Float64,1},P::Prior, x::Float64, x_old::Float64)

    ΔE = 0.0
    N_c = length(y)
    
    for k=1:N_c
        ΔE -= w[k]*y[k]
    end

    ΔE *= (x-x_old)
    #println("weights-dep energy ", ΔE)
    ΔE += Pot(P,x)-Pot(P,x_old)
    #println(P, " ", x, " ", Pot(P,x))
    #println(P, " ", x_old, " ", Pot(P,x_old))
    
    return ΔE

end

function energy(w::Matrix{T}, x::Vector{T}, Pv::Vector{P1}, Ph::Vector{P2}) where {T <: Real, P1 <: Prior, P2 <: Prior}

    N,M = (size(w,1),size(w,2))

    e_v = - pot(view(x,1:N),Pv)

    e_h = - pot(view(x,N+1:N+M),Ph)

    e_w = - dot(view(x,1:N),w*view(x,N+1:N+M))

    e_tot = e_w + e_v + e_h

    return e_v, e_h, e_w, e_tot

end

function pot(x, P::Vector{BinaryPrior{T}}) where T <: Real

    L = length(x)
    θ_B = zeros(L)
    θ_B = map(x->log((1-x.ρ)/x.ρ), P)
    return dot(θ_B,x)

end

function pot(x, P::Vector{ReLUPrior{T}}) where T <: Real

    L = length(x)
    θ = zeros(L)
    γ = zeros(L)

    θ = map(x->x.θ,P)
    γ = map(x->x.γ,P)
    return sum(-0.5 * γ .* x.^2 + θ .* x)

end

"""

    Functions defining different potential energies
    according to the prior

"""

function Pot(P::GaussianPrior,x::Float64)
    u_pot = 0.5*(x-P.μ)*(x-P.μ)*P.β

    return u_pot
end

function Pot(P::ReLUPrior,x::Float64)

    f(y) = y>=0 ? 0.5*P.γ*y*y - y*P.θ : typemax(y)

    u_pot = f(x)

    return u_pot
end

function Pot(P::dReLUPrior,x)

    if x > 0
        u_pot = 0.5*P.γ_p*x*x - x*P.θ_p
    else
        u_pot = 0.5*P.γ_m*x*x - x*P.θ_m
    end

    return u_pot
end

function Pot(P::BinaryPrior,x)

    θ_B = log(P.ρ/(1-P.ρ))

    f(y) = y==0 || y==1 ? y*θ_B : typemax(y)
    
    u_pot = f(x)

    return u_pot

end

function Pot(P::SpikeSlabPrior,x)

    if x == 0.0
        u_pot = -log(1.0 - P.ρ)
    else
        u_pot = -log(P.ρ) + 0.5*P.λ*x*x + 0.5*log(2 * π * (1.0/P.λ)) 
    end

    return u_pot

end

function bernoulli!(x::Array{T,1}) where{T <: Real}

    for i in eachindex(x)
        @inbounds x[i] = float(rand() < x[i])
     end
     x

end

function sigm!(x::Array{T,1}) where{T <: Real}
    
    for i in eachindex(x)
       @inbounds x[i] = 1.0/(1.0 + exp(-x[i]))
    end
    x
 end


"""

    JackKnife: function computing average and uncertainty via JK method

"""

function JackKnife(data::Array{Float64,1})

    L = length(data)
    d_hat = zeros(Float64,L)
    m_hat = 0.0
    s_hat = 0.0

    for i=1:L
        for k in setdiff(1:L,i)
            d_hat[i] += data[k]
        end
        d_hat[i] /= (L-1)
        m_hat += d_hat[i]/L
    end

    for i=1:L
        s_hat += (d_hat[i]-m_hat)*(d_hat[i]-m_hat)
    end

    s_hat = sqrt((L-1)/L*s_hat)

    return m_hat, s_hat

end

"""

    Autocorrelation: compute the autocorrelation time to
    estimate error or to pick independent measurements

"""

function Autocorrelation(x::Array{Float64,1}, C::Vector{Float64};
                         est_err::Bool=false)

    m = 0.0
    sig = 0.0
    tau = 0.0
    k = 0
    Sl = 0.0

    N = length(x)
    C = fill!(C, 1.0)


    for i=1:N
        m += x[i]
    end

    m /= N


    if est_err
        while k < min(N-1, 5.0*(0.5+Sl/C[1]))
            for i=1:N-k
                C[k+1] += (x[i+k] - m)*(x[i] - m);
		    end
            C[k+1] /= (N-k)
            if k!=0
                Sl += C[k+1]
            end
            #writedlm(io,hcat(k,C[k+1],Sl))
            k+=1
        end
        #close(io)
        tau = 0.5 + Sl/C[1]
        sig = sqrt(C[1]*2*tau/N)
    end


    return m, sig, tau

end