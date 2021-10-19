using FastGaussQuadrature, ForwardDiff

Φ(x) = 0.5*(1+erf(x/sqrt(2.0)))
ϕ(x) = exp(-x.^2/2)/sqrt(2π)

"""
Abstract Univariate Prior type
"""
abstract type Prior end

"""
    moments(p0::T, μ, σ) where T <:Prior -> (mean, variance)

    input: ``p_0, μ, σ``

    output: mean and variance of

    `` p(x) ∝ p_0(x) \\mathcal{N}(x;μ,σ) ``
"""
function moments(p0::T, μ, σ) where T <: Prior
    error("undefined moment calculation, assuming uniform prior")
    return μ,σ^2
end

"""

    Uniform prior

"""

struct UniformPrior <: Prior
    
end

function moments(p0::UniformPrior, μ, σ2)
    return μ,σ2
end

"""

    gradient(p0::T, μ, σ) -> nothing

    update parameters with a single learning gradient step (learning rate is stored in p0)
"""
function gradient(p0::T, μ, σ) where T <: Prior
    #by default, do nothing
    return
end

function gradient!(p0::Vector{P}, upd_grad::Symbol) where P <: Prior
    #by default, do nothing
    #println("do nothing")
    return 0.0
end

"""
Interval prior

Parameters: l,u

`` p_0(x) = \\frac{1}{u-l}\\mathbb{I}[l\\leq x\\leq u] ``
"""
struct IntervalPrior{T<:Real} <: Prior
    l::T
    u::T
end

function moments(p0::IntervalPrior,μ,σ)
    xl = (p0.l - μ)/σ
    xu = (p0.u - μ)/σ
    minval = min(abs(xl), abs(xu))

    if xu - xl < 1e-10
        return 0.5 * (xu + xl), -1
    end

    if minval <= 6.0 || xl * xu <= 0
        ϕu, Φu, ϕl, Φl  = ϕ(xu), Φ(xu), ϕ(xl), Φ(xl)
        av = (ϕl - ϕu) / (Φu - Φl)
        mom2 = (xl * ϕl - xu * ϕu) / (Φu - Φl)
    else
        Δ = (xu^2 - xl^2) * 0.5
        if Δ > 40.0
            av = xl^5 / (3.0 - xl^2 + xl^4)
            mom2 = xl^6 / (3.0 - xl^2 + xl^4)
        else
            eΔ = exp(Δ)
            av = (xl * xu)^5 * (1. - eΔ) / (-eΔ * (3.0 - xl^2 + xl^4) * xu^5 + xl^5 * (3.0 - xu^2 + xu^4))
            mom2 = (xl * xu)^5 * (xu - xl * eΔ) / (-eΔ * (3.0 - xl^2 + xl^4) * xu^5 + xl^5 * (3.0 - xu^2 + xu^4))
        end
    end
    va = mom2 - av^2
    return μ + av * σ, σ^2 * (1 + va)
end


"""
Spike-and-slab prior

Parameters: ρ,λ

`` p_0(x) ∝ (1-ρ) δ(x) + ρ \\mathcal{N}(x;0,λ^{-1}) ``
"""
mutable struct SpikeSlabPrior{T<:Real} <: Prior
    ρ::T
    λ::T
    δρ::T
    δλ::T
    ∂ρ::T
    ∂λ::T
end

SpikeSlabPrior(ρ,λ) = SpikeSlabPrior(ρ,λ,0.0,0.0,0.0,0.0);
SpikeSlabPrior(ρ,λ,δρ,δλ) = SpikeSlabPrior(ρ,λ,δρ,δλ,0.0,0.0);

"""
``p = \\frac1{(ℓ+1)((1/ρ-1) e^{-\\frac12 (μ/σ)^2 (2-\\frac1{1+ℓ})}\\sqrt{1+\\frac1{ℓ}}+1)}``
"""
function moments(p0::SpikeSlabPrior,μ,σ2)
#=
    s2 = σ^2
    d = 1 + p0.λ * s2;
    sd = 1 / (1/s2 + p0.λ);
    n = μ^2/(2*d*s2);
    Z = sqrt(p0.λ * sd) * p0.ρ;
    f = 1 + (1-p0.ρ) * exp(-n) / Z;
    av = μ / (d * f);
    va = (sd + (μ / d)^2 ) / f - av^2;
    #p0 = (1 - p0.params.ρ) * exp(-n) / (Z + (1-p0.params.ρ).*exp(-n));
    =#
    ℓ0 = p0.λ * σ2
    ℓ = 1 + ℓ0;
    z = ℓ * (1 + (1/p0.ρ-1) * exp(-0.5*(μ)^2/σ2/ℓ) * sqrt(ℓ/ℓ0))
    av = μ / z;
    va = (σ2 + μ^2*(1/ℓ - 1/z)) / z;
    return av, va
end


function gradient(p0::SpikeSlabPrior, μ, σ2)
    s = σ2
    d = 1 + p0.λ * s;
    q = sqrt(p0.λ * s / d);
    f = exp(-μ^2 / (2s*d));
    den = (1 - p0.ρ) * f + q * p0.ρ;
    # update ρ
    if p0.δρ > 0
        #p0.ρ += p0.δρ * (q - f) / den;
        #p0.ρ = clamp(p0.ρ, 0, 1)
        p0.∂ρ = (q-f) / den;
    end
    # update λ
    if p0.δλ > 0
        num = s * (1 + p0.λ * (s - μ^2)) / (2q * d^3) * p0.ρ;
        p0.∂λ = num/den
        #p0.λ += p0.δλ * num/den;
        #p0.λ = max(p0.λ, 0)
    end
end

function gradient!(p0::Array{SpikeSlabPrior{T},1}, upd_grad::Symbol) where {T <: Real}

    N = length(p0)    
    Δc = 0.0
    if upd_grad == :unique
        Δρ = 0.0
        Δλ = 0.0
        for i in 1:N
            Δρ += p0[i].∂ρ 
            Δλ += p0[i].∂λ 
        end
        ρ_old = p0[1].ρ
        λ_old = p0[1].λ
        for i in 1:N
            p0[i].ρ += p0[i].δρ * Δρ
            p0[i].λ += p0[i].δλ * Δλ
            p0[i].ρ = clamp(p0[i].ρ, 0.0, 1.0)
            p0[i].λ = max(p0[i].λ, 0.0)
        end
        Δc = max(abs(ρ_old - p0[1].ρ), abs(λ_old - p0[1].λ))
    else
        ρ_old = mean(p0[:].ρ)
        λ_old = mean(p0[:].λ)
        for i in 1:N
            p0[i].ρ += p0[i].δρ * p0[i].∂ρ
            p0[i].λ += p0[i].δλ * p0[i].∂λ
            p0[i].ρ = clamp(p0[i].ρ, 0.0, 1.0)
            p0[i].λ = max(p0[i].λ, 0.0)
        end
        Δc = max(abs(ρ_old - mean(p0[:].ρ)), abs(λ_old - mean(p0[:].λ)))
    end

    return Δc

end

"""
Binary Prior

p_0(x) ∝ ρ δ(x-x_0) + (1-ρ) δ(x-x_1)

"""
mutable struct BinaryPrior{T<:Real} <: Prior
    x0::T
    x1::T
    ρ::T
    δρ::T
    ∂ρ::T
end

BinaryPrior(x0, x1, ρ) = BinaryPrior(x0, x1, ρ, 0.0, 0.0)
BinaryPrior(x0, x1, ρ, δρ) = BinaryPrior(x0, x1, ρ, δρ, 0.0)

function moments(p0::BinaryPrior, μ, σ2)
    arg = -1/(2*σ2) * (-p0.x0^2 - 2*(p0.x1 -p0.x0) * μ + p0.x1^2);
    earg = exp(arg)
    Z = p0.ρ / earg + (1-p0.ρ);
    av = p0.ρ * p0.x0 / earg + (1-p0.ρ) * p0.x1;
    mom2 = p0.ρ * (p0.x0^2) / earg + (1-p0.ρ) * (p0.x1^2);
    if (isnan(Z) || isinf(Z))
        Z = p0.ρ + (1-p0.ρ) * earg;
        av = p0.ρ * p0.x0 + (1-p0.ρ) * p0.x1 * earg;
        mom2 = p0.ρ * (p0.x0^2) + (1-p0.ρ) * p0.x1 * earg;
    end
    av /= Z;
    mom2 /= Z;
    va = mom2 - av.^2;
    return av,va
end


function gradient(p0::BinaryPrior, μ, σ2)

    arg = -1/(2*σ2) * ( p0.x0^2 - p0.x1^2 + 2 * μ * (p0.x1 - p0.x0) )
    earg = exp(arg)
    Z = p0.ρ * earg + (1-p0.ρ)

    if (isnan(Z) || isinf(Z))
        println("Infinite Z\n")
        Z = p0.ρ + (1-p0.ρ) / earg;
        if p0.δρ > 0
            #p0.ρ += p0.δρ * (1 - 1 / earg) / Z;
            #p0.ρ = clamp(p0.ρ, 0, 1)
            p0.∂ρ = (1.0 - 1.0/earg)  / Z;
        end
    elseif p0.δρ > 0
        #p0.ρ += p0.δρ * (earg - 1) / Z;
        #p0.ρ = clamp(p0.ρ, 0, 1)
        p0.∂ρ = (earg - 1.0) / Z;
    end

end

function gradient!(p0::Array{BinaryPrior{T},1}, upd_grad::Symbol)  where {T <: Real}

    N = length(p0)
    Δc = 0.0
    if upd_grad == :unique
        Δρ = 0.0
        Δc = 0.0
        for i in 1:N
            Δρ += p0[i].∂ρ 
        end
        #Δρ /= N
        ρ_old = p0[1].ρ
        for i in 1:N
            p0[i].ρ += p0[i].δρ * Δρ
            p0[i].ρ = clamp(p0[i].ρ, 0.0, 1.0)
        end
        Δc = abs(ρ_old - p0[1].ρ)
    else
        ρ_old = mean(p0[:].ρ)
        for i in 1:N
            p0[i].ρ += p0[i].δρ * p0[i].∂ρ
            p0[i].ρ = clamp(p0[i].ρ, 0.0, 1.0)
        end
        Δc = abs(ρ_old - mean(p0[:].ρ))
    end

    return Δc

end

#Gaussian prior

mutable struct GaussianPrior{T<:Real} <: Prior
    μ::T
    β::T
    δβ::T
end

function moments(p0::GaussianPrior, μ, σ2)
    s = 1/(1/σ2 + p0.β)
    return s*(μ/σ2 + p0.μ * p0.β), s
end

"""
This is a fake Prior that can be used to fix experimental moments
Parameters: μ, v (variance, not std)
"""
struct Posterior2Prior{T<:Real} <: Prior
    μ::T
    v::T
end

function moments(p0::Posterior2Prior, μ, σ)
    return p0.μ,p0.v
end

struct Posterior1Prior{T<:Real} <: Prior
    μ::T
end

function moments(p0::Posterior1Prior, μ, σ2)
    return p0.μ,σ2
end


struct QuadraturePrior{T<:Real} <: Prior
    f
    X::Vector{T}
    W0::Vector{T}
    W1::Vector{T}
    W2::Vector{T}
    function (QuadraturePrior)(f; a::T=-1.0, b::T=1.0, points::Int64=1000) where {T<:Real}
        X,W = gausslegendre(points)
        X = 0.5*(X*(b-a).+(b+a))
        W .*= 0.5*(b-a)
        W0 = map(f,X) .* W;
        W1 = X .* W0;
        W2 = X .* W1;
        new{T}(f,X,W0,W1,W2)
    end
end

function moments(p0::QuadraturePrior, μ, σ)
    v = map(x->exp(-(x-μ)^2/2σ^2),p0.X)
    z0 = v ⋅ p0.W0
    av = (v ⋅ p0.W1)/z0
    va = (v ⋅ p0.W2)/z0 - av^2
    return av, va
end

mutable struct AutoPrior{T<:Real} <: Prior
    #real arguments
    f
    P::Vector{T}
    dP::Vector{T}
    #quadrature data
    X::Vector{T}
    W::Vector{T}
    #cached data (updated in ctor and update!)
    X2::Vector{T}
    oldP::Vector{T}
    FXW::Vector{T}
    DFXW::Matrix{T}
    cfg::Any
    function (AutoPrior)(f, P::Vector{T}, dP::Vector{T} = zeros(length(P)), a=-1, b=1, points=1000) where {T<:Real}
        X,W = gausslegendre(points)
        X = 0.5*(X*(b-a).+(b+a))
        W .*= 0.5*(b-a)
        np = length(P)
        new{T}(f,P,dP,X,W,X.^2,fill(Inf,np),fill(zero(T),points),fill(zero(T),np,points),ForwardDiff.GradientConfig(f,[X[1];P]))
    end
end

function moments(p0::AutoPrior, μ, σ2)
    do_update!(p0)
    s22 = 2σ2
    v = p0.FXW .* map(x->exp(-(x-μ)^2 / s22), p0.X)
    v .*= 1/sum(v)
    av = v ⋅ p0.X
    va = (v ⋅ p0.X2) - av^2
    return av, va
end

function do_update!(p0::AutoPrior)
    p0.P == p0.oldP && return
    copy!(p0.FXW, p0.f.([[x;p0.P] for x in p0.X]) .* p0.W)
    for i in 1:length(p0.X)
        p0.DFXW[:,i] = ForwardDiff.gradient(p0.f,[p0.X[i];p0.P],p0.cfg)[2:end]
    end
    copy!(p0.oldP, p0.P)
end

function gradient(p0::AutoPrior, μ, σ2)
    s22 = 2σ2
    v = map(x->exp(-(x-μ)^2 / s22), p0.X)
    z = sum(v)
    v ./= z
    p0.P .+= p0.dP .* (p0.DFXW * v)
end


###### Thomas's theta moments

#The quotient of normal PDF/CF with the proper expansion
#The last expansion term should not be necessary though.
function pdf_cf(x)
    if x < -8
        return -1/x-x+2x^(-3)-10x^(-5)
    end
    return sqrt(2/pi)*exp(-x^2/2)/(1+erf(x/sqrt(2)))
end

"""
A θ(x) prior
"""
struct ThetaPrior <: Prior end


function moments(::ThetaPrior,μ,σ)
    α=μ/σ
    av=μ+pdf_cf(α)*σ
    var=σ^2*(1-α*pdf_cf(α)-pdf_cf(α)^2)
    return av,var
end

"""
A mixture of theta priors: p_0(x)=η*Θ(x)+(1-η)*Θ(-x)
"""
mutable struct ThetaMixturePrior{T<:Real} <: Prior
    η::T
    δη::T
end

function theta_mixt_factor(x,η)
    f=exp(-0.5*x^2.0)/(η*erfc(-sqrt(0.5)*x)+(1.0-η)*erfc(sqrt(0.5)*x))
    return f
end

function moments(p0::ThetaMixturePrior,μ,σ)
    η=p0.η
    α=μ/σ
    f=theta_mixt_factor(α,η)
    χ=sqrt(2.0/π)*(2.0*η-1.0)*f
    av=μ+σ*χ
    va=σ^2.0*(1-χ^2.0)-μ*σ*χ
    return av,va
end

function gradient(p0::ThetaMixturePrior,μ,σ)
    η=p0.η
    x=μ/σ/sqrt(2)
    num=2*erf(x)
    den=η*erfc(-x)+(1-η)*erfc(x)
    p0.η+=p0.δη*num/den
    p0.η=clamp(p0.η,0,1)
end

"""

    ReLU prior: defines a potential for which
    the transfer function of the hidden layer
    variables are Rectified Linear Units.

    Defined by inverse variance γ and mean θ/γ


"""

mutable struct ReLUPrior{T<:Real} <: Prior
    γ::T
    θ::T
    δγ::T
    δθ::T
    ∂γ::T
    ∂θ::T
end

ReLUPrior(γ, θ) = ReLUPrior(γ, θ, 0.0, 0.0, 0.0, 0.0)
ReLUPrior(γ, θ, δγ, δθ) = ReLUPrior(γ, θ, δγ, δθ, 0.0, 0.0)

function moments(p0::ReLUPrior,μ,σ2)

    m = (μ+p0.θ*σ2)/(1+p0.γ*σ2)
    s = σ2/(1+p0.γ*σ2)

    if s <0
        throw(DomainError("Combined variance must be positive"))
    end

    α = m/sqrt(s)

    av = m + sqrt(s)*pdf_cf(α)
    va = s*(1-pdf_cf(α)^2-α*pdf_cf(α))

    return av, va

end

function gradient(p0::ReLUPrior,μ,σ2)

    m = (μ+p0.θ*σ2)/(1+p0.γ*σ2)
    s = σ2/(1+p0.γ*σ2)

    if s <0
        throw(DomainError("Combined variance must be positive"))
    end

    α_0 = p0.θ/sqrt(p0.γ)
    α = m/sqrt(s)

    g_γ = 0.5 * ( α_0 * pdf_cf(α_0)/p0.γ + 1/(p0.γ*(1+p0.γ*s)) - m * sqrt(s) * pdf_cf(α))
    g_θ = pdf_cf(α) * sqrt(s) - pdf_cf(α_0)/sqrt(p0.γ)

    #update
    if p0.δγ > 0
        #p0.γ += p0.δγ * g_γ
        p0.∂γ = g_γ
    end

    if p0.δθ > 0
        #p0.θ += p0.δθ * g_θ   
        p0.∂θ = g_θ 
    end

end


function gradient!(p0::Array{ReLUPrior{T},1}, upd_grad::Symbol) where {T <: Real}

    N = length(p0)
    Δc = 0.0
    if upd_grad == :unique
        Δγ = 0.0
        Δθ = 0.0
        for i in 1:N
            Δγ += p0[i].∂γ
            Δθ += p0[i].∂θ
        end
        γ_old = p0[1].γ
        θ_old = p0[1].θ
        for i in 1:N
            p0[i].γ += p0[i].δγ * Δγ
            p0[i].θ += p0[i].δθ * Δθ
        end
        Δc = max(abs(γ_old - p0[1].γ), abs(θ_old - p0[1].θ))
    else
        γ_old = mean(p0[:].γ)
        θ_old = mean(p0[:].θ)
        for i in 1:N
            p0[i].γ += p0[i].δγ * p0[i].∂γ
            p0[i].θ += p0[i].δθ * p0[i].∂θ
        end
        Δc = max(abs(γ_old - mean(p0[:].γ)), abs(θ_old - mean(p0[:].θ)))
    end

    return Δc

end


"""

    dReLU prior: defines a potential for which
    the transfer function of the hidden layer
    variables are  double Rectified Linear Units.

    Defined by inverse variances γ₊,γ₋ and means θ₊/γ₊,θ₋/γ₋


"""

struct dReLUPrior{T<:Real} <: Prior
    γ_p::T
    θ_P::T
    γ_m::T
    θ_m::T
end

function ψ(x::T) where {T<:Real}
    return 0.5*(1-erfc(x/sqrt(2.0)))
end

function pdf_cf2(x::T) where {T<:Real}
    return sqrt(2/π)*exp(-x^2/2)/(1 - erf(x/sqrt(2)))
end

function pdf(x::T) where {T<:Real}
    return exp(-x^2/2)/sqrt(2*π)
end

function moments(p0::dReLUPrior,μ,σ2; tol::T=1.0e-10) where {T <: Real}

    m_p = (μ+p0.θ_p*σ2)/(1+p0.γ_p*σ2)
    s_p = σ2/(1+p0.γ_p*σ2)
    m_m = (μ+p0.θ_m*σ2)/(1+p0.γ_m*σ2)
    s_m = σ2/(1+p0.γ_m*σ2)

    α_p = m_p/sqrt(s_p)
    α_m = m_m/sqrt(s_m)

    w1 = sqrt(1.0+p0.γ_m*σ2)*Φ(α_p)
    w2 = sqrt(1.0+p0.γ_p*σ2)*ψ(α_m)

    if abs(α_m) > tol
        av = (w1*m_p + w2*m_m)/(w1 + w2) + (sqrt(s_p)*w1*pdf_cf(α_p) + sqrt(s_m)*w2*pdf_cf2(α_m))/(w1 + w2)
        av2 = (w1*(m_p^2 + s_p) + w2*( m_m^2 + s_m))/(w1 + w2) + (m_p*sqrt(s_p)*w1*pdf_cf(α_p) + m_m*sqrt(s_m)*w2*pdf_cf2(α_m))/(w1 + w2)
    else
        av = (w1*m_p + w2*m_m)/(w1 + w2) + σ*(sqrt(s_p/s_m)*pdf(α_p) + sqrt(s_m/s_p)*pdf(α_m))/(w1 + w2)
        av2 = (w1*(m_p^2 + s_p) + w2*( m_m^2 + s_m))/(w1 + w2) + (m_p*sqrt(s_p)*w1*pdf_cf(α_p) + m_m*sqrt(s_m)*w2*pdf_cf2(α_m))/(w1 + w2)
    end

    va = av2 - av*av

    return av, va

end

