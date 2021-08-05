

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
        ΔE += w[k]*y[k]
    end

    ΔE *= (x-x_old)
    ΔE += Pot(P,x)-Pot(P,x_old)

    return ΔE

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

function Autocorrelation(x::Array{Float64,1}; 
                         est_err::Bool=false)

    m = 0.0
    sig = 0.0
    tau = 0.0
    k = 0
    Sl = 0.0

    N = length(x)
    C = fill(1.0,N)

    #io = open("/Users/luca/Desktop/MC/Autocorr_$var.txt","w")
    #write(io,"#k\tC\tSl\n")
    #close(io)

    #io = open("/Users/luca/Desktop/MC/Autocorr_$var.txt","a")

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