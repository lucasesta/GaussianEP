# In this file we define several MC strategies for estimating
# the moments of a RBM visibile and hidden variables, depending
# on the specific choice of the prior.

"""

    E(v,h) = ∑_iμ v_i * w_iμ *h_μ + ∑_μ U_μ(h_μ) + ∑_i V_i(v_i)
    P(v,h) ∝ exp[-E(v,h)]

"""

#Function performing a scan with respect to the displacement parameter Δ

function MC_Δ_scan(Δ_vec::Array{R,1}, w::Matrix{R},P::Array{T,1}, N_iter::Int64;
                        N::Int64=size(w,1),
                        M::Int64=size(w,2),
                        x_init::Array{R,1}=zeros(Float64,N+M)) where {T<:Prior, R <: Real}

    acc_tab = zeros(Float64,length(Δ_vec),2)
    acc_tab[:,1] .= Δ_vec
    x = copy(x_init)
    a = 0.0
    counter = 0

    for Δ in Δ_vec
        counter += 1
        for iter=1:N_iter

            a_cum = 0.0
    
            for k=1:N+M
                l = rand(1:N+M)
                x_old = x[l]

                if typeof(P[l]) == BinaryPrior{R}
                    x[l] = MoveBin(x_old)
                else
                    x[l] = Move(x_old,Δ)
                end

                #x[l] = Move(x_old,Δ)
                if l <= N
                    x[l], a = Accept(x[N+1:N+M],w[l,:],P[l],x[l],x_old)
                else
                    x[l], a =Accept(x[1:N],w[:,l-N],P[l],x[l],x_old)
                end
    
                a_cum += a
            
            end
    
            a_cum /= (N+M)
            acc_tab[counter,2] += a_cum
        end

        acc_tab[counter,2] /= N_iter

    end


    return acc_tab

end

#Function computing thermalization time via block analysis

function MC_t_therm(block_num::Int64, w::Matrix{R},P::Array{T,1}, Δ::R;
                        N::Int64=size(w,1),
                        M::Int64=size(w,2),
                        x_init::Array{R,1}=zeros(Float64,N+M)) where {T<:Prior, R <: Real}

    N_iter = 2^block_num
    x = copy(x_init)

    io = open("/Users/luca/Desktop/MC/MC_block.txt","w")
    write(io,"#t\tE\n")
    close(io)

    io = open("/Users/luca/Desktop/MC/MC_block.txt","a")

    for iter=1:N_iter

        E = 0.0

        for k=1:N+M
            l = rand(1:N+M)
            x_old = x[l]

            if typeof(P[l]) == BinaryPrior{R}
                x[l] = MoveBin(x_old)
            else
                x[l] = Move(x_old,Δ)
            end

            #x[l] = Move(x_old,Δ)
            if l <= N
                x[l], a = Accept(x[N+1:N+M],w[l,:],P[l],x[l],x_old)
            else
                x[l], a =Accept(x[1:N],w[:,l-N],P[l],x[l],x_old)
            end
        end

        for k=1:N+M
            E += Pot(P[k],x[k])
            if k <= N
                for μ=1:M
                    E += w[k,μ]*x[k]*x[N+μ]
                end
            end
        end

        writedlm(io,hcat(iter,E))

    end

    # for k=1:block_num+1
    #     bl_tab[k,3] -= bl_tab[k,2]^2
    #     bl_tab[k,3] = sqrt(bl_tab[k,3]/bl_size[k])
    # end

    #return bl_tab

    close(io)

end

#Function performing MC simulation: print on a file the state x=(v,h) for every
#MC step and computes the average values ⟨x_i⟩, ⟨x_i^2⟩, ⟨v_i*h_μ⟩

function MC_sim_av(w::Matrix{Float64},P::Array{T,1}, N_iter::Int64, Δ::Float64;
                        N::Int64=size(w,1),
                        M::Int64=size(w,2),
                        t_wait::Int64=10000,
                        i_print::Int64=500,
                        x_init::Array{Float64,1}=zeros(Float64,N+M)) where {T <: Prior}

    x = copy(x_init)
    av_x = zeros(Float64,N+M)
    av_x2 = zeros(Float64,N+M)
    cor_vh = zeros(Float64,N*M)
    a = 0.0

    io = open("/Users/luca/Desktop/MC/MC_path.txt","w")
    write(io,"#mc_step\ta\tv\th\tE\n")
    close(io)

    io = open("/Users/luca/Desktop/MC/MC_path.txt","a")

    for iter=1:N_iter

        E = 0.0
        a_cum = 0.0

        for k=1:N+M
            l = rand(1:N+M)
            x_old = x[l]
            x[l] = Move(x_old,Δ)
            if l <= N
                x[l], a = Accept(x[N+1:N+M],w[l,:],P[l],x[l],x_old)
            else
                x[l], a =Accept(x[1:N],w[:,l-N],P[l],x[l],x_old)
            end

            a_cum += a
        
        end

        a_cum /= (N+M)

        for k=1:N+M
            E += Pot(P[k],x[k])
            if k <= N
                for μ=1:M
                    E += w[k,μ]*x[k]*x[N+μ]
                end
            end
        end

        if  iter > t_wait
            for k=1:N+M
                av_x[k] += x[k]
                av_x2[k] += x[k]*x[k]

                if k <= N
                    for m=1:M
                        cor_vh[ (k-1)*M + m ] += x[k]*x[N+m]
                    end
                end
    
            end
        end

        if mod(iter,i_print) == 0 || iter == 1
            writedlm(io,hcat(Int64(iter),a_cum,x',E))
        end
        

    end

    close(io)

    av_x /= (N_iter-t_wait)
    av_x2 /= (N_iter-t_wait)
    cor_vh /= (N_iter-t_wait)


    return (av_x = av_x, av_x2 = av_x2, cor_vh = cor_vh)

end

#Function performing a MC simulation: print on file the state x=(v,h)
#for every MC step after a thermalization time t_wait

function MC_sim(w::Matrix{R},P::Array{T,1}, t_wait::Int64, Δ::R;
                        N::Int64=size(w,1),
                        M::Int64=size(w,2),
                        N_iter=100000,
                        x_init::Array{Float64,1}=zeros(Float64,N+M)) where {T <: Prior, R <: Real}

    x = copy(x_init)
    x2 = zeros(Float64,N+M)
    vh = zeros(Float64,N*M)
    a = 0.0

    io = open("/Users/luca/Desktop/MC/MC_path.txt","w")
    write(io,"#mc_step\ta\tv\th\tv2\th2\tvh\tE\n")
    close(io)

    io = open("/Users/luca/Desktop/MC/MC_path.txt","a")

    for iter=1:N_iter+t_wait

        E = 0.0
        a_cum = 0.0

        for k=1:N+M
            l = rand(1:N+M)
            x_old = x[l]

            if typeof(P[l]) == BinaryPrior{R}
                x[l] = MoveBin(x_old)
            else
                x[l] = Move(x_old,Δ)
            end

            if l <= N
                x[l], a = Accept(x[N+1:N+M],w[l,:],P[l],x[l],x_old)
            else
                x[l], a =Accept(x[1:N],w[:,l-N],P[l],x[l],x_old)
            end

            if k>N && P[k]==ReLUPrior{R} && x[k]<0
                println("Not accessible region")
            end

            a_cum += a
        
        end

        a_cum /= (N+M)

        if  iter > t_wait
            for k=1:N+M
                x2[k] = x[k]*x[k]
                E += Pot(P[k],x[k])
                if k <= N
                    for μ=1:M
                        E += w[k,μ]*x[k]*x[N+μ]
                        vh[ (k-1)*M + μ ] = x[k]*x[N+μ]
                    end
                end
            end
            writedlm(io,hcat(Int64(iter-t_wait),a_cum,x',x2',vh',E))
        end
        

    end

    close(io)


    return 

end

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

    f(y) = y==0 || y==1 ? x*θ_B : typemax(y)
    
    u_pot = f(x)

    return u_pot

end

"""

    Block analysis function: given a MC path (whose length is a power of 2)
    divides data in block of size 2^k and for each one computes the average
    energy and uncertainty via JK.

"""

function block_anal(block_num::Int64,file_name::String)

    e_data = readdlm(file_name; comments = true)

    @assert size(e_data,1) == 2^block_num

    N_iter = size(e_data,1)

    bl_tab = zeros(Float64,block_num,3)

    bl_lim = zeros(Int64,block_num+1)
    bl_lim[2:end] .= [N_iter÷(2^(block_num-b)) for b in 1:1:block_num]
    bl_size = bl_lim[2:end].-bl_lim[1:end-1]

    bl_tab[:,1] .= bl_size

    for b=1:block_num
        data_block = zeros(Float64,bl_size[b])
        idx_bl = bl_lim[b]+1:bl_lim[b+1]
        data_block .= e_data[idx_bl,2]
        av, sig = JackKnife(data_block)
        bl_tab[b,2:3] .= [av, sig]
    end

    return bl_tab

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

function Autocorrelation(data::Array{Float64,1},var::String)

    m = 0.0
    sig = 0.0
    tau = 0.0
    k = 0
    Sl = 0.0

    x=copy(data)

    N=length(data)
    C = fill(1.0,N)

    io = open("/Users/luca/Desktop/MC/Autocorr_$var.txt","w")
    write(io,"#k\tC\tSl\n")
    close(io)

    io = open("/Users/luca/Desktop/MC/Autocorr_$var.txt","a")

    for i=1:N
        m += x[i]
    end

    m /= N

    while k < 5.0*(0.5+Sl/C[1])
        for i=1:N-k
			C[k+1] += (x[i+k] - m)*(x[i] - m);
		end
        C[k+1] /= (N-k)
        if k!=0
            Sl += C[k+1]
        end

        writedlm(io,hcat(k,C[k+1],Sl))

        k+=1

    end

    close(io)

    tau = 0.5 + Sl/C[1]

    sig = sqrt(C[1]*2*tau/N)

    return m, sig, tau

end