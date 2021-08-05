
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

mutable struct MCout{T<:AbstractFloat}
    samples::Matrix{T}
    samples2::Matrix{T}
    vh::Array{T,3}
    E::Vector{T}
end

function MC_sim(w::Matrix{R},P::Array{T,1}, t_wait::Int64, Δ::R;
                        N::Int64=size(w,1),
                        M::Int64=size(w,2),
                        N_iter=100000,
                        x_init::Array{Float64,1}=zeros(Float64,N+M)) where {T <: Prior, R <: Real}

    x = copy(x_init)
    x2 = zeros(Float64,N+M)
    vh = zeros(Float64,N*M)
    a = 0.0

    #io = open("/Users/luca/Desktop/MC/MC_path.txt","w")
    #write(io,"#mc_step\ta\tv\th\tv2\th2\tvh\tE\n")
    #close(io)

    #io = open("/Users/luca/Desktop/MC/MC_path.txt","a")

    iter = 1
    while iter < t_wait
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
        iter = iter + 1
    end

    E = zeros(N_iter,)
    a_cum = zeros(N_iter,)
    samples = zeros(N_iter, N+M)
    samples2 = zeros(N_iter, N+M)
    vh = zeros(N_iter, N , M)

    for iter=1:N_iter

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

            a_cum[iter] += a
        
        end

        a_cum[iter] /= (N+M)
        samples[iter,:] .= x
        for k=1:N+M
            samples2[iter,k] = x[k]*x[k]
            E[iter] += Pot(P[k],x[k])
            if k <= N
                for μ=1:M
                    E[iter] += w[k,μ]*x[k]*x[N+μ]
                    vh[iter, k, μ ] = x[k]*x[N+μ]
                end
            end
        end
        #writedlm(io,hcat(Int64(iter-t_wait),a_cum,x',x2',vh',E))
        
    end

    #close(io)


    return MCout(samples, samples2, vh, E)

end

function compute_statistics(mc_out::MCout)

    
    N = size(mc_out.vh, 2)
    M = size(mc_out.vh, 3)
    av_mc = zeros(N+M,)
    va_mc = zeros(N+M,)
    cov_mc = zeros(N,M)
    for i = 1:N
        va_mc[i], _, _ = Sampling.Autocorrelation(mc_out.samples2[:,i])
        av_mc[i], _, _ = Sampling.Autocorrelation(mc_out.samples[:,i])
        for j = 1:M
            cov_mc[i,j], _, _ = Sampling.Autocorrelation(mc_out.vh[:,i,j])
        end

    end
    for i = 1:M
        av_mc[N+i], _, _ = Sampling.Autocorrelation(mc_out.samples[:,N+i])
        va_mc[N+i], _, _ = Sampling.Autocorrelation(mc_out.samples2[:,N+i])
    end

    for i = 1:N+M
        va_mc[i] = va_mc[i] - av_mc[i]^2
    end


    return av_mc, va_mc, cov_mc

end