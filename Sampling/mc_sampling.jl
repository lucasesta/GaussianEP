
#Function performing a MC simulation: print on file the state x=(v,h)
#for every MC step after a thermalization time t_wait
using BenchmarkTools
mutable struct MCout{T<:AbstractFloat}
    samples::Matrix{T}
    vh::Array{T,3}
    E::Vector{T}
end

function MC_sim(w::Matrix{R},P::Array{T,1}, t_wait::Int64, Δ::R;
                        N::Int64=size(w,1),
                        M::Int64=size(w,2),
                        N_iter=100000,
                        x_init::Array{Float64,1}=zeros(Float64,N+M)) where {T <: Prior, R <: Real}

    x = copy(x_init)
    a = 0.0

    iter = 1
    while iter < t_wait
        a_cum = 0.0
        for k=1:N+M
            l = rand(1:N+M)
            x_old = x[l]
            if typeof(P[l]) == BinaryPrior{R}
                x[l] = MoveBin(x_old)
            elseif typeof(P[l]) == SpikeSlabPrior{R}
                x[l] = MoveSpAndSlab(x_old, Δ, P[l].ρ)
            else
                x[l] = Move(x_old,Δ)
            end

            if l <= N
                x[l], a = Accept(x[N+1:N+M],w[l,:],P[l],x[l],x_old)
            else
                x[l], a = Accept(x[1:N],w[:,l-N],P[l],x[l],x_old)
            end

            if k>N && P[k]==ReLUPrior{R} && x[k]<0
                println("Not accessible region")
            end

            #a_cum += a
        
        end
        #a_cum /= (N+M)
        iter = iter + 1
    end

    #E = zeros(N_iter,)
    #a_cum = zeros(N_iter,)
    samples = zeros(N_iter, N+M)
    #vh = zeros(N_iter, N , M)

    for iter=1:N_iter

        for k=1:N+M
            l = rand(1:N+M)
            x_old = x[l]

            if typeof(P[l]) == BinaryPrior{R}
                x[l] = MoveBin(x_old)
            elseif typeof(P[l]) == SpikeSlabPrior{R}
                x[l] = MoveSpAndSlab(x_old, Δ, P[l].ρ)
            else
                x[l] = Move(x_old,Δ)
            end

            if l <= N
                x[l], a = Accept(x[N+1:N+M], w[l,:],P[l],x[l],x_old)
            else
                x[l], a =Accept(x[1:N], w[:,l-N],P[l],x[l],x_old)
            end

            if k>N && P[k]==ReLUPrior{R} && x[k]<0
                println("Not accessible region")
            end

            #a_cum[iter] += a
        
        end

        #a_cum[iter] /= (N+M)
        samples[iter,:] .= x
        #en_prior = 0.0
        #en_w = 0.0
        #for k=1:N+M
            #samples2[iter,k] = x[k]*x[k]
            #E[iter] += Pot(P[k],x[k])
            #en_prior += Pot(P[k], x[k])
            #if k <= N
                #for μ=1:M
                    #en_w -= w[k,μ] * x[k] * x[N+μ]
                    #E[iter] -= w[k,μ]*x[k]*x[N+μ]
                    #vh[iter, k, μ ] = x[k]*x[N+μ]
                #end
            #end
        #end
        
    end

    return samples

end

function compute_statistics(samples::Matrix{Float64}, N::Int, M::Int)

    
    N_iter = size(samples, 1)
    av_mc = zeros(N+M,)
    va_mc = zeros(N+M,)
    cov_mc = zeros(N,M)
    C = ones(N,)

    samples2 = zeros(N_iter,)
    vh = zeros(N_iter, )

    for i = 1:N
        samples2 .= samples[:,i] .^2
        va_mc[i], _, _ = Sampling.Autocorrelation(samples2, C)
        av_mc[i], _, _ = Sampling.Autocorrelation(samples[:,i],C)
        for j = 1:M
            vh .= samples[:,i] .* samples[:,N+j]
            cov_mc[i,j], _, _ = Sampling.Autocorrelation(vh, C)
        end

    end
    for i = 1:M
        samples2 .= samples[:,N+i] .^2
        av_mc[N+i], _, _ = Sampling.Autocorrelation(samples[:,N+i], C)
        va_mc[N+i], _, _ = Sampling.Autocorrelation(samples2, C)
    end

    for i = 1:N+M
        va_mc[i] = va_mc[i] - av_mc[i]^2
    end


    return av_mc, va_mc, cov_mc

end