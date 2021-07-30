
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

    #io = open("/Users/luca/Desktop/MC/MC_block.txt","w")
    #write(io,"#t\tE\n")
    #close(io)

    #io = open("/Users/luca/Desktop/MC/MC_block.txt","a")

    E = zeros(N_iter,)
    for iter=1:N_iter
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
            E[iter] += Pot(P[k],x[k])
            if k <= N
                for μ=1:M
                    E[iter] += w[k,μ]*x[k]*x[N+μ]
                end
            end
        end

        #writedlm(io,hcat(iter,E))

    end

    # for k=1:block_num+1
    #     bl_tab[k,3] -= bl_tab[k,2]^2
    #     bl_tab[k,3] = sqrt(bl_tab[k,3]/bl_size[k])
    # end

    #return bl_tab

    #close(io)

    return E

end

"""

    Block analysis function: given a MC path (whose length is a power of 2)
    divides data in block of size 2^k and for each one computes the average
    energy and uncertainty via JK.

"""

function block_anal(block_num::Int64, e_data::Vector)

    #e_data = readdlm(file_name; comments = true)

    @assert length(e_data) == 2^block_num

    N_iter = length(e_data)

    bl_tab = zeros(Float64,block_num,3)

    bl_lim = zeros(Int64,block_num+1)
    bl_lim[2:end] .= [N_iter÷(2^(block_num-b)) for b in 1:1:block_num]
    bl_size = bl_lim[2:end].-bl_lim[1:end-1]

    bl_tab[:,1] .= bl_size

    for b=1:block_num
        data_block = zeros(Float64,bl_size[b])
        idx_bl = bl_lim[b]+1:bl_lim[b+1]
        data_block .= e_data[idx_bl]

        av, sig = mean(data_block), std(data_block)/sqrt(length(idx_bl))
        bl_tab[b,2:3] .= [av, sig]
        #bl_tab[b,4:5] .= JackKnife(data_block)
    end

    return bl_tab

end
