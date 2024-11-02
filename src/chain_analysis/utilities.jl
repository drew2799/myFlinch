#using Turing
using Distributions
using Plots
using LinearAlgebra
using Random
using StatsFuns
include("diffSHT_healpix.jl")

#   Manipulating a_{\ell m}
#   =======================
# 
#   The input list of a_{\ell m} of both E and B mode and for each tomographic
#   bin is organised as a vector of matrices. The vector index vary among the
#   \ell values, then in each matrix the columns correspond to every valid m (m
#   \in [0, \ell]) while the rows separate the a_{\ell m} belonging to different
#   modes and bins. We need a function that convert this object into an
#   healpix.Alm object, in which the a_{\ell m} are stored in a single vector
#   ordered by m.

function from_alm_to_healpix_alm(alm, l_max, nbin)
    Alms = []
    for i in 1:nbin
        alm_array = zeros(ComplexF64, numberOfAlms(l_max))
        for l in 1:l_max+1
            alm_array[l] = alm[l][i, 1]
        end
        j = l_max + 1
        for m in 2:2:(2*l_max + 1)
            for l in (Int(m/2) +1):(l_max+1)
                j += 1
                alm_array[j] = alm[l][i,m] + alm[l][i,m+1]*im
            end
        end
        push!(Alms, Alm(l_max, l_max, alm_array))
    end
    return Alms
end

#   Since fromalmtohealpixalm appears in the posterior that we want to sample,
#   we need a rule to differentiate this function. However, to do this we
#   firstly need the inverse function that maps an healpix.Alm to a vector of
#   matrices. The reason for this is that fromalmtohealpixalm does not change
#   the value of each coefficient, it just reorganizes them and hence, the rule
#   does not nedd to do anything in particular. But be careful, the adjoint of
#   the output is an healpix.Alm object while its pullback must be a vector of
#   matrices. This is way we need the inverse function to pull the adjoint back
#   to the input space.

function from_healpix_alm_to_alm(Alms, lmax, nbin)
    alm_array = []
    for l in 0:lmax
        #alm = Matrix{ComplexF64}(undef, (2*nbin, 2*(l + 1)))
        alm = Matrix{Float64}(undef, (nbin, 2*(l + 1)))
        for i in 1:nbin
            j = 1
            for m in each_m_idx(Alms[i], l)
                alm[i,j]=real(Alms[i].alm[m])
                alm[i,j+1]=imag(Alms[i].alm[m])
                j+=2
            end
        end
        push!(alm_array, alm[:, 1:end .!=2])
    end
    return alm_array
end

#   @adjoint function fromalmtohealpixalm(alm, lmax, nbin) y =
#   fromalmtohealpixalm(alm, lmax, nbin) function fathapullback(ȳ) return
#   (fromhealpixalmtoalm(ȳ, lmax, nbin), nothing, nothing) end return y,
#   fatha_pullback end

function ChainRulesCore.rrule(::typeof(from_alm_to_healpix_alm), alm, l_max, nbin)
    y = from_alm_to_healpix_alm(alm, l_max, nbin)
    function fatha_pullback(ȳ)
        x̄ = @thunk(from_healpix_alm_to_alm(ȳ, l_max, nbin))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end
    return y, fatha_pullback
end

#   Cholesky coordinates
#   ====================
# 
#   We cannot straightforwardly sample the elements of C since the HMC random
#   walk can easily wander out of the subset of positive-definite matrices. To
#   handle this we instead explored sampling the ‘diagonal-log’ K of the
#   Cholesky factor L of the covariance matrix. For the strong correlations
#   inherent in cosmic shear, the Cholesky decomposition was found to lead to
#   chains with shorter correlation lengths.
# 
# :$
# 
#   \mathrm{C}\ell = \mathrm{L\ell L\ell ^T}, :$ with L the Cholesky factor, a
#   lower-triangular matrix :$ \mathbf{a} = \mathrm{L}\mathbf{x}. :$ We also
#   need the "diagonal-log" K of the Cholesky factor :$ \mathrm{K}{\alpha\beta}
#   = \begin{cases} \ln{(\mathrm{L}{\alpha\beta})} & \mathrm{if} \ \alpha=\beta,
#   \ \mathrm{L}{\alpha\beta} & \mathrm{otherwise} \end{cases} :$
# 
#   We write a function that change from (C, \mathbf{a}) to (L, \mathbf{x})
#   coordinates.

function from_Cholesky(L::Cholesky{Float64, Matrix{Float64}}, x::Matrix{Float64})
    C = L.L*L.L'
    a = L.L*x
    return C, a
end

function global_from_Cholesky(L::Vector{Cholesky{Float64, Matrix{Float64}}}, x::Vector{Matrix{Float64}})
    C = Vector{Matrix{Float64}}(undef, length(L))
    a = Vector{Matrix{Float64}}(undef, length(L))
    for i in 1:length(L)
        C[i], a[i] = from_Cholesky(L[i], x[i])
    end
    return C, a
end

function Chol_Lx2a(x::Vector{Matrix{Float64}}, L::Vector{Matrix{Float64}})
    a = Vector{Matrix{Float64}}(undef, length(x))
    for i in 1:length(x)
        a[i] = L[i]*x[i]
        a[i][:,2:end] *= (1/sqrt(2))
    end
    return a
end

function ChainRulesCore.rrule(::typeof(Chol_Lx2a), x::Vector{Matrix{Float64}}, L::Vector{Matrix{Float64}})
    a = Chol_Lx2a(x, L)
    function Chol_Lx2a_pullback(ā)
        L̄ = Vector{Matrix{Float64}}(undef, length(L))
        x̄ = Vector{Matrix{Float64}}(undef, length(x))
        for i in 1:length(ā)
            ā[i][:,2:end] *= (1/sqrt(2))
            L̄[i] = ā[i]*transpose(x[i])
            x̄[i] = transpose(L[i])*ā[i]
        end
        return ChainRulesCore.NoTangent(), x̄, L̄
    end
    return a, Chol_Lx2a_pullback
end

#   @adjoint function CholLx2a(x::Vector{Matrix{Float64}},
#   L::Vector{Matrix{Float64}}) a = CholLx2a(x, L) function CholLx2apullback(ā)
#   L̄ = Vector{Matrix{Float64}}(undef, length(L)) x̄ =
#   Vector{Matrix{Float64}}(undef, length(x)) Threads.@threads for i in
#   1:length(ā) ā[i][:,2:end] = (1/sqrt(2)) L̄[i] = ā[i]transpose(x[i]) x̄[i] =
#   transpose(L[i])*ā[i] end return (x̄, L̄) end return a, CholLx2apullback end

#   From K_{\ell} to L_{\ell}
#   =========================

#   Mapping from the vector of free parameters of all the K matrices to a vector
#   of L matrices.

function inv_k_transf(k, nbin)
    l = zeros(length(k))
    s = nbin
    for i in 1:nbin-1
        l[i] = normcdf(k[i])
        for j in (i+1):nbin
            l[s] = 2*normcdf(k[s]) - 1
            s += 1
        end
    end
    return l
end

function single_from_k_to_L(k, nbin)
    L = Matrix{Float64}(I,(nbin,nbin))
    s = nbin+1
    for i in 1:nbin
        L[i,i] = exp(k[i])
        for j in (i+1):nbin
            L[j,i] = k[s]
            s+=1
        end
    end
    return L
end

function vector_from_k_to_L(all_k, nbin, lmax, freeparam_n)
    L = Vector{Matrix{Float64}}(undef, lmax+1)
    for i in 1:lmax+1
        L[i] = single_from_k_to_L(all_k[(freeparam_n*(i-1)+1):freeparam_n*i], nbin)
    end
    return L
end 

function vector_from_L_to_k(L, nbin, lmax)
    vec = []
    for l in 1:lmax+1
        vec = vcat(vec,log.(diag(L[l])))
        for i in 1:nbin-1
            for j in i+1:nbin
                push!(vec, L[l][j,i])
            end
        end
    end
    return vec
end

@adjoint function vector_from_k_to_L(all_k, nbin::Int64, lmax::Int64, freeparam_n::Int64)
    y = vector_from_k_to_L(all_k, nbin, lmax, freeparam_n)
    function vecK2L_pullback(ȳ)
        k̄ = []
        for l in 1:length(ȳ)
            for i in 1:dim(ȳ[1])
                push!(k̄, y[l][i,i]*ȳ[l][i,i])
            end
            #k̄ = vcat( k̄, diag(y[l])[2:end] .* diag(ȳ[l])[2:end] )
            for i in 1:dim(ȳ[1])-1
                for j in i+1:dim(ȳ[1])
                    push!(k̄, ȳ[l][j,i])
                end
            end
        end
        return (k̄, nothing, nothing, nothing)
    end
    return y, vecK2L_pullback
end

#   Transforming a vector of components into K matrices

function singleK_vec2vecmat(k, nbin)
    K = Matrix{Float64}(I,(nbin,nbin))
    s = nbin+1
    for i in 1:nbin
        K[i,i] = k[i]
        for j in (i+1):nbin
            K[j,i] = k[s]
            s+=1
        end
    end
    return K
end

function vectorK_vec2vecmat(all_k, nbin, lmax, freeparam_n)
    K = Vector{Matrix{Float64}}(undef, lmax+1)
    for i in 1:lmax+1
        K[i] = singleK_vec2vecmat(all_k[(freeparam_n*(i-1)+1):freeparam_n*i], nbin)
    end
    return K
end 

function vectorK_vecmat2vec(K, nbin, lmax)
    vec = []
    for l in 1:lmax+1
        #vec = vcat(vec,diag(K[l])[2:end])
        for i in 1:nbin
            push!(vec, K[l][i,i])
        end
        for i in 1:nbin-1
            for j in i+1:nbin
                push!(vec, K[l][j,i])
            end
        end
    end
    return vec
end

#   @adjoint function vectorKvec2vecmat(allk::Vector{Float64}, nbin::Int64,
#   lmax::Int64, freeparamn::Int64) y = vectorKvec2vecmat(allk, nbin, lmax,
#   freeparamn) function vectorKvec2vecmatpullback(ȳ) k̄ = zeros(length(allk)) s
#   = 1 for l in 1:lmax+1 for i in 2:2nbin k̄[s+(l-1)freeparamn] = ȳ[l][i,i] s +=
#   1 end for i in 1:2nbin-1 for j in i+1:2nbin push!(k̄, ȳ[l][j,i]) end end end
#   return (k̄, nothing, nothing, nothing) #return (vectorKvecmat2vec(ȳ, nbin,
#   lmax), nothing, nothing, nothing) end return y, vectorKvec2vecmat_pullback
#   end

@adjoint function vectorK_vec2vecmat(all_k, nbin::Int64, lmax::Int64, freeparam_n::Int64)
    y = vectorK_vec2vecmat(all_k, nbin, lmax, freeparam_n)
    function vectorK_vec2vecmat_pullback(ȳ)
        k̄ = []
        for l in 1:lmax+1
            #vec = vcat(vec,diag(K[l])[2:end])
            for i in 1:nbin
                push!(k̄, ȳ[l][i,i])
            end
            for i in 1:nbin-1
                for j in i+1:nbin
                    push!(k̄, ȳ[l][j,i])
                end
            end
        end
        return (k̄, nothing, nothing, nothing)
        #return (vectorK_vecmat2vec(ȳ, nbin, lmax), nothing, nothing, nothing)
    end
    return y, vectorK_vec2vecmat_pullback
end 

#   Manipulating x_{\ell m} and L_{\ell} / K_{\ell}
#   ===============================================

#   Convert vector of matrices of x_{\ell m} to a flatten vector of all the
#   matrices' components and viceversa with differentiation rule.

function x_vecmat2vec(x, lmax::Int64, nbin::Int64) #::Vector{Any}
    all_x_per_field = reduce(hcat, x)
    vec_x = reshape(all_x_per_field, (nbin*(numberOfAlms(lmax)*2 - (lmax+1)), 1))
    return vec(vec_x)
end

function x_vec2vecmat(vec_x::Vector{Float64}, lmax::Int64, nbin::Int64)
    #all_x_per_field = transpose(reshape(vec_x, (2*numberOfAlms(lmax)-(lmax+1), 2*nbin)))
    all_x_per_field = reshape(vec_x, ( nbin, 2*numberOfAlms(lmax)-(lmax+1)))
    x = Vector{Matrix{Float64}}(undef, lmax+1)
    for l in 0:lmax
        j_in = l^2 + 1
        j_fin = j_in + 2*l
        x[l+1] = all_x_per_field[:,j_in:j_fin]
    end
    return x
end

@adjoint function x_vec2vecmat(vec_x::Vector{Float64}, lmax::Int64, nbin::Int64)
    y = x_vec2vecmat(vec_x, lmax, nbin)
    function x_vec2vecmat_pullback(ȳ)
        return (x_vecmat2vec(ȳ, lmax, nbin), nothing, nothing)
    end
    return y, x_vec2vecmat_pullback
end

#   Convert vector of matrices of L_{\ell} to a flatten vector of all the
#   matrices' components and viceversa with differentiation rule.

function L_vecmat2vec(L, lmax::Int64, nbin::Int64)
    vec_L = reshape(L[1], (nbin^2, 1))
    for l in 1:lmax
        vec_L = vcat(vec_L, reshape(L[l+1], (nbin^2, 1)))
    end
    return vec(vec_L)
end

function L_vec2vecmat(vec_L::Vector{Float64}, lmax::Int64, nbin::Int64)
    L = Vector{Matrix{Float64}}(undef, lmax+1)
    for i in 0:lmax
        L[i+1] = reshape(vec_L[i*(nbin^2)+1:(i+1)*(nbin^2)], (nbin, nbin))
    end
    return L
end

@adjoint function L_vec2vecmat(vec_L::Vector{Float64}, lmax::Int64, nbin::Int64)
    y = L_vec2vecmat(vec_L, lmax, nbin)
    function L_vec2vecmat_pullback(ȳ)
        return (L_vecmat2vec(ȳ, lmax, nbin), nothing, nothing)
    end
    return y, L_vec2vecmat_pullback
end
