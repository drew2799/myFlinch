using Zygote: @adjoint
using Zygote
using ChainRules.ChainRulesCore
using Healpix
#using BenchmarkTools

#   adjoint_alm2map/map2alm
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

function adjoint_alm2map(v::Vector{Float64}; lmax=nothing, mmax=nothing)
    map = HealpixMap{Float64,RingOrder}(v)
    nside = map.resolution.nside
    lmax = isnothing(lmax) ? 3 * nside - 1 : lmax
    mmax = isnothing(mmax) ? lmax : mmax
    nalms = numberOfAlms(lmax, mmax)
    alm = Alm(lmax, mmax, zeros(ComplexF64, nalms))
    adjoint_alm2map!(map, alm)
    return alm
end

function adjoint_alm2map(map::HealpixMap{Float64,RingOrder,Array{Float64,1}}; lmax=nothing, mmax=nothing)
    nside = map.resolution.nside
    lmax = isnothing(lmax) ? 3 * nside - 1 : lmax
    mmax = isnothing(mmax) ? lmax : mmax
    nalms = numberOfAlms(lmax, mmax)
    alm = Alm(lmax, mmax, zeros(ComplexF64, nalms))
    adjoint_alm2map!(map, alm)
    return alm
end

function adjoint_alm2map(map::PolarizedHealpixMap{Float64,RingOrder,Array{Float64,1}}; lmax=nothing, mmax=nothing)
    nside = map.i.resolution.nside
    lmax = isnothing(lmax) ? 3 * nside - 1 : lmax
    mmax = isnothing(mmax) ? lmax : mmax
    nalms = numberOfAlms(lmax, mmax)
    alms = [ Alm(lmax, mmax, zeros(ComplexF64, nalms)),
        Alm(lmax, mmax, zeros(ComplexF64, nalms)),
        Alm(lmax, mmax, zeros(ComplexF64, nalms)) ]
    adjoint_alm2map!(map, alms)
    return alms
end

function adjoint_alm2map(map::HealpixMap{T,RingOrder,AA}; lmax=nothing, mmax=nothing) where {T <: Real, AA <: AbstractArray{T,1}}
    map_float = HealpixMap{Float64,RingOrder}(convert(Array{Float64,1}, map.pixels))
    return adjoint_alm2map(map_float, lmax=lmax, mmax=mmax)
end

function adjoint_alm2map(map::PolarizedHealpixMap{T,RingOrder,AA}; lmax=nothing, mmax=nothing) where {T <: Real, AA <: AbstractArray{T,1}}
    m_i = convert(Array{Float64,1}, map.i)
    m_q = convert(Array{Float64,1}, map.q)
    m_u = convert(Array{Float64,1}, map.u)
    pol_map_float = PolarizedHealpixMap{Float64,RingOrder}(m_i, m_q, m_u)
    return adjoint_alm2map(pol_map_float, lamx=lmax, mmax=mmax)
end

function adjoint_map2alm(alm::Alm{ComplexF64, Array{ComplexF64, 1}}, nside::Integer)
    npix = nside2npix(nside)
    map = HealpixMap{Float64,RingOrder}(zeros(Float64, npix))
    adjoint_map2alm!(alm, map)
    return map
end

function adjoint_map2alm(alm::Array{Alm{ComplexF64,Array{ComplexF64,1}},1}, nside::Integer)
    npix = nside2npix(nside)
    map = PolarizedHealpixMap{Float64,RingOrder}(
        zeros(Float64, npix),
        zeros(Float64, npix),
        zeros(Float64, npix),
    )
    adjoint_alm2map!(alm, map)
    return map
end

function adjoint_map2alm(alm::Alm{T}, nside::Integer) where {T}
    alm_float = Alm{ComplexF64,Array{ComplexF64,1}}(
        alm.lmax,
        alm.mmax,
        convert(Array{ComplexF64,1}, alm.alm),
    )
    return adjoint_map2alm(alm_float, nside)
end

function adjoint_map2alm(alms::Array{Alm{Complex{T},Array{Complex{T},1}},1}, nside::Integer) where {T <: Real}
    lmax = alms[1].lmax
    mmax = alms[1].mmax
    alm_t = Alm(lmax, mmax, convert(Array{ComplexF64,1}, alms[1].alm))
    alm_e = Alm(lmax, mmax, convert(Array{ComplexF64,1}, alms[2].alm))
    alm_b = Alm(lmax, mmax, convert(Array{ComplexF64,1}, alms[3].alm))
    return adjoint_map2alm([alm_t, alm_e, alm_b], nside)
end

#   ChainRulesCore.rrule for alm2map and map2alm
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

function ChainRulesCore.rrule(::typeof(alm2map), alm, nside::Integer)
    p = alm2map(alm, nside)
    project_alm = ChainRulesCore.ProjectTo(alm)
    function alm2map_pullback(p̄)
        a = alm_scalar_prod(2., adjoint_alm2map(ChainRulesCore.unthunk(p̄), lmax=alm.lmax, mmax=alm.mmax))
        for i in 1:(alm.lmax+1)
            a.alm[i] /= 2.
        end
        ā = @thunk(project_alm(a))
        return ChainRulesCore.NoTangent(), ā, ChainRulesCore.NoTangent()
    end
    return p, alm2map_pullback
end

function ChainRulesCore.rrule(::typeof(map2alm), map)
    nside = map.resolution.nside
    a = map2alm(map)
    project_map = ChainRulesCore.ProjectTo(map)
    function map2alm_pullback(ā)
        p̄ = @thunk( conj(adjoint_map2alm(conj(ā), nside)) )
        return ChainRulesCore.NoTangent(), project_map(p̄), ChainRulesCore.NoTangent()
    end
    return a, map2alm_pullback
end

#   Other functions:
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

## Posso anche scriverla come CahinRulesCore.rrule() ?
@adjoint function Alm{T,AA}(lmax, mmax, arr::AA) where {T <: Number,AA <: AbstractArray{T,1}}
    #(numberOfAlms(lmax, mmax) == length(arr)) || throw(DomainError())
    y = Alm{T,AA}(lmax, mmax, arr::AA)
    function Alm_pullback(ȳ)
        return (nothing, nothing, ȳ.alm)
    end
    return y, Alm_pullback
end

@adjoint function HealpixMap{Float64,RingOrder,Array{Float64,1}}(pix::Vector{Float64})
    y = HealpixMap{Float64,RingOrder,Array{Float64,1}}(pix)
    function HealpixMap_pullback(ȳ)
        return (ȳ.pixels,)
    end
    return y, HealpixMap_pullback
end

function constructor_VecHealpixMap(pixs::Matrix{Float64})
    y = HealpixMap{Float64,RingOrder,Array{Float64,1}}.([pixs[:,i] for i in 1:size(pixs)[2]])
    return y
end

@adjoint function constructor_VecHealpixMap(pixs::Matrix{Float64})
    y = constructor_VecHealpixMap(pixs)
    function VecHealpixMap_pullback(ȳ)
        x̄ = Matrix{Float64}(undef, size(pixs))
        for i in 1:size(x̄)[2]
            x̄[:,i] = ȳ[i]
        end
        return x̄
    end
    return y, VecHealpixMap_pullback
end

#   function ChainRulesCore.rrule(::typeof(+), alm₁::Alm{Complex{T}},
#   alm₂::Alm{Complex{T}}) where {T <: Number} y = alm₁ + alm₂ function
#   almsumpullback(ȳ) ā₁ = @thunk(ȳ.alm) ā₂ = @thunk(ȳ.alm) return
#   ChainRulesCore.NoTangent(), ā₁, ā₂ end return y, almsumpullback end

function ChainRulesCore.rrule(::typeof(sum), map::HealpixMap{T,O,AA}) where{T, O <:Order, AA<:AbstractArray{T,1}}
    y = sum(map)
    function sum_pullback(ȳ)
        x̄ = @thunk( HealpixMap{T,O,AA}(fill!(similar(map.pixels), ȳ) ))
        return ChainRulesCore.NoTangent(), x̄
    end
    return y, sum_pullback
end

@adjoint function Base.sum(map::HealpixMap{T,O,AA}) where{T, O <:Order, AA<:AbstractArray{T,1}}
    y = sum(map)
    function sum_pullback(ȳ)
        return (HealpixMap{T,O,AA}(fill!(similar(map.pixels), ȳ)), )
    end
    return y, sum_pullback
end

function Base.sum(alm::Alm{ComplexF64, Array{ComplexF64, 1}})
    return sum(alm.alm)
end

function ChainRulesCore.rrule(::typeof(sum), alm::Alm{ComplexF64, Array{ComplexF64, 1}})
    y = sum(alm)
    function sum_pullback(ȳ)
        x̄ = @thunk( Alm(alm.lmax, alm.mmax, fill!(similar(alm.alm), ȳ)) )
        return ChainRulesCore.NoTangent(), x̄
    end
    return y, sum_pullback
end
    
@adjoint function Base.sum(alm::Alm{ComplexF64, Array{ComplexF64, 1}})
    y = sum(alm)
    function sum_pullback(ȳ)
        return ( Alm(alm.lmax, alm.mmax, fill!(similar(alm.alm), ȳ)), )
    end
    return y, sum_pullback
end

function alm_prod(alm1::Alm{ComplexF64, Vector{ComplexF64}}, alm2::Alm{ComplexF64, Vector{ComplexF64}})
    y = conj(alm1) * alm2
    return y
end

@adjoint function alm_prod(alm1::Alm{ComplexF64, Vector{ComplexF64}}, alm2::Alm{ComplexF64, Vector{ComplexF64}})
    y = alm_prod(alm1, alm2)
    function prod_pullback(ȳ)
        x̄1 = conj(ȳ)*alm2
        x̄2 = alm1*ȳ
        return (x̄1, x̄2)
    end
    return y, prod_pullback
end

function Base.conj(alm::Alm{ComplexF64, Array{ComplexF64, 1}})
    return Alm(alm.lmax, alm.mmax, conj(alm.alm))
end

@adjoint function Base.conj(alm::Alm{ComplexF64, Array{ComplexF64, 1}})
    y = conj(alm)
    function conj_pullback(ȳ)
       # return (Alm(alm.lmax, alm.mmax, conj(ȳ.alm)), )
        return (conj(ȳ), )
        #return (Alm(nothing, nothing, conj(ȳ.alm)), )
    end
    return y, conj_pullback
end      

#   function Base.length(alm::Alm{ComplexF64, Array{ComplexF64, 1}}) return
#   length(alm.alm) end function Base.iterate(alm::Alm{ComplexF64,
#   Array{ComplexF64, 1}}, i::Int64) return iterate(alm.alm, i) end

function alm_scalar_prod(l::Float64, alm::Alm{ComplexF64, Vector{ComplexF64}})
    return Alm(alm.lmax, alm.mmax, l * alm.alm)
end
