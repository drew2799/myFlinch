using Distributions
using LinearAlgebra
using StatsFuns
using StatsBase

using LogDensityProblems
using LogDensityProblemsAD
using AbstractDifferentiation
using MCMCDiagnosticTools
using AdvancedHMC
using MicroCanonicalHMC
using Pathfinder
using Transducers

using Healpix
using HealpixMPI
using MPI

using Plots
using StatsPlots
using LaTeXStrings
using DataFrames

using Random
using ProgressMeter
using BenchmarkTools
using Chairmarks
using NPZ
using CSV
using Test

using Zygote: @adjoint
using Zygote
using ChainRules.ChainRulesCore

include("AD_parallSHT.jl")
include("gen_datatest.jl")
include("reparam.jl")
include("utils.jl")
include("inference_model.jl")
include("neglogproblem.jl")

seed = 1123

#   RESOLUTION PARAMETERS
nside = 64
lmax = 2*nside - 1

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0
ncore = 64

#   REALIZATION MAP
realiz_Cl, realiz_HAlm, realiz_HMap = Realization("Capse_fiducial_Dl.csv", nside, lmax, seed)
realiz_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([realiz_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(realiz_Cl))

d = length(realiz_θ)

#   SURVEY MASK
mask_512 = readMapFromFITS("wmap_temperature_kq85_analysis_mask_r9_9yr_v5.fits",2,Float64)
mask_nside = udgrade(nest2ring(mask_512), nside)
for i in 1:length(mask_nside.pixels)
    if mask_nside.pixels[i]<=0.5
        mask_nside.pixels[i]=1
    else
        mask_nside.pixels[i]=0
    end
end

#   GENERATED DATA MEASUREMENTS
#   Noise
Random.seed!(1123)
Bl = gaussbeam(0.001, lmax, pol=false)
Pl = pixwin(nside, pol=false)[1:lmax+1]

ϵ=1
N = ϵ*ones(nside2npix(nside))
N[mask_nside.==1] .= 5*10^4
#   Data Map
gen_Cl, gen_HAlm, gen_HMap = Measurement(realiz_Cl, Bl, Pl, mask_nside, N, nside, lmax, seed)
gen_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([gen_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(gen_Cl))
invN_HMap = HealpixMap{Float64,RingOrder}(1 ./ N)

#   STARTING POINT
start_Cl, start_HAlm, start_HMap = StartingPoint(gen_Cl, nside)
start_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([start_HAlm], lmax, 1, comm, root=root), lmax, 1, comm, root=root), Cl2Kl(start_Cl))

#   PROMOTE HEALPIX.MAP TO HEALIPIXMPI.DMAP
gen_DMap = DMap{RR}(comm)
invN_DMap = DMap{RR}(comm)
HealpixMPI.Scatter!(gen_HMap, gen_DMap, comm, clear=true)
HealpixMPI.Scatter!(invN_HMap, invN_DMap, comm, clear=true)

helper_DMap = deepcopy(gen_DMap)

nlp = nℓπ(start_θ, data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, Bl=Bl, Pl=Pl, ncore=ncore, comm=comm, root=root)
nlp_grad = nℓπ_grad(start_θ, data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, invN=invN_DMap, Bl=Bl, Pl=Pl, ncore=ncore, comm=comm, root=root)

## BENCHMARKING POSTERIOR and POSTERIOR+GRADIENTS TIMINGS

MPI.Barrier(comm)

nlp_bm = @be nℓπ(start_θ) samples=1000 evals=50 seconds=Inf

MPI.Barrier(comm)

nlp_grad_bm = @be nℓπ_grad(start_θ) samples=1000 evals=50 seconds=Inf

print(median(nlp_bm).time, "\n")
print(median(nlp_grad_bm).time)

#=
## PATHFINDER INITIALIZATION
struct PF_LogTargetDensity
    dim::Int
end
LogDensityProblems.capabilities(::Type{PF_LogTargetDensity}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::PF_LogTargetDensity) = p.dim
LogDensityProblems.logdensity(p::PF_LogTargetDensity, θ) = -nℓπ(θ)

PF_problem = ADgradient(:Zygote, PF_LogTargetDensity(d))

PFinit_θ = Vector{Vector{Float64}}(undef, 5)
for i in 1:5
    PFinit_θ[i] = rand(MvNormal(start_θ,0.01*I))
end

MPI.Barrier(comm)
t0 = time()
result = multipathfinder(PF_problem, 10; init=PFinit_θ)
PF_t = time()-t0

#Checking goodness of PF initialization
#PF_alm_0 = x_vec2vecmat(result.draws[1:end-lmax-1,end], lmax, 1, comm, root)
#PF_map_0 = Healpix.alm2map(from_alm_to_healpix_alm(PF_alm_0, lmax, 1)[1], nside)

#PF_start_θ = mean(result.draws, dims=2)[:,1]

npzwrite("MPI_chains/PATHinit_$nside.npy", result.draws)

MPI.Finalize()
=#