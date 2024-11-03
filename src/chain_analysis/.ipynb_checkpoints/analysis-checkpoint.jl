using Distributions
using Plots
using StatsPlots
using LinearAlgebra
using Random
using LaTeXStrings
using StatsBase
using StatsFuns
using CSV, DataFrames
using NPZ
using Healpix
using AdvancedHMC
using MicroCanonicalHMC

include("diffSHT_healpix.jl")
include("utilities.jl")

Random.seed!(1123)

function Cl2Kl(Cl)
    C̃ = Cl./1_500
    return norminvcdf.(0,1,C̃)
end

function Kl2Cl(Kl)
    return 1_500 .* normcdf.(0,1,Kl)
end

@adjoint function Kl2Cl(Kl)
    
    y = Kl2Cl(Kl)
    
    function Kl2Cl_PB(ȳ)
        x̄ = 1_500 .* pdf.(Normal(0,1), Kl) .* ȳ
        return (x̄,)
    end
    return y, Kl2Cl_PB
end     

nside=256
lmax=511
algo="MCHMC"

realiz_Dl = CSV.read("Capse_fiducial_Dl.csv", DataFrame)[1:lmax-1,1]
realiz_Cl = dl2cl(realiz_Dl, 2)
realiz_Cl[1] += 1e-10
realiz_Cl[2] += 1e-10
realiz_map = synfast(realiz_Cl, nside);

#   MASKED
#   ======

chain = npzread("../MPI_chains/mask_$(algo)_nside_$(nside).npy")

EssRhat = npzread("../MPI_chains/mask_$(algo)_EssRhat_nside_$(nside).npy");

if algo != "MCHMC"
    stats = npzread("../MPI_chains/mask_$(algo)_stats_nside_$(nside).npy")
end

SumPerf = npzread("../MPI_chains/mask_$(algo)_SumPerf_nside_$(nside).npy");

#   Performances
#   ––––––––––––

histogram(EssRhat[1:end-3,2], color="goldenrod2", label=algo, lw=0, xlim=(0.998,1.1), title="Gelman-Rubin")
vline!([median(EssRhat[1:end-3,2])], color="black", label="median="*string(round(median(EssRhat[1:end-3,2]),digits=3)), lw=2)
savefig("$(algo)_GelmanRubin.pdf")

t = SumPerf[1]

println("$(algo): ess/s = ", string(round(mean(EssRhat[1:end-3,1])/t, digits=5)))

grads = 10_000 # HMC = 1_000*50, NUTS = sum(2 .^ NUTS_stats[2001:3000])

println("$(algo): ess/grad_evals = ", string(round(mean(EssRhat[1:end-3,1])/grads, digits=4)))

println("$(algo): corr length = ", string(round(5_000/mean(EssRhat[1:end-3,1]), digits=4)))

plot(NUTS_stats[125:1000])

#   Reconstruction
#   ––––––––––––––

#HMC_alm = x_vec2vecmat(vec(mean(HMC_chain[1:end-lmax-1,:], dims=2)), lmax, 1)
#HMC_map = alm2map(from_alm_to_healpix_alm(HMC_alm, lmax, 1)[1], nside);

#NUTS_alm = x_vec2vecmat(vec(mean(NUTS_chain[1:end-lmax-1,125:end], dims=2)), lmax, 1)
#NUTS_map = alm2map(from_alm_to_healpix_alm(NUTS_alm, lmax, 1)[1], nside);

alm = x_vec2vecmat(vec(mean(chain[1:end-lmax-4,:], dims=2)), lmax, 1)
map = alm2map(from_alm_to_healpix_alm(alm, lmax, 1)[1], nside);

pyplot()
plot(map, color=:coolwarm, title=algo, colorbar_title=L"\mathrm{\mu K}", clim=(-400,400),
        titlefontsize=30, colorbar_titlefontsize=24, colorbar_tickfontsize=16)
savefig("$(algo)_ReconMap.pdf")

plot(realiz_map, color=:coolwarm, title="Realization", colorbar_title=L"\mathrm{\mu K}", clim=(-400,400),
        titlefontsize=30, colorbar_titlefontsize=24, colorbar_tickfontsize=16)
savefig("$(algo)_RealizMap.pdf")

maps_vec = Vector(undef, 1_000)
for i in 1:1_000
    alm_i = x_vec2vecmat(vec(chain[1:end-lmax-4,i]), lmax, 1)
    maps_vec[i] = deepcopy(alm2map(from_alm_to_healpix_alm(alm_i, lmax, 1)[1], nside))
end

plot(HealpixMap{Float64,RingOrder,Array{Float64,1}}(std(maps_vec)), color=:coolwarm, title=algo, colorbar_title=L"\mathrm{\mu K}",
        titlefontsize=30, colorbar_titlefontsize=24, colorbar_tickfontsize=16)
savefig("$(algo)_StdMap.pdf")

Δ = HealpixMap{Float64,RingOrder,Array{Float64,1}}((map.pixels-realiz_map.pixels)./std(maps_vec))

histogram(Δ.pixels, color="goldenrod2", label=algo, normalize=:pdf, xlim=(-5,5))
plot!(Normal(), color="black", label="Normal", plot_title="Pixel residuals")
savefig("$(algo)_Residuals.pdf")

p = plot(layout=(2,3), size=(900,600),  xtickfontsize=10, ytickfontsize=10, xguidefontsize=15, yguidefontsize=15, 
    legend_font_pointsize=8, titlefontsize=16, xformatter=:latex, yformatter=:latex)

ℓs = [6, 16, 420, 620, 820, 1020]

for i in 1:6

    density!(p, Kl2Cl(chain[end-lmax-4+ℓs[i],:]), color="goldenrod2", lw=2, 
        fillrange=0, fillalpha=0.75, label=algo, subplot=i)
    vline!(p, [realiz_Cl[ℓs[i]]], color="black", label="Realization", lw=2, 
        title=L"\ell="*latexstring(ℓs[i]), xlabel=L"C_{\ell}", subplot=i, bottom_margin=7Plots.mm)
end
display(p)
savefig("$(algo)_ClDistr.pdf")

Kl = vec(median(chain[end-lmax-3:end-3,:], dims=2))
σ_Kl = vec(std(chain[end-lmax-3:end-3,:], dims=2))
Cl = Kl2Cl(Kl)
σ_Cl = abs.(diag(jacobian(x->Kl2Cl(x), Kl)[1])).*σ_Kl;

dl_c = [l*(l+1)/2π for l in 0:lmax];

p = plot(layout=@layout([a;b{0.25h}]), plot_title=L"\mathrm{Power \ Spectrum}", xtickfontsize=10, ytickfontsize=10, xguidefontsize=15, yguidefontsize=15, 
    legend_font_pointsize=10, titlefontsize=16, size=(650,450), xformatter=:latex, yformatter=:latex)

plot!(p, cl2dl(MCHMC_Cl,0), color="goldenrod2", label=algo, yerror=dl_c.*σ_Cl, 
    markershape=:circle, markerstrokecolor="goldenrod2", subplot=1)
plot!(p, cl2dl(realiz_Cl,0), label = L"\mathrm{fiducial}", color="black", 
    lw=2, ylabel=L"D_\ell", subplot=1, left_margin=2Plots.mm)

scatter!(p, (cl2dl(Cl,0).-(cl2dl(Cl,0)))./(dl_c.*σ_Cl), color="goldenrod2", label = "", markershape=:circle, 
    markerstrokecolor="goldenrod2", subplot=2, xlabel=L"\ell", ylabel=L"\Delta D_\ell / \mathrm{STD}")
hline!([0], color="black", lw=2, subplot=2, label="")
savefig("$(algo)_PowSpec.pdf")

#infer_Dl = Kl2Cl(HMC_chain[end-lmax:end,:])

#=for i in 1:1000
    infer_Dl[:,i]=cl2dl(infer_Dl[:,i], 0)
end=#

#=infer_Dl_stats_95 = []
for i in 1:512
    push!(infer_Dl_stats_95, quantile(infer_Dl[i,:], [0.025,0.5,0.975]))
end
infer_Dl_stats_68 = []
for i in 1:512
    push!(infer_Dl_stats_68, quantile(infer_Dl[i,:], [0.16,0.5,0.84]))
end=#

#=for i in 1:512
    infer_Dl_stats_95[i][1]=infer_Dl_stats_95[i][2]-infer_Dl_stats_95[i][1]
    infer_Dl_stats_95[i][3]=infer_Dl_stats_95[i][3]-infer_Dl_stats_95[i][2]
end
for i in 1:512
    infer_Dl_stats_68[i][1]=infer_Dl_stats_68[i][2]-infer_Dl_stats_68[i][1]
    infer_Dl_stats_68[i][3]=infer_Dl_stats_68[i][3]-infer_Dl_stats_68[i][2]
end=#

#=scatter(reduce(hcat, infer_Dl_stats_95)[2,:], yerror=(reduce(hcat, infer_Dl_stats_95)[1,:],reduce(hcat, infer_Dl_stats_95)[3,:]), markershape=:hline,
markerstrokecolor="lightblue4", color="lightblue4", label="95% CL")
scatter!(reduce(hcat, infer_Dl_stats_68)[2,:], yerror=(reduce(hcat, infer_Dl_stats_68)[1,:],reduce(hcat, infer_Dl_stats_68)[3,:]), markershape=:hline,
markerstrokecolor="blue", color="blue", label="68% CL")
#scatter!(reduce(hcat, infer_Dl_stats_68)[2,:], markershape=:hline,
#markerstrokecolor="black", color="blue", label="")
plot!(cl2dl(realiz_Cl,0), label = L"\mathrm{fiducial}", color="black", lw=2, ylabel=L"D_\ell", left_margin=2Plots.mm)
 =#   

#=plot(reduce(hcat, infer_Dl_stats_95)[2,:], ribbon=(reduce(hcat, infer_Dl_stats_95)[1,:],reduce(hcat, infer_Dl_stats_95)[3,:]), markershape=:none,
markerstrokecolor="lightblue2", color="lightblue2", label="95% CL")
plot!(reduce(hcat, infer_Dl_stats_68)[2,:], ribbon=(reduce(hcat, infer_Dl_stats_68)[1,:],reduce(hcat, infer_Dl_stats_68)[3,:]), markershape=:none,
markerstrokecolor="dodgerblue2", color="dodgerblue2", label="68% CL")

plot!(cl2dl(realiz_Cl,0), label = L"\mathrm{fiducial}", color="black", lw=2, ylabel=L"D_\ell", left_margin=2Plots.mm)
    =#