function nℓπ(θ; data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, Bl=Bl, Pl=Pl, invN=invN_DMap, ncore=ncore, comm=comm, root=root)

    nlp = NegLogPosterior(θ, comm, ncore, helper_DMap, data=data, lmax=lmax, nside=nside, invN=invN, Bl=Bl, Pl=Pl, root = root)

    #if crank == root
    return nlp
    #end
end

function nℓπ_grad(θ; data=gen_DMap, helper_DMap=helper_DMap, lmax=lmax, nside=nside, Bl=Bl, Pl=Pl, invN=invN_DMap, ncore=ncore, comm=comm, root=root)

    nlp_grad = gradient(x->NegLogPosterior(x, comm, ncore, helper_DMap, data=data, lmax=lmax, nside=nside, invN=invN, Bl=Bl, Pl=Pl, root=root), θ)

    #if crank == root
    return nlp_grad
    #end
end
