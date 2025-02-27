using Interpolations
using SpecialFunctions
using LinearAlgebra
using QuadGK

include("matrixsampler.jl")

const sqpi = sqrt(π)

function mu_space(x0=100, xf=10000, xinf=1e7, nsteps=1000)
    return [[-xinf]; LinRange(-x0, xf, nsteps); [xinf]] 
end

function tabulate_reponse!(phi_aux, mu_tab, theta_E, theta_I, sigma_E, sigma_I; V_r = 10.0, tau_E = 0.02, tau_I=0.01, tau_rp = 2e-3)
    phi_aux .= comp_phi_tab.(mu_tab, tau_E, tau_rp, theta_E, sigma_E, V_r)
    #phi_aux .= tanh.(mu_tab) .* (mu_tab .> 0.) 
    phi_E = linear_interpolation(mu_tab, phi_aux)
    phi_aux .= comp_phi_tab.(mu_tab, tau_I, tau_rp, theta_I, sigma_I, V_r)

    return phi_E, linear_interpolation(mu_tab, phi_aux)
end

@fastmath function comp_phi_tab(mu, tau, tau_rp, theta, sigma, V_r; uthr=10)
    min_u = (V_r - mu) / sigma
    max_u = (theta - mu) / sigma

    if max_u < uthr            
        return 1.0/(tau_rp+tau*sqpi*compute_integral(min_u,max_u))
    else
        return max_u/tau/sqpi*exp(-max_u^2)
    end
end

@fastmath function integrand(x; thres = -5.5)
    if x>=thres
        return exp(x*x)*(1 + erf(x))
    else
        x2 = 1.0/(x*x)
        return (0.5*x2  - 0.75*x2*x2 - 1) / (sqpi * x)  
    end
end

function compute_integral(min, max; thres=11.)
    if max < thres
        #integral, error = quadgk(integrand, min, max)
        #return integral
        return sum(integrand.(LinRange(min, max, 10000))) * (max - min) / 10000
    else
        #@fastmath return exp(max*max)/max
        return exp(max*max)/max
    end
end

@inbounds @fastmath function system_euler!(ne, ni, rL23, current, QJ_L23, ext_input, nits, tau_E, tau_I, phi_E, phi_I, hEI, hII; dt=0.01)
    ni0 = 1 + ne
    dtE = dt/tau_E
    dtI = dt/tau_I
    for i=1:nits-1
        current .= QJ_L23 * rL23[:,i] .+ ext_input 

        @. rL23[1:ne, i+1] = (1-dtE)*rL23[1:ne,i] + dtE*phi_E(tau_E * current[1:ne])
        @. rL23[ni0:end, i+1] = (1-dtI)*rL23[ni0:end,i] + dtI*phi_I(tau_I * current[ni0:end])
    end

    return nothing
end

function do_dynamics!(nits, rL23, current, ne, ni, QJ, rx, phi_E, phi_I, hEI, hII; tau_E = 0.02, tau_I = 0.01, dt=0.005)

    QJ_L23 = QJ[:, 1:ne+ni]
    @views QJ_L4  = QJ[:, ne+ni+1:end]
    ext_input = QJ_L4 * rx - [hEI * ones(ne, 8); hII * ones(ni, 8)] 

    finalrates = Matrix{Float64}(undef, ne+ni, 8)
    
    @simd for idx_theta = 1:nangles
        rL23[:, 1] .= 0.         
        system_euler!(ne, ni, rL23, current, QJ_L23, ext_input[:, idx_theta], nits, tau_E, tau_I, phi_E, phi_I, hEI, hII; dt=dt)
        @views finalrates[:, idx_theta] .= rL23[:, end]
    end
    return finalrates 
end

function do_dynamics_noglobals(N, J, g, thetaE, thetaI, sigmaE, sigmaI, hEI, hII; tau_E=0.01, tau_I=0.02, nits=400)
    mu_tab = mu_space()
    phiaux = Vector{Float64}(undef, length(mu_tab))
    phi_E, phi_I = tabulate_reponse!(phiaux, mu_tab, thetaE, thetaI, sigmaE, sigmaI)

    fractions, probtable, popweights, L23ix, L4ix, pref_ori, tunedL4 = readCSVdata("../../data")
    rates, nedata, ntunedxdata = get_rates!(pref_ori, L23ix, L4ix, tunedL4, "../../data")
    ne = round(Int, N * fractions[1])
    ni = round(Int, N * fractions[2])
    nx = round(Int, N * fractions[3])

    rates_L4 = rates[1+nedata:end, :]
    @views L4oris = pref_ori[1+nedata:end]

    rates_sampled = Matrix{Float64}(undef, nx, nangles)
    fractions_L4 = fractions[1+3+2*nangles:end]

    new_oris = sampleL4rates(rates_L4, rates_sampled, fractions_L4, ntunedxdata, L4oris, "normal")

    Q = Matrix{Float64}(undef, ne + ni, N)
    k_ee = 400 
    cos23 = [1.0, 0.4]
    cos4  = [1.0, 0.2]
    cosI  = [0., 0.]
    sample_matrix!(Q, ne, ni, nx, k_ee, J, g, probtable, fractions, popweights, cos23, cos4, cosI; tunedinh=false, prepath="data")

    current = Vector{Float64}(undef, ne+ni)
    rL23 = Matrix{Float64}(undef, ne+ni, nits)

    QJ_L23 = Q[:, 1:ne+ni]
    QJ_L4  = Q[:, ne+ni+1:end]
    ext_input = QJ_L4 * rates_sampled - [hEI * ones(ne, 8); hII * ones(ni, 8)] 
    
    @simd for idx_theta = 1:nangles
        rL23[:, 1] .= 0.         
        system_euler!(ne, ni, rL23, current, QJ_L23, ext_input[:, idx_theta], nits, tau_E, tau_I, phi_E, phi_I, hEI, hII; dt=0.005)
    end
    return nothing
end


function analyse_results!(rates_f, pref_oris, tuning_curve)
    pref_oris .= argmax.(eachrow(rates_f))
    tuning_curve .= mean(shift_multi(rates_f, nangles .- pref_oris), dims=1) 
    return nothing
end

#function make_simulation(hEI, hII, QJ, rates, rL23, current)
#    do_dynamics!(rates, rL23, current, ne, ni, nx, QJ, rx, tau_E, tau_I, phi_E, phi_I, hEI, hII)
#end

function mwe_euler(dt, nits, r, ne, Q_1, ext_input, tauE, tauI, phiE, phiI, current)
    ni0 = 1 + ne
    dtE = dt/tauE
    dtI = dt/tauI
    for i=1:nits-1
        current .= Q_1 * r .+ ext_input 

        @. r[1:ne, i+1] = (1-dtE)*r[1:ne, i] + dtE * phiE(tau_E * current[1:ne])
        @. r[ni0:end, i+1] = (1-dtI)*r[ni0:end, i] + dtI * phiI(tau_I * current[ni0:end])
    end

    return nothing
end

function mwe_caller(nits, ne, ni, r, rx, Q, hEI, hII, dt)
    Q_1 = Q[:, 1:ne]
    Q_2 = Q[:, ne+ni+1:end]
    ext_input = Q_2 * rx - [hEI * ones(ne, 8); hII * ones(ni, 8)] 
    
    @simd for idx_theta = 1:nangles
        r[:, 1] .= 0.         
        mwe_euler(dt, nits, r, ne, Q_1, ext_input[:, idx_theta], tauE, tauI, phiE, phiI, current)
    end
    return nothing
end