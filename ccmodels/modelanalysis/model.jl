using Interpolations
using SpecialFunctions
using QuadGK

include("matrixsampler.jl")

const sqpi = sqrt(π)

function mu_space(x0=100, xf=10000, xinf=1e7, nsteps=10000)
    return [[-xinf]; LinRange(-x0, xf, nsteps); [xinf]] 
end

function tabulate_reponse!(phi_tab_E, phi_tab_I, mu_tab, theta_E, theta_I, sigma_E, sigma_I; V_r = 10.0, tau_E = 0.02, tau_I=0.01, tau_rp = 2e-3)
    phi_tab_E .= comp_phi_tab.(mu_tab, tau_E, tau_rp, sigma_E, theta_E, V_r)
    phi_tab_I .= comp_phi_tab.(mu_tab, tau_I, tau_rp, sigma_I, theta_I, V_r)

    return linear_interpolation(mu_tab, phi_tab_E), linear_interpolation(mu_tab, phi_tab_E)
end

@fastmath function comp_phi_tab(mu, tau, tau_rp, sigma, theta, V_r; uthr=10)
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
        x2 = 1.0/x*x
        #return (0.5*x^(-2) - 0.75*x^(-4) - 1) / (sqpi * x)  
        return (0.5*x2  - 0.75*x2*x2 - 1) / (sqpi * x)  
    end
end

function compute_integral(min, max; thres=11)
    if max < thres
        integral, error = quadgk(integrand, min, max)
        return integral
    else
        return exp(max*max)/max
    end
end

function system_euler!(ne, ni, rL23, rL4, rates, current, QJ, nits, tau_E, tau_I, phi_E, phi_I, hEI, hII; dt=0.01)
    println(size(rates), " ", size([rL23[:, 1]; rL4]))
    rates .= [rL23[:, 1]; rL4]
    #rates .= 0

    ni0 = 1 + ne
    nif = ne + ni 
    dtE = dt/tau_E
    dtI = dt/tau_I

    for i=1:nits-1
        rates[1:nif] .= rL23[:,i]
        current .= QJ * rates

        @views @. rL23[1:ne, i+1] = (1-dtE)*rL23[1:ne,i] + dtE*phi_E(tau_E * (current[1:ne] - hEI))
        @views @. rL23[ni0:end, i+1] = (1-dtI)*rL23[ni0:end,i] + dtI*phi_I(tau_I * (current[ni0:end] - hII))
    end

    return nothing
end

function do_dynamics!(nits, rates, rL23, current, ne, ni, QJ, rx, phi_E, phi_I, hEI, hII, final_rates; tau_E = 0.02, tau_I = 0.01, dt=0.01)

    println("yay")
    for idx_theta = 1:nangles
        rL23[:, 1] .= 0.         
        println(size(rates), " ", size([rL23[:, 1]; rx[:, idx_theta]]))
        #@views system_euler!(ne, ni, rL23, rx[:, idx_theta], rates, current, QJ, nits, tau_E, tau_I, phi_E, phi_I, hEI, hII; dt=dt)
        #final_rates[:, idx_theta] .= rL23[:, end]
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

