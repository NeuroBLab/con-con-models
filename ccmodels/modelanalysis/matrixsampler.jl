using DataFrames
using CSV
using Statistics

#Number of angles
const nangles = 8

#Indices of E,I,X for several arrays
const Ec = 1
const Ic = 2
const Xc = 3

#Indices of the columns and rows for the table
const Eix = 1:nangles
const Iix = 1+nangles:2*nangles
const Xix = 1+2*nangles:3*nangles
const Nrows = 2*nangles
const Ncols = 3*nangles

"""
Fast function to shift each i-th row in rates by a quantity specified in the 
i-th element of shifts
"""
function shift_multi(rates, shifts)
    #Declare variables
    m,n = size(rates);
    B = Matrix{Float64}(undef, m, n) 

    #Do the circular shift
    @inbounds @simd for i = 1 : m
        @views B[i, :] .=  [rates[i, n - shifts[i] + 1 : n]; rates[i, 1 : n - shifts[i]]]
    end

    return B
end

"""
Obtain the information from the CSV files
"""
function readCSVdata(prepath="data")
    #Read the file
    fracfile = DataFrame(CSV.File("$prepath/model/fractions_populations.csv"))

    #Get the fraction of EIX, in that order
    fractions :: Vector{Float64} = [fracfile[1,2], fracfile[2,2], fracfile[3,2]]

    #Get the fraction of each angle for L23
    #Note that in the CSV ET and XT are intertwined so we need to 
    #use the fancy slicing
    total_ET ::Float64 = fracfile[4,2]
    tunedfrac :: Vector{Float64} = fracfile[8:2:8+15, 2] / total_ET
    #Tunedfrac goes twice in order to account for inhibitory populations
    fractions = [fractions; tunedfrac; tunedfrac] 

    #Same for L4
    total_XT :: Float64= fracfile[6,2]
    tunedfrac = fracfile[9:2:8+15, 2] / total_XT
    fractions = [fractions; tunedfrac]


    #Read connectomics probabilities of connection
    probfile = DataFrame(CSV.File("$prepath/model/prob_connectomics.csv"))
    probtable = Matrix{Float64}(probfile[:, 2:end])

    #Get the weights between two different populations
    #in a nice dictionary, using E=1, I=2, X=3
    popweights = Dict{Tuple{Int64, Int64}, Vector{Float64}}()

    #Read the unit table and add an id to each row
    units = DataFrame(CSV.File("$prepath/preprocessed/unit_table.csv"))
    units[:, :id] .= 1:nrow(units)

    #Get the ids of each population we are interested in
    L23ix :: Vector{Int} = filter([:layer, :tuning_type] => (l, t) -> (l=="L23")&&(t != "not_matched"), units).id
    L4ix :: Vector{Int} = filter([:layer, :tuning_type] => (l, t) -> (l=="L4")&&(t != "not_matched"), units).id
    pref_ori :: Vector{Int} = Int.(filter(:tuning_type => t -> t != "not_matched", units).pref_ori)
    tunedL4  :: Vector{Int}   = Int.(filter([:layer, :tuning_type] => (l, t) -> (t != "not_selective")&&(t != "not_matched")&&(l=="L4"), units).pref_ori)

    #Normalize synaptic volumes
    links = DataFrame(CSV.File("$prepath/preprocessed/connections_table.csv"))
    links = filter([:pre_pt_root_id, :post_pt_root_id] => (pre, post) -> pre != post, links)
    links.syn_volume ./= mean(links.syn_volume)

    #Get the indices of each type of unit, because we will use that to Obtain
    #the weights among populations...
    E_units = filter([:cell_type, :layer] => (c,l) -> c=="exc" && l=="L23", units)
    I_units = filter([:cell_type, :layer] => (c,l) -> c=="inh" && l=="L23", units)
    X_units = filter([:cell_type, :layer] => (c,l) -> c=="exc" && l=="L4", units)

    #...get them as Set to speed up a lot the search below
    Eixs :: Set{Int} = Set(E_units.pt_root_id)
    Iixs :: Set{Int} = Set(I_units.pt_root_id)
    Xixs :: Set{Int} = Set(X_units.pt_root_id)

    #Get the synaptic weights from each pair of populations
    popweights[(1,1)] = filter([:pre_pt_root_id, :post_pt_root_id] => (pre, post) -> (pre in Eixs)&&(post in Eixs), links).syn_volume
    popweights[(1,2)] = filter([:pre_pt_root_id, :post_pt_root_id] => (pre, post) -> (pre in Iixs)&&(post in Eixs), links).syn_volume
    popweights[(1,3)] = filter([:pre_pt_root_id, :post_pt_root_id] => (pre, post) -> (pre in Xixs)&&(post in Eixs), links).syn_volume
    popweights[(2,1)] = filter([:pre_pt_root_id, :post_pt_root_id] => (pre, post) -> (pre in Eixs)&&(post in Iixs), links).syn_volume
    popweights[(2,2)] = filter([:pre_pt_root_id, :post_pt_root_id] => (pre, post) -> (pre in Iixs)&&(post in Iixs), links).syn_volume
    popweights[(2,3)] = filter([:pre_pt_root_id, :post_pt_root_id] => (pre, post) -> (pre in Xixs)&&(post in Iixs), links).syn_volume

    return fractions, probtable, popweights, L23ix, L4ix, pref_ori, tunedL4
end

"""
Read and process the rates from the files so they can be used immediately
"""
function get_rates!(pref_ori, L23ix, L4ix, tunedL4, prepath="data")
    #Number of exc and tuned neurons in L23 and L4 respectively
    ne = length(L23ix)
    ntunedx = length(tunedL4)

    ratetable = DataFrame(CSV.File("$prepath/preprocessed/activity_table.csv"))

    activity = ratetable[:, :rate]

    #Reshape and tranpose in order to have the rates as a matrix with the correct shape
    rates = Matrix(transpose(reshape(activity, (16, length(activity)÷16))))
    #Shift everybody to zero, then take the first 8 angles. Shift_multi only works
    #with positive quantities so -pref_ori is 16 - po in mod algebra 
    rates = shift_multi(rates, 16 .- pref_ori)
    rates = rates[:, 1:nangles]
    
    #Return everybody where it was, with angles from 0 to 7
    @. pref_ori = mod(pref_ori, nangles)
    rates = shift_multi(rates, pref_ori)

    #Then concatenate results in the desired order an return 
    pref_ori .= [pref_ori[L23ix]; pref_ori[L4ix]]
    return [rates[L23ix, :]; rates[L4ix, :]], ne, ntunedx
end

"""
Sample rates in L4. It will overwrite rates_sampled with the correct result,
"""
function sampleL4rates(rates_L4, rates_sampled, fractions_L4, ntunedx, L4oris, mode)

    nsample, _ = size(rates_sampled)
    nx, _ = size(rates_L4)

    #For random rates, just take whatever from the data
    if mode == "random"
        idx = rand(1:nx, nsample)
        rates_sampled .= rates_L4[idx, :]
    else
        #Here, first check how many tuned neurons we have 
        frac_tuned = ntunedx / nx
        ntuned_sample = Int(round(nsample * frac_tuned)) 

        #Shift the neurons to po = 0
        shifted_L4 = shift_multi(rates_L4, nangles .- L4oris)

        #Take the untuned neurons and set them to their mean
        @views shifted_L4[ntunedx+1:end, :] .= mean(shifted_L4[ntunedx+1:end, :], dims=2)

        #Sample ids for selected rates to use 
        idx_tuned = rand(1:ntunedx, ntuned_sample)
        idx_untuned = rand(ntunedx+1:nx, nsample - ntuned_sample)

        #Sample by slicing the experimental (shifted) dasta with the preivous ids 
        rates_sampled .= shifted_L4[[idx_tuned; idx_untuned], :]
        new_oris = zeros(Int64, ntuned_sample)

        #Get the number of neurons in each orientation
        npop = zeros(Int64, nangles+1) 
        @. npop[2:end] = Int(round(ntuned_sample * fractions_L4)) 
        npop = cumsum(npop)
        #Sometimes round will give 1 extra and produce an error, avoid it
        npop[end] = min(npop[end], ntuned_sample)

        #Set the new orientation of the neurons by hand
        @inbounds @simd for i=1:nangles
            new_oris[1+npop[i]:npop[i+1]] .= i-1
        end

        #Reshift and return
        @views rates_sampled[1:ntuned_sample, :] .= shift_multi(rates_sampled[1:ntuned_sample, :], new_oris)
    end

    return nothing
end

"""
Sample a matrix 
"""
function sample_matrix!(Q, ne, ni, nx, k_ee, J, g, probtable, fractions, popweights, cos_L23, cos_L4, cos_I; tunedinh=false, prepath="data")

    N = ne + ni + nx

    #Compute the cosine modulation for them
    cosvalues = cos.(2π*(0:nangles-1)/nangles)

    modulated_EE = cos_L23[1] .+ cos_L23[2] * cosvalues
    modulated_EX = cos_L4[1] .+ cos_L4[2] * cosvalues
    modulated_EI = cos_I[1] .+ cos_I[2] * cosvalues

    #Scale the connection probability to fix kee
    scaling_prob = k_ee / (N * fractions[Ec] * probtable[Ec, Ec])

    #Time to set the probabilities...
    ptable = Matrix{Float64}(undef, nangles*2, nangles*3)

    for i=1:nangles 
        #Fill all the probabilities for the Exc L23 neurons 
        ptable[i, Eix] .=  probtable[Ec, Ec] * circshift(modulated_EE, i-1)
        ptable[i, Iix] .=  probtable[Ec, Ic] * (1 .+ tunedinh * circshift(modulated_EI, i-1))
        ptable[i, Xix] .=  probtable[Ec, Xc] * circshift(modulated_EX, i-1)

        #Very same thing for inh. There's no L4 so this is all
        ptable[i+nangles, Eix] .=  probtable[Ic, Ec] * (1 .+ tunedinh * circshift(modulated_EE, i-1)) 
        ptable[i+nangles, Iix] .=  probtable[Ic, Ic] * (1 .+ tunedinh * circshift(modulated_EI, i-1)) 
        ptable[i+nangles, Xix] .=  probtable[Ic, Xc] * (1 .+ tunedinh * circshift(modulated_EX, i-1)) 
    end

    ptable .*= scaling_prob

    #Mark the positions of the modules, which coincide with the number of 
    #neurons that with a certain tuning. Basically, start_col is [1, n_neurons_theta=0, n_neurons_theta=0]
    ones8 = ones(8)
    frac_conect = [ones8*ne/N; ones8*ni/N; ones8*nx/N]
    start_col = Vector{Int}(undef, Ncols+1)
    start_col[1] = 1
    @views start_col[2:Ncols+1] .= Int.(round.(cumsum(N*fractions[4:end] .* frac_conect)))

    frac_conect = [ones8*ne/(ne+ni); ones8*ni/(ne+ni)]
    start_row = Vector{Int}(undef, Nrows+1)
    start_row[1] = 1
    @views start_row[2:Nrows+1] .= Int.(round.(cumsum((ne+ni)*fractions[4:19] .* frac_conect)))

    #Fill the matrix
    Q .= 0. 
    for col = 1:Ncols
        #c0,cf,r0,rf are the coordinates of this block
        c0 = start_col[col]
        cf = start_col[col+1]
        #Which connectomic population are we
        pop_pre = 1 + (col-1) ÷ nangles 
        for row = 1:Nrows 
            r0 = start_row[row]
            rf = start_row[row+1]
            pop_post = 1 + (row-1) ÷ nangles 

            #Get a block of random connections 
            block = Float64.(rand(1+rf-r0, 1+cf-c0) .< ptable[row, col]) 
            n_synapses = Int.(sum(block))
            #Scale the block's weights sampling from the popweights
            block[block .> 0] .*= rand(popweights[(pop_post, pop_pre)], n_synapses)
            #Write the block in the matrix
            Q[r0:rf, c0:cf] .= block 
        end
    end

    #Negative weights for I neurons
    Q[1+ne:ne+ni] .*= -g
    Q .*= J

    return nothing 
end
