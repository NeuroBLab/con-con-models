import numpy as np
import pandas as pd

import sbi.utils as sbiut
import torch

import sys
import os
sys.path.append(os.getcwd())
import ccmodels.modelanalysis.sbi_utils as msbi
import ccmodels.modelanalysis.utils as mut
import ccmodels.utils.watermark as wtm

from scipy.stats import skew

inputfolder = sys.argv[1]
networkname = sys.argv[2]
sample_mode = sys.argv[3]

if sample_mode == 'kin':
    nparameters = 9
else:
    nparameters = 8

datafolder = "data"

def load_rates(simid):
    rate_id = simid // 1000
    part_id = simid - rate_id * 1000
    return np.load(f"{datafolder}/model/simulations/{inputfolder}/{rate_id}_rates{part_id}.npy")

nfiles = 100
params = np.empty((0, nparameters))
summary_stats = np.empty((0,7)) 

for i in range(nfiles):
    inputfile = np.loadtxt(f"{datafolder}/model/simulations/{inputfolder}/{i}.txt")
    if inputfile.size > 0:
        params = np.vstack((params, inputfile[:, :nparameters]))
        #Observe that even if mode is NOT kin, the indegree is always stored as a parameter (it's just 400) so the summary stats start always from 9 and does not depend on nparameters
        #Take the tuning curve at three points: start, minimum, and end 
        cvo, cvd = mut.compute_circular_variance(inputfile[:, 9:17], orionly=True)
        r0 = inputfile[:, 9]
        rf = inputfile[:, 16]

        #Connection probability reduction at beginning and end for L23 adn L4
        #p23 starts at 17, p4 at 22. Each has 5 values
        pL23 = inputfile[:,21] 
        pL4  = inputfile[:,26] 

        midpL23 = inputfile[:,19] 
        midpL4  = inputfile[:,24] 


        stats = np.vstack((r0, rf, cvd, pL23, pL4, midpL23, midpL4)).transpose()

        summary_stats = np.vstack((summary_stats, stats))

        

#Lets add another 16 summary stats, CV + rate dists
sims_per_file = 1000

#Increase the size of the summary stats
summary_stats = np.hstack((summary_stats, np.zeros((summary_stats.shape[0], 5))))

for sim in range(sims_per_file * nfiles):
    rates = load_rates(sim)

    cvo, cvd = mut.compute_circular_variance(rates, orionly=True)
    summary_stats[sim, 7] = np.mean(cvd)
    summary_stats[sim, 8] = np.std(cvd)
    summary_stats[sim, 9] = skew(cvd) 

    logrates = np.log(rates.flatten()) 
    summary_stats[sim, 10] = np.mean(logrates) 
    summary_stats[sim, 11] = np.std(logrates) 

print(summary_stats.shape)

        
if sample_mode == 'kin':
    #Select parameters J,g,sE,sI,hEI,hII,kin, leaving out the two betas
    params = torch.tensor(params[:, [0,1,2,3,4,5,8]])
    summary_stats = torch.tensor(summary_stats)

    #Do the SBI
    j0, jf = 0., 4.
    g0, gf = 0., 5.
    #theta0, thetaf = 20., 20.
    sigmaE0, sigmaEf = 7., 12.
    sigmaI0, sigmaIf = 7., 12.
    hei0, heif = 50., 150.
    hii0, hiif = 100., 500.
    kin0, kinf= 30, 600 

    prior_lowbound =  torch.tensor([j0, g0, sigmaE0, sigmaI0, hei0, hii0, kin0])
    prior_highbound = torch.tensor([jf, gf, sigmaEf, sigmaIf, heif, hiif, kinf])
else:
    params = torch.tensor(params)
    summary_stats = torch.tensor(summary_stats)

    #Do the SBI
    j0, jf = 0., 4.
    g0, gf = 0., 5.
    #theta0, thetaf = 20., 20.
    sigmaE0, sigmaEf = 7., 12.
    sigmaI0, sigmaIf = 7., 12.
    hei0, heif = 50., 150.
    hii0, hiif = 100., 500.
    b230, b23f = 0.1, 0.6 
    b40, b4f   = 0.1, 0.6 

    #prior, intervals = msbi.setup_prior(j0, jf, g0, gf, theta0, thetaf, sigma0, sigmaf, hei0, heif, hii0, hiif)
    prior_lowbound =  torch.tensor([j0, g0, sigmaE0, sigmaI0, hei0, hii0, b230, b40])
    prior_highbound = torch.tensor([jf, gf, sigmaEf, sigmaIf, heif, hiif, b23f, b4f])

prior = sbiut.BoxUniform(low=prior_lowbound, high=prior_highbound)
posterior = msbi.train_sbi(prior, params, summary_stats)

#Save the results
print("Save network at")
print(f"{datafolder}/model/sbi_networks/{networkname}.sbi")
msbi.save_posterior(f"{datafolder}/model/sbi_networks/{networkname}.sbi", posterior)
print("Network saved?")
output = open(f'{datafolder}/model/sbi_networks/{networkname}_metadata.txt', 'w')
gitc = wtm.get_commit_hash()
output.write(f"Network generated with commit hash: {gitc} and sample mode {sample_mode} using L23 L4 CV rates entirely")
output.close()
print("eof")
