#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:55:46 2025

@author: heitor
"""

import numpy as np
from scipy.interpolate import interp1d


import multiprocessing
import multiprocessing as mp


import scipy.optimize.minimize

from lmfit import minimize, Parameters
from multiprocessing import Pool


from data_reader import read_fits_GaiaESO

from makeSB import make_synSB

#---------------------------------------

def chi_squared(params, x, y, LEFT_LIM, RIGHT_LIM):

    #mask = (x < x_l) & (x > x_u)
    #x = x[mask]  
    #y = y[mask]
     
    # function make_synSB
    syb_SB = make_synSB(LEFT_LIM, RIGHT_LIM, params['teff_a'].value,params['logg_a'].value,params['met_a'].value,params['vmic_a'].value,params['FWHM_a'].value,
                params['teff_b'].value,params['logg_b'].value,params['met_b'].value,params['vmic_b'].value,params['FWHM_b'].value,
                params['RV_a'].value,params['RV_b'].value, params['ratio'].value, 
                {'Mg': params['Mg_a'].value},{'Mg': params['Mg_b'].value})

    #x and y of the combined
    #wavelength_b = Out[0][0][0]
    #flux_b = Out[0][1][0]
    
    
    interp_func = interp1d(syb_SB[0][0][0], syb_SB[0][1][0], kind='linear')
    syn_template_flux = interp_func(x)
    
    return y - syn_template_flux


#have this option from fits and also one from the .csv already normalized?
def fit_one_spectrum(CNAME, LEFT_LIM, RIGHT_LIM,  RV1, RV2, method_min, return_dict):
    
    x_l, x_u, grouped_spectra_l_final, grouped_spectra_u_final, dateobs = read_fits_GaiaESO(CNAME)
    
    #JUST DOING THE XL NOW!


    #merle_results = Table.read('../final_gaiaESO_SB2.fits')
    #merle = merle_results[merle_results['CName'] == CName]

    
    params = Parameters()

    params.add('RV_a', RV1, min=-250, max=250, vary=True)
    params.add('teff_a', 5770, min=3500, max=7000, vary=True)
    params.add('logg_a', 4, min=0, max=5.5, vary=True)
    params.add('met_a', 0, min=-3, max=1, vary=True)
    params.add('vmic_a', 1, min=0, max=2, vary=True)
    params.add('FWHM_a', 0.1, min=0, max=1, vary=True)

    params.add('RV_b', RV2, min=-250, max=250, vary=True)
    params.add('teff_b', 5770, min=3500, max=7000, vary=True)
    params.add('logg_b', 4, min=0, max=5.5, vary=True)
    params.add('met_b', 0, min=-3, max=1, vary=True)
    params.add('vmic_b', 1, min=0, max=2, vary=True)
    params.add('FWHM_b', 0.1, min=0, max=1, vary=True)

    params.add('ratio', 1, min=0, max=1, vary=True)
    
    params.add('Mg_a', 0.0, min = -3.0, max = 2.0, vary=True)
    params.add('Mg_b', 0.0, min = -3.0, max = 2.0, vary=True)


    # ??????????? issues with x_u x_l ?
    for i in range(len(grouped_spectra_l_final)):
        mask_l = np.isfinite(grouped_spectra_l_final[i]) & (x_l > LEFT_LIM) & (x_l < RIGHT_LIM)
        #mask_u = np.isfinite(grouped_spectra_u_final[i]) & (x_u > LEFT_LIM) & (x_u < RIGHT_LIM)

    # minimization part
    result = minimize(chi_squared, params, method=method_min, args=(x_l[mask_l], grouped_spectra_l_final[i][mask_l],LEFT_LIM, RIGHT_LIM))
    #result = minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

    
    print(result.params)
    return_dict[dateobs[i]] = result
    
    #minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    
    
    # ERROR :   better values!
    #Value of parameter <log_g> needs to be in range -0.5 to 5.5. You requested 5.436726357156354, and due to missing models we could not interpolate.
    #Interpolation failed? 'NoneType' object is not subscriptable
    #TS completed
    #TS failed


#---------------------------------------

# Example usage
if __name__ == "__main__": 
    
    #CNAME for test 
    CNAME = '12000916-4101004'
    
        
    LEFT_LIM = 5100
    RIGHT_LIM = 5200

    RV1 = -47.51
    RV2 = 18.99

    method_min = 'leastsq'

    manager = mp.Manager()
    return_dict = manager.dict()
    ps = []


    p = multiprocessing.Process(target=fit_one_spectrum, args=(CNAME, LEFT_LIM, RIGHT_LIM,  RV1, RV2, method_min, return_dict))
    ps.append(p)
    p.start()

    for p in ps:
        p.join()

    result = dict(return_dict)

    print(result)





#