#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:24:11 2024

@author: heitor
"""

#turbospectrum
from __future__ import annotations
try:
    from scripts_for_plotting import *
except ModuleNotFoundError:
    import sys
    sys.path.append('../TSFitPy/')
    from scripts_for_plotting import *
    
import joblib
import importlib  
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from astropy.table import Table
import time

import multiprocessing
import multiprocessing as mp

import lmfit
from lmfit import Model
from lmfit import minimize, Parameters
from multiprocessing import Pool
import tqdm

import emcee

from scipy.interpolate import interp1d
#from scipy.optimize import minimize
#from scipy.optimize import minimize, differential_evolution

sns.set_style("white")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})



turbospectrum_paths = {"turbospec_path": "../TSFitPy/turbospectrum/exec-gf/",  # change to /exec-gf/ if gnu compiler
                       "interpol_path": "../TSFitPy/scripts/model_interpolators/",
                       "model_atom_path": "../TSFitPy/input_files/nlte_data/model_atoms/",
                       "departure_file_path": "../TSFitPy/input_files/nlte_data/",
                       "model_atmosphere_grid_path": "../TSFitPy/input_files/model_atmospheres/",
                       "line_list_path": "../TSFitPy/input_files/linelists/linelist_for_fitting/"}



LEFT_LIM = 5100
RIGHT_LIM = 5200
#----------------------------------------------
    
def timer(s):
    d =  s/ (24 * 60 * 60)
    h = (d - int(d)) *24
    m = (h-int(h)) *60
    sec= (m-int(m)) *60

    print("%d:%02d:%02d:%02d" % (d, h, m, sec))

def chebyshev(p, ye, mask, deg):
    coef = np.polynomial.chebyshev.chebfit(p[0][mask], p[1][mask], deg)
    cont = np.polynomial.chebyshev.chebval(p[0], coef)
    return cont

def sclip(p, fit, n, deg, ye=[], sl=99999, su=99999, min=0, max=0, min_data=1, grow=0, verbose=True):
    """
    p: array of coordinate vectors. Last line in the array must be values that are fitted. The rest are coordinates.
    fit: name of the fitting function. It must have arguments x,y,ye,and mask and return an array of values of the fitted function at coordinates x
    n: number of iterations
    ye: array of errors for each point
    sl: lower limit in sigma units
    su: upper limit in sigma units
    min: number or fraction of rejected points below the fitted curve
    max: number or fraction of rejected points above the fitted curve
    min_data: minimal number of points that can still be used to make a constrained fit
    grow: number of points to reject around the rejected point.
    verbose: print the results or not
    """

    nv, dim = np.shape(p)

    # if error vector is not given, assume errors are equal to 0:
    if ye == []:
        ye = np.zeros(dim)
    # if a single number is given for y errors, assume it means the same error is for all points:
    if isinstance(ye, (int, float)):
        ye = np.ones(dim) * ye

    f_initial = fit(p, ye, np.ones(dim, dtype=bool), deg)
    s_initial = np.std(p[-1] - f_initial)

    f = f_initial
    s = s_initial

    tmp_results = []

    b_old = np.ones(dim, dtype=bool)

    for step in range(n):
        # check that only sigmas or only min/max are given:
        if (sl != 99999 or su != 99999) and (min != 0 or max != 0):
            raise RuntimeError(
                "Sigmas and min/max are given. Only one can be used."
            )

        # if sigmas are given:
        if sl != 99999 or su != 99999:
            b = np.zeros(dim, dtype=bool)
            if sl >= 99999 and su != sl:
                sl = su  # check if only one is given. In this case set the other to the same value
            if su >= 99999 and sl != su:
                su = sl

            good_values = np.where(
                ((f - p[-1]) < (sl * (s + ye))) & ((f - p[-1]) > -(su * (s + ye)))
            )  # find points that pass the sigma test
            b[good_values] = True

        # if min/max are given
        if min != 0 or max != 0:
            b = np.ones(dim, dtype=bool)
            if min < 1:
                min = (
                    dim * min
                )  # detect if min is in number of points or percentage
            if max < 1:
                max = (
                    dim * max
                )  # detect if max is in number of points or percentage

            bad_values = np.concatenate(
                (
                    (p[-1] - f).argsort()[-int(max) :],
                    (p[-1] - f).argsort()[: int(min)],
                )
            )
            b[bad_values] = False

        # check the grow parameter:
        if grow >= 1 and nv == 2:
            b_grown = np.ones(dim, dtype=bool)
            for ind, val in enumerate(b):
                if val == False:
                    ind_l = ind - int(grow)
                    ind_u = ind + int(grow) + 1
                    if ind_l < 0:
                        ind_l = 0
                    b_grown[ind_l:ind_u] = False

            b = b_grown

        tmp_results.append(f)

        # check that the minimal number of good points is not too low:
        if len(b[b]) < min:
            step = step - 1
            b = b_old
            break

        # check if the new b is the same as old one and break if yes:
        if np.array_equal(b, b_old):
            step = step - 1
            break

        # fit again
        f = fit(p, ye, b, deg)
        s = np.std(p[-1][b] - f[b])
        b_old = b

    if verbose:
        print("FITTING RESULTS:")
        print()
        print("Number of iterations requested:    ", n)
        print()
        print("Number of iterations performed:    ", step + 1)
        print()
        print("Initial standard deviation:        ", s_initial)
        print()
        print("Final standard deviation:          ", s)
        print()
        print(
            "Number of rejected points:         ", len(np.invert(b[np.invert(b)]))
        )
        print()

    return f, tmp_results, b

def read_fits(CName, plot=False):
    
    hdu_l = fits.open(f'../GaiaESO_Merle_data/spectra_casu/uvl_{CName}_580.0.fits', memmap=False)
    hdu_u = fits.open(f'../GaiaESO_Merle_data/spectra_casu/uvu_{CName}_580.0.fits', memmap=False)
    
    data_l = hdu_l['normalised_spectrum'].data
    cont_l = hdu_l['continuum'].data
    crval1_l = hdu_l['normalised_spectrum'].header['CRVAL1']
    cdelt1_l = hdu_l['normalised_spectrum'].header['CDELT1']
    input_spectra_l = hdu_l['input_spectra'].data
    input_spectra_metadata_l = hdu_l['Inputinfo'].data
    Nspec = len(input_spectra_l)

    data_u = hdu_u['normalised_spectrum'].data
    cont_u = hdu_u['continuum'].data
    crval1_u = hdu_u['normalised_spectrum'].header['CRVAL1']
    cdelt1_u = hdu_u['normalised_spectrum'].header['CDELT1']
    input_spectra_u = hdu_u['input_spectra'].data
    input_spectra_metadata_u = hdu_u['Inputinfo'].data

    if Nspec != len(input_spectra_u):
        print("GREAT PROBLEM!!!!")

    x_l = np.linspace(crval1_l, crval1_l+len(data_l)*cdelt1_l, num=len(data_l), endpoint=True)
    x_u = np.linspace(crval1_u, crval1_u+len(data_u)*cdelt1_u, num=len(data_u), endpoint=True)

    #print(CName)
    #for i in range(Nspec):
    #    if len(input_spectra_l[i][~np.isfinite(input_spectra_l[i])]) > 0:
    #        print("AAAAAAAAAAAAAA")
    #    if len(input_spectra_u[i][~np.isfinite(input_spectra_u[i])]) > 0:
    #        print("AAAAAAAAAAAAAA")
        
    # Group individual exposures by obsdate
    grouped_spectra_l_final = []
    grouped_spectra_u_final = []

    grouped_spectra_l = [input_spectra_l[0]]
    grouped_spectra_u = [input_spectra_u[0]]
    dateobs = [f"{CName}_{input_spectra_metadata_l[0]['DATE_OBS']}"]
    Nstack = []

    n = 100
    deg = 7
    su = 3.5
    sl = 2.5

    for i in range(Nspec):
        if i == 0:
            continue
        if (input_spectra_metadata_l[i]['MJD_OBS'] - input_spectra_metadata_l[i-1]['MJD_OBS']) > 0.05:
            Nstack.append(len(grouped_spectra_l))
            sum_l = np.sum(np.array(grouped_spectra_l), axis=0) / cont_l
            sum_u = np.sum(np.array(grouped_spectra_u), axis=0) / cont_u
            result_l = sclip((x_l, sum_l), chebyshev, n=n, deg=deg, su=su, sl=sl, min_data=1, verbose=True)
            result_u = sclip((x_u, sum_u), chebyshev, n=n, deg=deg, su=su, sl=sl, min_data=1, verbose=True)
            grouped_spectra_l_final.append(sum_l/result_l[0])
            grouped_spectra_u_final.append(sum_u/result_u[0])
            dateobs.append(f"{CName}_{input_spectra_metadata_l[i]['DATE_OBS']}")
            grouped_spectra_l = [input_spectra_l[i]]
            grouped_spectra_u = [input_spectra_u[i]]

        else:
            grouped_spectra_l.append(input_spectra_l[i])
            grouped_spectra_u.append(input_spectra_u[i])
    
    Nstack.append(len(grouped_spectra_l))
    sum_l = np.sum(np.array(grouped_spectra_l), axis=0) / cont_l
    sum_u = np.sum(np.array(grouped_spectra_u), axis=0) / cont_u
    result_l = sclip((x_l, sum_l), chebyshev, n=n, deg=deg, su=su, sl=sl, min_data=1, verbose=True)
    result_u = sclip((x_u, sum_u), chebyshev, n=n, deg=deg, su=su, sl=sl, min_data=1, verbose=True)
    grouped_spectra_l_final.append(sum_l/result_l[0])
    grouped_spectra_u_final.append(sum_u/result_u[0])
    print(input_spectra_metadata_l['MJD_OBS'])
    print(np.std(input_spectra_metadata_l['MJD_OBS']))

    if plot:
        plt.plot(x_l, data_l, label=f'all {Nspec} stacked')
        plt.plot(x_u, data_u, label=f'all {Nspec} stacked')       

        for i in range(len(grouped_spectra_l_final)):
            plt.plot(x_l, grouped_spectra_l_final[i], lw=1, label=f'{Nstack[i]} stacked {dateobs[i]}')
            plt.plot(x_u, grouped_spectra_u_final[i], lw=1, label=f'{Nstack[i]} stacked {dateobs[i]}')
            #plt.plot(x, np.sum(input_spectra, axis=0)/cont)
        plt.legend()
        plt.show()
    
    
    return x_l, x_u, grouped_spectra_l_final, grouped_spectra_u_final, dateobs
    
def turbo(teff,logg,met,vmic,lmin,lmax,FWHM,resolution,abond):
    
    #teff = 5500
    #logg = 4.0
    #met = -1.0
    #vmic = 1.0
    
    #lmin = 5100
    #lmax = 5200
    #6500-6600 first 
    #5100-5200 second
    # compare
    stellar parameters in 6500-6600
    second step: 

    ldelta = 0.01 # wavelength step in angstroms
    
    atmosphere_type = "1D"   # "1D" or "3D"
    nlte_flag = False
    
    if abond != False:
        # only standard stellar parameters
        elements_in_nlte = []  # can choose several elements, used ONLY if nlte_flag = True
        element_abundances = {"Ni": abond["Ni"], "Mg": abond["Mg"]}  # elemental abundances [X/Fe]; if not written solar scaled ones are used
        include_molecules = False  # way faster without them
    else:
        elements_in_nlte = ["Fe", "Mg"]  # can choose several elements, used ONLY if nlte_flag = True
        element_abundances = {}  # elemental abundances [X/Fe]; if not written solar scaled ones are used
        include_molecules = False  # way faster without them
    
    # plots the data, but can also save it for later use
    wavelength, flux = plot_synthetic_data(turbospectrum_paths, teff, logg, met, vmic, lmin, lmax, ldelta, atmosphere_type, nlte_flag, elements_in_nlte, element_abundances, include_molecules, resolution=resolution, macro=0, rotation=0, verbose=False)

    #convolution

    #FHWM (0.12A = 12 pixels) = 2.354 * sigma = 2.354 * 5.09 pixels
    #FWHM= 0.12
    
    pix = FWHM/ldelta

    sig = pix/2.354

    z = gaussian_filter1d(flux, sigma=sig)
    
    return wavelength, z

def make_synSB(x_l, x_u, teff1, logg1, met1, vmic1, FWHM1, teff2, logg2, met2, vmic2, FWHM2, RV1, RV2, ratios, abond, individual=False):
    
    y_arr = []
    sint1a_arr = []
    sint2a_arr = []
    
    #print(teff1, logg1, met1, vmic1, FWHM1, teff2, logg2, met2, vmic2, FWHM2, RV1, RV2, ratios, abond)

    for i, j in enumerate([x_l, x_u]):
        try:
            #making the syn A spectrum     
            template_wavelength_1, template_flux_1 = turbo(teff1,logg1,met1,vmic1,np.min(j),np.max(j),FWHM1,0,abond)
            
            #making the syn B spectrum     
            template_wavelength_2, template_flux_2 = turbo(teff2,logg2,met2,vmic2,np.min(j),np.max(j),FWHM2,0,abond)

            #doppler shift
            template_wavelength_1_rv = doppler_shift(template_wavelength_1, RV1)
            template_wavelength_2_rv = doppler_shift(template_wavelength_2, RV2)

            sint1a = 1./(1.+ratios[i]) * np.interp(j, template_wavelength_1_rv, template_flux_1, 1, 1)
            sint2a = 1./(1.+(1./ratios[i])) * np.interp(j, template_wavelength_2_rv, template_flux_2, 1, 1)

            sint1a_arr.append(sint1a)
            sint2a_arr.append(sint2a)

            y_arr.append(sint1a + sint2a)
        except:
            y_arr.append(np.ones(len(j)))
            sint1a_arr.append(np.ones(len(j)))
            sint2a_arr.append(np.ones(len(j)))
            

    if individual:
        return sint1a_arr, sint2a_arr
    else:
        return y_arr

def chi_squared(x, x_l, x_u, y_l, y_u, params=False):
    #x_l, x_u, y_l, y_u = data


    if False:
        param_names = [i for i in params]
        x = dict(zip(param_names, x))

        if not np.all([params[i].min < x[i] < params[i].max for i in param_names]):
            return np.ones(len(x_l)+len(x_u))
    else:
        syb_SB = make_synSB(x_l, x_u,
            x['teff_a'].value,x['logg_a'].value,x['met'].value,x['vmic_a'].value,x['FWHM_a'].value,
            x['teff_b'].value,x['logg_b'].value,x['met'].value,x['vmic_b'].value,x['FWHM_b'].value,
            x['RV_a'].value,x['RV_b'].value, [x['ratio1'].value, x['ratio2'].value], {'Ni': x['Ni'].value, 'Mg': x['Mg'].value}
        )
    #print(x)
    #print(x_l, syb_SB[0])
    #plt.plot(x_l, y_l)
    #plt.plot(x_l, syb_SB[0])
    #plt.plot(x_u, y_u)
    #plt.plot(x_u, syb_SB[1])
    #plt.show()

    dif1 = list(y_l - syb_SB[0])
    dif2 = list(y_u - syb_SB[1])
    
    return np.concatenate((dif1, dif2))
    #print(-0.5 * np.sum(np.concatenate((dif1, dif2))**2))
    #return -0.5 * np.sum(np.concatenate((dif1, dif2)**2))


  
def convergence_achieved(state, nwalkers, span, mcmc_files='', id='', ndim='', aut_force_walk=False):
  try:
    samples_all = np.genfromtxt(state, skip_header=1)
  except:
    print_exception()

  samples = samples_all[:, 1:]

  span = span-50
  first_chunk = np.array([samples_all[i::nwalkers][-span:-span+15, -1] for i in range(nwalkers)]).flatten()
  last_chunk = np.array([samples_all[i::nwalkers][-15:, -1] for i in range(nwalkers)]).flatten()
  last_step = np.array([samples_all[i::nwalkers][-1, -1] for i in range(nwalkers)]).flatten()

  first_chunk = first_chunk[~(np.isnan(first_chunk) | np.isinf(first_chunk))]
  last_chunk = last_chunk[~(np.isnan(last_chunk) | np.isinf(last_chunk))]
  last_step = last_step[~(np.isnan(last_step) | np.isinf(last_step))]

  upp = last_step[last_step >= np.percentile(last_step, 90)]
  dow = last_step[last_step <= np.percentile(last_step, 10)]
  
  slope = (abs(np.nanmedian(first_chunk) - np.nanmedian(last_chunk)) > (0.5 * np.std(last_chunk)))
  branching = 20*np.nanmean([np.nanstd(upp), np.nanstd(dow)]) < abs(np.nanmedian(dow) - np.nanmedian(upp))

  print("conv crit: ", abs(np.median(first_chunk) - np.median(last_chunk)), np.std(last_chunk))
  # prevent split (although converged) branches
  print("average std, split diff: ", 20*np.average([np.std(upp), np.std(dow)]), abs(np.median(dow) - np.median(upp)))
  
  if aut_force_walk:
    if branching and \
        np.isfinite(20*np.nanmean([np.nanstd(upp), np.nanstd(dow)])) and \
        np.isfinite(abs(np.nanmedian(dow) - np.nanmedian(upp))):
      force_walkers(mcmc_files, id, nwalkers, ndim, 75, aut_force_walk=aut_force_walk)

  # Check slope of convergence and multiple branches
  if branching or slope:
    return False
  else:
    return True
    
def force_walkers(mcmc_files, id, nwalkers, ndim, percentile, nwalkers_out=64, aut_force_walk=False):
  # ========= FORCE WALKERS TO LOWEST BRANCH IN THE MIDDLE OF ITERATIONS

  if nwalkers_out == 1024:
    id += '_spec_prior'
  try:
    samples_all = np.genfromtxt('%s/%s.mcmc' % (mcmc_files, id), skip_header=1)
  except:
    print_exception()
    
    

  samples = samples_all[:, 1:]
  lnprob_lim_high = np.inf

  all_w = np.array([samples_all[i::nwalkers][-1:, -1] for i in range(nwalkers)]).flatten()
  all_w = all_w[~(np.isnan(all_w) | np.isinf(all_w))]
  perc_up10 = np.nanpercentile(all_w, percentile)
  print("forcing ", id, " to ", percentile, 'th percentile: ', perc_up10)
  lnprob_lim_low = perc_up10
  
  samples_by_walkers = []

  for i in range(nwalkers):
    walkers_burnin_cut = samples_all[i::nwalkers][-1:, -1]       
    
    #if len(walkers_burnin_cut) == len(walkers_burnin_cut[(walkers_burnin_cut >= lnprob_lim_low) & (walkers_burnin_cut <= lnprob_lim_high)]):
    if walkers_burnin_cut > lnprob_lim_low:
      samples_by_walkers.append(samples[i::nwalkers][-1:])  
  
  samples = np.concatenate(np.transpose(np.array(samples_by_walkers), (1,0,2)))  
  
    
  if False:#aut_force_walk:
    samples = samples[:, 0:ndim+1]  
    #print samples
    sigmas = np.std(samples, axis=0)
    medians = np.median(samples, axis=0)
    
    cor = np.zeros((ndim+1, ndim+1))
    for i in range(0, ndim+1):
      for j in range(0, ndim+1):
        prs = st.pearsonr(samples[:, i], samples[:, j])[0]
        cor[i][j] = prs * sigmas[i] * sigmas[j]          

    smpls = np.random.multivariate_normal(medians, cor, nwalkers_out)
  else:
    # DIFFERENT FORCING FOR RADII, JUST DUPLICATE BEST WALKERS
    #samples = samples[:, 0:ndim+1]  
    samples = sorted(samples, key=lambda x: x[-1], reverse=True)
    smpls = []
    #print len(samples), nwalkers_out
    #samples = list(samples)
    #lensamp = len(samples)
    if nwalkers_out > len(samples):
      samples = samples * int(np.ceil(nwalkers_out/len(samples))+2)
    
    #count = 1
    for i in range(nwalkers_out): 
      neww = samples.pop(0)
      #print count, neww
      #if count > lensamp:
      #  #print "vecji", count
      #  neww *= (1.+0.001*(random.random()-0.5))
      #print count, neww
      smpls.append(neww)
      
      #count += 1




  f = open('%s/%s.mcmc' % (mcmc_files, id), 'a')

  for i, sm in enumerate(smpls):
    f.write("%d " % (i))
    for j in sm:
      f.write('%.5f ' % j)
    f.write('\n')
  f.close()
  # ============================================

def mcmc_fit(spec_name, Nwalkers, niter, p0, params, data):
  
    ndim = len(params)

    pool = Pool(Nwalkers)
    sampler = emcee.EnsembleSampler(Nwalkers, ndim, chi_squared, args=[data, params], pool=pool)

    print("start mcmc for %s" % spec_name)



    count = 0
    f = open('%s.mcmc' % (spec_name), "w")
  
    for result in sampler.sample(p0, iterations=niter):        
        #position = result[0]
        #computed = result[3]
        count += 1    
        #print count
        #print np.average(sampler.acceptance_fraction)
        #print count
        #if count % 100 == 0:
        #  try:
        #    print sampler.acor
        #  except:
        #    pass

        position, log_prob, random_state, blobs = result
        #print blobs

        for k in range(position.shape[0]):
          
          #position[k][-4] = 10**position[k][-4]
          #position[k][-3] = 10**position[k][-3]
          f.write("%d %s %f\n" % (k, # walker #
                                  " ".join(['%.5f' % i for i in position[k]]), # adjpars
                                  log_prob[k] # lnprob value
                                  )
          )
          
    f.close()

    return True

def fit_one_spectrum(CName, return_dict):
    x_l, x_u, grouped_spectra_l_final, grouped_spectra_u_final, dateobs = read_fits(CName)
    
    merle_results = Table.read('../final_gaiaESO_SB2.fits')
    merle = merle_results[merle_results['CName'] == CName]
    


    params = Parameters()

    params.add('RV_a', float(merle['RV1']), min=-250, max=250, vary=True)
    params.add('teff_a', 5770, min=3500, max=7000, vary=True)
    params.add('logg_a', 4, min=0, max=5.5, vary=True)
    params.add('met', 0, min=-3, max=1, vary=True)
    params.add('vmic_a', 1, min=0, max=2, vary=True)
    params.add('FWHM_a', 0.1, min=0, max=1, vary=True)

    params.add('RV_b', float(merle['RV2']), min=-250, max=250, vary=True)
    params.add('teff_b', 5770, min=3500, max=7000, vary=True)
    params.add('logg_b', 4, min=0, max=5.5, vary=True)
    #params.add('met_b', float(pars['feh_50']), vary=True)
    params.add('vmic_b', 1, min=0, max=2, vary=True)
    params.add('FWHM_b', 0.1, min=0, max=1, vary=True)

    params.add('ratio1', 1, min=0, max=1, vary=True)
    params.add('ratio2', 1, min=0, max=1, vary=True)
    
    params.add('Ni', 0.0, min = -3.0, max = 2.0, vary=True)
    params.add('Mg', 0.0, min = -3.0, max = 2.0, vary=True)

    Nwalkers = 100
    niter = 10000
    init_pos = np.zeros((Nwalkers, len(params)))

    for i in range(Nwalkers):
        for j, k in enumerate(params):
            init_pos[i, j] = np.random.uniform(low=params[k].min, high=params[k].max, size=1)

    for i in range(len(grouped_spectra_l_final)):
        mask_l = np.isfinite(grouped_spectra_l_final[i]) & (x_l > LEFT_LIM) & (x_l < RIGHT_LIM)
        mask_u = np.isfinite(grouped_spectra_u_final[i]) & (x_u > LEFT_LIM) & (x_u < RIGHT_LIM)

        if False:
            mcmc_fit(dateobs[i], Nwalkers, niter, init_pos, params, (x_l[mask_l], x_u[mask_u], grouped_spectra_l_final[i][mask_l], grouped_spectra_u_final[i][mask_u]))
            
            '''
            force_walkers(mcmc_files, id, nwalkers, npars, 75, aut_force_walk=True)
            
            repeats = 15

            for convit in range(repeats):
                # ================== If convergence not achieved, try a little bit more
                conv = convergence_achieved('%s/%s.mcmc'%(mcmc_files, id), 64, midspan, mcmc_files, id, npars, aut_force_walk=True)

                if not conv: 
                  mcmc_fit(njob, midspan, 64, mcmc_files, spec1, '%s/%s.mcmc'%(mcmc_files, id), extinction)

                else:
                  break

                print "mem 5"
                mem()
                # =====================================================================
            '''

            '''
            mini = lmfit.Minimizer(chi_squared, params, fcn_args=(x_l[mask_l], x_u[mask_u], grouped_spectra_l_final[i][mask_l], grouped_spectra_u_final[i][mask_u]))
            with Pool(Nwalkers) as pool:
                result = tqdm(mini.emcee(nwalkers=Nwalkers, steps=1, workers=pool, pos=init_pos, reuse_sampler=False, float_behavior='posterior', progress=True))
            '''
        else:

            result = minimize(chi_squared, params, method='nelder', args=(x_l[mask_l], x_u[mask_u], grouped_spectra_l_final[i][mask_l], grouped_spectra_u_final[i][mask_u]))
            print(result.params)
            return_dict[dateobs[i]] = result
        


fitted_pars = ['RV_a', 'teff_a', 'logg_a', 'vmic_a', 'FWHM_a', 'met', 'RV_b', 'teff_b', 'logg_b', 'vmic_b', 'FWHM_b', 'ratio1', 'ratio2', 'Ni', 'Mg']
#fitted_pars = ['RV_a', 'teff_a', 'logg_a', 'vmic_a', 'FWHM_a', 'met', 'RV_b', 'teff_b', 'logg_b', 'vmic_b', 'FWHM_b', 'ratio1', 'ratio2']

listb = '''00301156-5001500
00324599-4354509
01592290-4658510
03201610-5601321
03401027+0002559
04202910-0019338
04301327-5001191
05402480-4726342
05562593-6029184
08231542-0535165
08233762-0536506
10092718-4128583
10224640-3541044
10232266-3541019
12000916-4101004
12005511-3711201
12273877-4056402
14222902-4402086
15095773-2000080
18103653-4455176
18135851-4226346
19000942-4231227
20192137-4706271
21101784-0205349
21402535-0055041
22593725-0052333
23501961-5012563
17571482-4147030
11085326-7519374
08072516-4712522
08073722-4705053
08093589-4718525
08103996-4714428
08115305-4654115
08385566-5257516
08393881-5310071
10460575-6420184
17453692+0542424
08511868+1147026
08511901+1150056
08512291+1148493
08512940+1154139
06404608+0949173
06413150+0954548
06413207+1001049
06414775+0952023
07401559-3735416
07405697-3721458
07455390-3812406
07593671-6021483
07594121-6109251
08081564-4908244
11085927-5849560
18281038+0647407
19105940-5957059'''
listb = listb.split()

if False:
    for i in listb:
        x_l, x_u, grouped_spectra_l_final, grouped_spectra_u_final, dateobs = read_fits(i, plot=True)



if True:
    manager = mp.Manager()
    return_dict = manager.dict()
    ps = []

    for k, i in enumerate(listb):
      p = multiprocessing.Process(target=fit_one_spectrum, args=(i, return_dict))
      ps.append(p)
      p.start()

    for p in ps:
      p.join()


    result = dict(return_dict)

    print(result)



    with open('results_gaiaESO_Ni_Mg.csv', 'w') as f:
        f.write('CNAMEdateobs,%s\n' % (','.join([k for k in fitted_pars])))

        for i, j in result.items():
            print(i, j)
            f.write('%s,%s\n' % (i, ','.join([str(j.params[k].value) for k in fitted_pars])))


if True:
    results = Table.read('results_gaiaESO_Ni_Mg.csv')
    merle_results = Table.read('../final_gaiaESO_SB2.fits')
    #results.rename_column('#CNAMEdateobs', 'CNAMEdateobs')
    print(results)

    for j, i in enumerate(results):

        x_l, x_u, grouped_spectra_l_final, grouped_spectra_u_final, dateobs = read_fits(i['CNAMEdateobs'].split('_')[0])

        for m, k in enumerate(dateobs):
            if k != i['CNAMEdateobs']:
                continue
            
            CName = i['CNAMEdateobs'].split('_')[0]
            print(f"plotting CName: {CName}")
            merle = merle_results[merle_results['CName'] == CName]
            print(merle)

            mask_l = np.isfinite(grouped_spectra_l_final[m]) & (x_l > LEFT_LIM) & (x_l < RIGHT_LIM)
            mask_u = np.isfinite(grouped_spectra_u_final[m]) & (x_u > LEFT_LIM) & (x_u < RIGHT_LIM)
            x_l = x_l[mask_l]
            x_u = x_u[mask_u]

            sint1a, sint2a = make_synSB(x_l, x_u, 
                i['teff_a'],i['logg_a'],i['met'],i['vmic_a'],i['FWHM_a'],
                i['teff_b'],i['logg_b'],i['met'],i['vmic_b'],i['FWHM_b'],
                i['RV_a'],i['RV_b'], [i['ratio1'], i['ratio2']], {'Ni': i['Ni'], 'Mg': i['Mg']},
                #i['RV_a'], i['RV_b'], [i['ratio1'], i['ratio2']], False,
                individual=True
            )
            

            fig2, ax = plt.subplots(nrows=2, ncols=1, figsize=(40, 15))
            
            ax[0].plot(x_l, grouped_spectra_l_final[m][mask_l] - (sint1a[0]+sint2a[0]), c='red', lw=1, label='TURBOSPECTRUM')
            ax[0].plot(x_u, grouped_spectra_u_final[m][mask_u] - (sint1a[1]+sint2a[1]), c='red', lw=1, label='TURBOSPECTRUM')
            #ax[0].set_xlim([5100, 5200])
            ax[0].axhline(0.0, c='black', lw=0.5)
            ax[0].set_ylim([-0.2,0.2])
            ax[0].set_ylabel('O-C')
            ax[0].legend()

            ax[1].plot(x_l, grouped_spectra_l_final[m][mask_l], c='black', lw=2, label='observed')
            ax[1].plot(x_u, grouped_spectra_u_final[m][mask_u], c='black', lw=2)

            ax[1].plot(x_l, sint1a[0], c='green', lw=1, label='primary')
            ax[1].plot(x_u, sint1a[1], c='green', lw=1)
            ax[1].plot(x_l, sint2a[0], c='blue', lw=1, label='secondary')
            ax[1].plot(x_u, sint2a[1], c='blue', lw=1)
            
            ax[1].plot(x_l, sint1a[0]+sint2a[0], c='red', lw=1, label='primary+secondary')
            ax[1].plot(x_u, sint1a[1]+sint2a[1], c='red', lw=1)
            #ax[1].set_xlim([5100, 5200])
            ax[1].set_ylim([0,1.3])
            ax[1].legend()

            fig2.suptitle(f"{i['CNAMEdateobs']}\nTURBOSPECTRUM: {', '.join([str(k)+': '+'%.2f'%i[k] for k in fitted_pars])}\nMerle: RV1: {float(merle['RV1'])}, RV2: {float(merle['RV2'])}")
            #niceplot(fig2)
            fig2.savefig(f"gaiaESO_{i['CNAMEdateobs']}.png")
            plt.close(fig2)

