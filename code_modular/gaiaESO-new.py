#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:24:11 2024

@author: heitor
"""

#-----------------------------

#turbospectrum
from __future__ import annotations
try:
    from scripts_for_plotting import *
except ModuleNotFoundError:
    import sys
    sys.path.append('/Users/heitor/Desktop/NLTE-code/TSFitPy/')
    from scripts_for_plotting import *
    
    
#-----------------------------

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns


from astropy.io import fits
from astropy.table import Table


import multiprocessing
import multiprocessing as mp


from makeSB import make_synSB
from minimizationSB import fit_one_spectrum

#-----------------------------

#plot design
sns.set_style("white")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})


#-----------------------------  
    
turbospectrum_paths = {"turbospec_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/turbospectrum/exec-gf/",  # change to /exec-gf/ if gnu compiler
                       "interpol_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/scripts/model_interpolators/",
                       "model_atom_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/model_atoms/",
                       "departure_file_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/",
                       "model_atmosphere_grid_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/model_atmospheres/",
                       "line_list_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/linelists/linelist_for_fitting_MY/"}


#-----------------------------


#----------------------------------------------

def timer(s):
    d =  s/ (24 * 60 * 60)
    h = (d - int(d)) *24
    m = (h-int(h)) *60
    sec= (m-int(m)) *60

    print("%d:%02d:%02d:%02d" % (d, h, m, sec))


#NORMALIZATION

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


#to read the data

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
        fsfsfsdfs

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
    



#fitted_pars = ['RV_a', 'teff_a', 'logg_a', 'vmic_a', 'FWHM_a', 'met', 'RV_b', 'teff_b', 'logg_b', 'vmic_b', 'FWHM_b', 'ratio1', 'ratio2', 'Ni', 'Mg']
fitted_pars = ['RV_a', 'teff_a', 'logg_a', 'vmic_a', 'FWHM_a', 'met', 'RV_b', 'teff_b', 'logg_b', 'vmic_b', 'FWHM_b', 'ratio1', 'ratio2']


LEFT_LIM = 5100
RIGHT_LIM = 5200


# for testing
listb = '''1401170015013053'''
listb = listb.split()

suffix = f'test_leastsq_{LEFT_LIM}_{RIGHT_LIM}'


# preparing spectra and running 

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


    filename = f'results_gaiaESO_{suffix}.csv'
    with open(filename, 'w') as f:
        f.write('CNAMEdateobs,%s\n' % (','.join([k for k in fitted_pars])))

        for i, j in result.items():
            print(i, j)
            f.write('%s,%s\n' % (i, ','.join([str(j.params[k].value) for k in fitted_pars])))


    results = Table.read(filename)
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
                #i['RV_a'],i['RV_b'], [i['ratio1'], i['ratio2']], {'Ni': i['Ni'], 'Mg': i['Mg']},
                i['RV_a'], i['RV_b'], [i['ratio1'], i['ratio2']], False,
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
            fig2.savefig(f"gaiaESO_{i['CNAMEdateobs']}_{suffix}.png")
            plt.close(fig2)

