#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:02:35 2025

@author: heitor
"""

    
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits



#Gaia ESO data-reader

def read_fits_GaiaESO(CName, plot=False):
    
    hdu_l = fits.open(f'./spectra_casu/uvl_{CName}_580.0.fits', memmap=False)
    hdu_u = fits.open(f'./spectra_casu/uvu_{CName}_580.0.fits', memmap=False)
    
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

        
    # Group individual exposures by obsdate
    grouped_spectra_l_final = []
    grouped_spectra_u_final = []

    grouped_spectra_l = [input_spectra_l[0]]
    grouped_spectra_u = [input_spectra_u[0]]
    dateobs = [f"{CName}_{input_spectra_metadata_l[0]['DATE_OBS']}"]
    Nstack = []

    for i in range(Nspec):
        if i == 0:
            continue
        #MJD_OBS in days
        if (input_spectra_metadata_l[i]['MJD_OBS'] - input_spectra_metadata_l[i-1]['MJD_OBS']) > 0.05:
            Nstack.append(len(grouped_spectra_l))
            sum_l = np.sum(np.array(grouped_spectra_l), axis=0) / cont_l
            sum_u = np.sum(np.array(grouped_spectra_u), axis=0) / cont_u
            
            grouped_spectra_l_final.append(sum_l)
            grouped_spectra_u_final.append(sum_u)
            dateobs.append(f"{CName}_{input_spectra_metadata_l[i]['DATE_OBS']}")
            grouped_spectra_l = [input_spectra_l[i]]
            grouped_spectra_u = [input_spectra_u[i]]

        else:
            grouped_spectra_l.append(input_spectra_l[i])
            grouped_spectra_u.append(input_spectra_u[i])
    
    Nstack.append(len(grouped_spectra_l))
    sum_l = np.sum(np.array(grouped_spectra_l), axis=0) / cont_l
    sum_u = np.sum(np.array(grouped_spectra_u), axis=0) / cont_u

    grouped_spectra_l_final.append(sum_l)
    grouped_spectra_u_final.append(sum_u)
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





# Example usage
if __name__ == "__main__": 
    
    #CNAME test
    CNAME = '12000916-4101004'

    debug = True
    
    #example
    #uvl_18135851-4226346_580.0.fits

    #for inpit in line
    #in_file = sys.argv[1]

    star_file = CNAME

    print('Observed read')

    x_l, x_u, grouped_spectra_l_final, grouped_spectra_u_final, dateobs = read_fits_GaiaESO(CNAME, plot=False)

    #INPUT spectra
    observed_wave = x_l
    observed_flux = grouped_spectra_l_final[0]

    #--------
    #debug

    if debug == True:
        
        plt.plot(observed_wave,observed_flux)




































#