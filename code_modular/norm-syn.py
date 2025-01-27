#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:38:17 2024

@author: heitor
"""

#turbospectrum
from __future__ import annotations
try:
    from scripts_for_plotting import *
except ModuleNotFoundError:
    import sys
    sys.path.append('/Users/heitor/Desktop/NLTE-code/TSFitPy/')
    from scripts_for_plotting import *
 
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

    
import seaborn as sns
import csv
import sys

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from numpy.polynomial.chebyshev import Chebyshev
from scipy.ndimage import gaussian_filter1d


from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


sns.set_style("white")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})



turbospectrum_paths = {"turbospec_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/turbospectrum/exec-gf/",  # change to /exec-gf/ if gnu compiler
                       "interpol_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/scripts/model_interpolators/",
                       "model_atom_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/model_atoms/",
                       "departure_file_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/",
                       "model_atmosphere_grid_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/model_atmospheres/",
                       "line_list_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/linelists/linelist_for_fitting_MY/"}



def turbo(teff,logg,met,vmic,lmin,lmax,FWHM):
    
    print('=====================================================================')
    
    #teff = 5500
    #logg = 4.0
    #met = -1.0
    #vmic = 1.0
    
    #lmin = 4600
    #lmax = 5500

    ldelta = 0.01
    
    atmosphere_type = "1D"   # "1D" or "3D"
    nlte_flag = True
    
    elements_in_nlte = ["H"]  # can choose several elements, used ONLY if nlte_flag = True
    element_abundances = {"Mg": 0.0, "O": 0.0}  # elemental abundances [X/Fe]; if not written solar scaled ones are used
    include_molecules = False  # way faster without them
    
    # plots the data, but can also save it for later use
    wavelength, flux = plot_synthetic_data(turbospectrum_paths, teff, logg, met, vmic, lmin, lmax, ldelta, atmosphere_type, nlte_flag, elements_in_nlte, element_abundances, include_molecules, resolution=0, macro=0, rotation=0, verbose=False)

    #convolution

    #FHWM (0.12A = 12 pixels) = 2.354 * sigma = 2.354 * 5.09 pixels
    #FWHM= 0.12
    
    pix = FWHM/ldelta

    sig= pix/2.354

    z = gaussian_filter1d(flux, sigma=sig)
    
    print('=====================================================================')

    
    return wavelength, z




def norm_syn(observed_wave, observed_flux, template_wave, template_flux, deg, cont_cut):
        

    #first normalizations
    # Step 1: Bin the template spectrum to match the observed spectrum's wavelength grid
    # This interpolates the template_flux to the observed_wave grid
    template_flux_binned = np.interp(observed_wave, template_wave, template_flux)

    # Step 2: Identify the continuum regions
    # We assume continuum regions are where the template flux is close to 1.0 (continuum level)
    continuum_threshold = cont_cut  # Set a threshold around 1 (small deviation for continuum)
    continuum_indices = np.where(np.abs(template_flux_binned) > continuum_threshold)[0]


    # Extract the continuum points from the observed spectrum
    observed_continuum_wave = observed_wave[continuum_indices]
    observed_continuum_flux = observed_flux[continuum_indices]

    # Step 3: Fit a Chebyshev polynomial to the continuum regions
    # Fit a Chebyshev polynomial to the continuum points (degree 3 by default)
    degree =  deg  # You can adjust the degree as needed
    cheby_poly = Chebyshev.fit(observed_continuum_wave, observed_continuum_flux, degree)

    # Evaluate the Chebyshev polynomial over the entire wavelength range of the observed spectrum
    continuum_fit = cheby_poly(observed_wave)

    # Step 4: Normalize the observed spectrum by the fitted continuum
    normalized_flux = observed_flux / continuum_fit
    
    # tep 5:
    #second normalization where we define the continuum points but cut if it is lower than 0.9999
    
    continuum_indices_2 = np.where( (normalized_flux > 0.99) & (np.abs(template_flux_binned) > continuum_threshold) )[0]

    # Extract the continuum points from the observed spectrum
    observed_continuum_wave_2 = observed_wave[continuum_indices_2]
    observed_continuum_flux_2 = normalized_flux[continuum_indices_2]
    
    cheby_poly_2 = Chebyshev.fit(observed_continuum_wave_2, observed_continuum_flux_2, degree)
    
    continuum_fit_2 = cheby_poly_2(observed_wave)
    
    normalized_flux_2 = normalized_flux / continuum_fit_2
    

    
    return [observed_continuum_wave,  observed_continuum_flux, continuum_fit, normalized_flux, observed_continuum_wave_2, observed_continuum_flux_2, continuum_fit_2, normalized_flux_2]



#------------#------------#------------
# Step 1: Read the observed and template spectra from CSV files

debug = 0

#paths and names

path='../out/coadded/'

in_file = 'HD-106516_564u_summed_spectrum.csv'

#in_file = sys.argv[1]

#'HD-106516_390_summed_spectrum.csv'
#'HD-106516_564l_summed_spectrum.csv'

star_file = in_file[:-4]

output_plot = f"{star_file}-norm_plot.pdf"



print('Observed read')

observed_df = pd.read_csv(path+in_file)
observed_wave = observed_df['wavelength'].values
observed_flux = observed_df['flux'].values


observed_wave = observed_df['wavelength'].values


observed_flux = observed_df['flux'].values


if debug == 1:
    
    plt.plot(observed_wave,observed_flux)
    
 
#load template:
    
#template_df = pd.read_csv('template_spectrum.csv')
#template_wave = template_df['wave'].values
#template_flux = template_df['flux'].values


#Parameters for the normalization
cont_cut=0.9992
deg=3

#------------
#turbo standard parameters
teff = 5500
logg = 4.0
met = -1.0

vmic = 1.0

FWHM= 0.20

#-----------

print('Creating Template')


#390 p1
if max(observed_wave) < 4600 and min(observed_wave) > 3100:   
    print('It is the setup 390 you should use the other code for that')

    
#564l
if max(observed_wave) < 5800 and min(observed_wave) > 4500 :
    lmin = 4625
    lmax = 5595
    
    #make template
    template_wave, template_flux = turbo(teff,logg,met,vmic,lmin,lmax,FWHM)
    
    
#564u
if max(observed_wave) < 6800 and min(observed_wave) > 4800 :
    lmin = 5680
    lmax = 6670
    
    #make template
    template_wave, template_flux = turbo(teff,logg,met,vmic,lmin,lmax,FWHM)




#-----------

#-----quick check-------

if debug == 1:
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(observed_wave, observed_flux/np.mean(observed_flux), label='Observed-spec', linestyle='dashed', color='k')
    ax.plot(template_wave, template_flux, label='Syn', color='red', linewidth=1.5)

    ax.legend()
    ax.set_xlabel('Wavelength (Angstroms)')
    ax.set_ylabel('Intensity')
    
#------------
print('Normalization...')
#prenormalization and remove the zeros from flux
observed_wave, observed_flux = observed_wave[observed_flux != 0], observed_flux[observed_flux != 0]
observed_flux = observed_flux / np.median(observed_flux)



#NORMALIZATION Steps 1-5
norm_pars_out = norm_syn(observed_wave, observed_flux, template_wave, template_flux, deg, cont_cut)

observed_continuum_wave_1,  observed_continuum_flux_1, continuum_fit_1, normalized_flux_1, observed_continuum_wave, observed_continuum_flux, continuum_fit, normalized_flux = norm_pars_out


#-----------
print('Plotting')
# Step 7: Plot the results
# Create subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0})  # hspace=0 removes vertical space

# Plot in each subplot


axs[0].plot(observed_wave, observed_flux, label='Observed Spectrum', color='black', alpha=0.8)
axs[0].plot(template_wave, template_flux, label='Template Spectrum', color='blue', alpha=0.6)


axs[1].plot(observed_wave, observed_flux, label='Observed Spectrum', color='black', alpha=0.4)
axs[1].scatter(observed_continuum_wave_1, observed_continuum_flux_1, color='red', label='Continuum Points')
axs[1].plot(observed_wave, continuum_fit_1, label='Fitted Continuum', color='orange')


axs[2].plot(observed_wave, normalized_flux_1, label='Observed Spectrum', color='black', alpha=0.4)
axs[2].scatter(observed_continuum_wave, observed_continuum_flux, color='red', label='Continuum Points')
axs[2].plot(observed_wave, continuum_fit, label='Fitted Continuum', color='orange')



axs[3].plot(observed_wave, normalized_flux, label='Normalized Spectrum', color='green')
axs[3].plot(template_wave, template_flux, label='Template Spectrum', color='blue', alpha=0.4)



# Add tick markers and label in the corner for each panel
for ax in axs:
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True, direction='in')

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    
    
    ax.legend(loc='lower right')  # Legend in upper right corner of each subplot
    
    
    

# Label for x and y
axs[-1].set_xlabel('Wavelength')
axs[0].set_ylabel('Flux')

plt.tight_layout()
plt.show()

plt.savefig('../fig/norm/'+output_plot)



#-----------
# Step 8: Save the normalized spectrum to a new CSV file
normalized_df = pd.DataFrame({'wave': observed_wave, 'flux': normalized_flux})
normalized_df.to_csv('../out/norm/'+star_file+'_normalized_spectrum.csv', index=False)

























#