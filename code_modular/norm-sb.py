#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:23:11 2025

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

    
import seaborn as sns
from astropy.io import fits
import sys
import os


from numpy.polynomial.chebyshev import Chebyshev



from makeSB import turbo
from makeSB import make_synSB
from data_reader import read_fits_GaiaESO


sns.set_style("white")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})



turbospectrum_paths = {"turbospec_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/turbospectrum/exec-gf/",  # change to /exec-gf/ if gnu compiler
                       "interpol_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/scripts/model_interpolators/",
                       "model_atom_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/model_atoms/",
                       "departure_file_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/",
                       "model_atmosphere_grid_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/model_atmospheres/",
                       "line_list_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/linelists/linelist_for_fitting_MY/"}




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
    
    continuum_indices_2 = np.where( (normalized_flux > 0.99) & (normalized_flux < 1.15) & (np.abs(template_flux_binned) > continuum_threshold) )[0]

    # Extract the continuum points from the observed spectrum
    observed_continuum_wave_2 = observed_wave[continuum_indices_2]
    observed_continuum_flux_2 = normalized_flux[continuum_indices_2]
    
    cheby_poly_2 = Chebyshev.fit(observed_continuum_wave_2, observed_continuum_flux_2, degree)
    
    continuum_fit_2 = cheby_poly_2(observed_wave)
    
    normalized_flux_2 = normalized_flux / continuum_fit_2
    

    
    return [observed_continuum_wave,  observed_continuum_flux, continuum_fit, normalized_flux, observed_continuum_wave_2, observed_continuum_flux_2, continuum_fit_2, normalized_flux_2]



#------------#------------#------------
# Step 1: Read the observed and template spectra from CSV files


# Example usage
if __name__ == "__main__": 

    debug = 0

    #CNAME test
    CNAME = '12000916-4101004'

    #uvl_18135851-4226346_580.0.fits

    #for inpit in line
    #in_file = sys.argv[1]

    star_file = CNAME

    print('Observed read')

    x_l, x_u, grouped_spectra_l_final, grouped_spectra_u_final, dateobs = read_fits_GaiaESO(CNAME, plot=False)




    #star the normalization here
    #----------------------------------------

    #INPUT spectra
    observed_wave = x_l
    observed_flux = grouped_spectra_l_final[0]



    #--------
    #debug

    if debug == 1:
    
        plt.plot(observed_wave,observed_flux)
    
    adding_template = False

    #--------
    
    #if you want to provide a template
    #load template:
        
    if adding_template == True:
        template_df = pd.read_csv('template_spectrum.csv')
        template_wave = template_df['wave'].values
        template_flux = template_df['flux'].values
        
        #FIX IT 
    
    #--------
    
    #Parameters for the normalization
    cont_cut=0.92
    deg=3
    
    #-----------
    
    if adding_template == False:
    
        print('Creating Template')
    
    
        #580l
        if max(observed_wave) < 5900 and min(observed_wave) > 4500 :
            lmin = 4750
            lmax = 5800
        
            #-----------
            #parameters rought numbers
            print('Loading Template 1')
    
            teff_a = 5171
            logg_a = 4.1
            met_a = -0.2
            vmic_a = 1.0
            FWHM_a = 0.34
            abond_a= {"Li": 0.0, "O": 0.0} 
    
            #-----------
    
            #parameters rought numbers
            print('Loading Template 2')
    
            teff_b = 5839
            logg_b = 3.5
            met_b = -0.2
    
            vmic_b = 1.0
    
            FWHM_b = 0.34
    
    
            abond_b= {"Li": 0.0, "O": 0.0} 
        
        
            #radial velocities
            RV_a = -47.51
            RV_b = 18.99
    
            #mag ratio between the two components 
            ratio = 0.67371
        
            #make template
            Out = make_synSB(lmin,lmax, teff_a,logg_a,met_a,vmic_a,FWHM_a,teff_b,logg_b,met_b,vmic_b,FWHM_b,RV_a,RV_b,ratio, abond_a, abond_b)
    
            template_wave = Out[0][0][0]
            template_flux = Out[0][1][0]
        
        
        
            #564u
            if max(observed_wave) < 6900 and min(observed_wave) > 5600 :
                lmin = 5800
                lmax = 6830
        
    
                #-----------
                #parameters rought numbers
                print('Loading Template 1')
    
                teff_a = 5171
                logg_a = 4.1
                met_a = -0.2
                vmic_a = 1.0
                FWHM_a = 0.34
                abond_a= {"Li": 0.0, "O": 0.0} 
    
                #-----------
    
                #parameters rought numbers
                print('Loading Template 2')
    
                teff_b = 5839
                logg_b = 3.5
                met_b = -0.2
    
                vmic_b = 1.0
    
                FWHM_b = 0.34
    
                abond_b= {"Li": 0.0, "O": 0.0} 
        
        
                #radial velocities
                RV_a = 105
                RV_b = 10
    
                #mag ratio between the two components 
                ratio = 0.67371
        
                #make template
                Out = make_synSB(lmin,lmax, teff_a,logg_a,met_a,vmic_a,FWHM_a,teff_b,logg_b,met_b,vmic_b,FWHM_b,RV_a,RV_b,ratio, abond_a, abond_b)
    
                template_wave = Out[0][0]
                template_flux = Out[0][1]
    
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
    
    
    
    #NORMALIZATION Steps 1-6
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
    
        #ax.xaxis.set_major_locator(MultipleLocator(100))
        #ax.xaxis.set_minor_locator(MultipleLocator(10))
        
        #ax.yaxis.set_major_locator(MultipleLocator(0.5))
        #ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax.legend(loc='upper left')  # Legend in upper right corner of each subplot
        
        
        
    
    # Label for x and y
    axs[-1].set_xlabel('Wavelength')
    axs[0].set_ylabel('Flux')
    
    plt.tight_layout()
    #plt.show()
    
    
    #create file 
    try:
        os.system('mkdir fig_norm')
        os.system('mkdir out_norm')
    except:
        print('fig and out directory already exist')
    
    
    plt.savefig('./fig_norm/'+star_file+'_normalized_spectrum.pdf')
    
    
    #-----------
    # Step 8: Save the normalized spectrum to a new CSV file
    normalized_df = pd.DataFrame({'wave': observed_wave, 'flux': normalized_flux})
    normalized_df.to_csv('./out_norm/'+star_file+'_normalized_spectrum.csv', index=False)

    print(star_file +'Normalized...')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #