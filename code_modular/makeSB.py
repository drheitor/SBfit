#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:54:43 2024

@author: heitor
"""

from __future__ import annotations
try:
    from scripts_for_plotting import *
except ModuleNotFoundError:
    import sys
    sys.path.append('/Users/heitor/Desktop/NLTE-code/TSFitPy/')
    from scripts_for_plotting import *
    
    
turbospectrum_paths = {"turbospec_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/turbospectrum/exec-gf/",  # change to /exec-gf/ if gnu compiler
                       "interpol_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/scripts/model_interpolators/",
                       "model_atom_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/model_atoms/",
                       "departure_file_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/",
                       "model_atmosphere_grid_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/model_atmospheres/",
                       "line_list_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/linelists/linelist_for_fitting_MY/"}


from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Function to correct radial velocity
def doppler_shift(wavelength, velocity):  
    c = 299792.458  # speed of light in km/s  
    return wavelength * np.sqrt((1 + velocity / c) / (1 - velocity / c))

# Function to make the synthetic spectrum using turbospectrum
def turbo(teff,logg,met,vmic,lmin,lmax,FWHM,resolution,abond):
    
    #teff = 5500
    #logg = 4.0
    #met = -1.0
    #vmic = 1.0
    
    #lmin = 4600
    #lmax = 5500

    ldelta = 0.01
    
    atmosphere_type = "1D"   # "1D" or "3D"
    nlte_flag = False
    
    elements_in_nlte = ["Fe", "Mg"]  # can choose several elements, used ONLY if nlte_flag = True
    #the expected format for the abundance dict
    #element_abundances = {"Li": 0.0, "O": 0.0}  # elemental abundances [X/Fe]; if not written solar scaled ones are used
    element_abundances = abond

    include_molecules = False  # way faster without them
    
    # plots the data, but can also save it for later use
    wavelength, flux = plot_synthetic_data(turbospectrum_paths, teff, logg, met, vmic, lmin, lmax, ldelta, atmosphere_type, nlte_flag, elements_in_nlte, element_abundances, include_molecules, resolution=resolution, macro=0, rotation=0, verbose=False)

    #convolution
    #FHWM (0.12A = 12 pixels) = 2.354 * sigma = 2.354 * 5.09 pixels
    #FWHM= 0.12
    
    pix = FWHM/ldelta
    sig= pix/2.354
    z = gaussian_filter1d(flux, sigma=sig)
    
    return wavelength, z



def make_synSB(x_l, x_u,teff1,logg1,met1,vmic1,FWHM1,teff2,logg2,met2,vmic2,FWHM2,RV1,RV2,ratio, abond_a, abond_b):

    
    print('-----------------------------')
    print('Making syn SB2')

    x_arr = []
    y_arr = []
    sint1a_arr = []
    sint2a_arr = []
    
    print('-----------------------------')
    print('Running first model')
            #making the syn A spectrum     
    template_wavelength_1, template_flux_1 = turbo(teff1,logg1,met1,vmic1,x_l,x_u,FWHM1,0,abond_a)
    print('-----------------------------')
    print('Running second model')       
            #making the syn B spectrum     
    template_wavelength_2, template_flux_2 = turbo(teff2,logg2,met2,vmic2,x_l,x_u,FWHM2,0,abond_b)

            #doppler shift
    template_wavelength_1_rv = doppler_shift(template_wavelength_1, RV1)
    template_wavelength_2_rv = doppler_shift(template_wavelength_2, RV2)

            # interpolating the spectra
    sint1a = 1./(1.+ratio) * template_flux_1
    sint2a = 1./(1.+(1./ratio)) * template_flux_2


    sint1a_arr.append(sint1a)
    sint2a_arr.append(sint2a)

    print('-----------------------------')
    print('Combining the models')  
    #y_arr = sint1a + sint2a
    y_arr.append(sint1a + sint2a)
    x_arr.append(template_wavelength_1_rv)
    print('-----------------------------')   
        #return the arrays
    return [[x_arr,y_arr],[template_wavelength_1_rv, sint1a_arr], [template_wavelength_2_rv, sint2a_arr]]

    
    
# Example usage
if __name__ == "__main__": 

    plotting = True
    #Li line
    # 6707.7635
    lmin = 6450
    lmax = 6750

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


    #-----------
    #make template
    
    #radial velocities
    RV_a = 105
    RV_b = 10
    
    #mag ratio between the two components
    ratio = 0.67371

    Out = make_synSB(4000, 4500, teff_a,logg_a,met_a,vmic_a,FWHM_a,teff_b,logg_b,met_b,vmic_b,FWHM_b,RV_a,RV_b,ratio, abond_a, abond_b)

    wavelength_b = Out[0][0]
    flux_b = Out[0][1]

    if plotting==True:
        plt.plot(wavelength_b, flux_b,'.',color='k')
        plt.show()
    
    
    
    
    
    
    
    
#






 
    
