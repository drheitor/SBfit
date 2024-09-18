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
    sys.path.append('/Users/heitor/Desktop/NLTE-code/TSFitPy/')
    from scripts_for_plotting import *
    


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits

from lmfit import Model
from lmfit import minimize, Parameters


from scipy.interpolate import interp1d
#from scipy.optimize import minimize
#from scipy.optimize import minimize, differential_evolution



from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
sns.set_style("white")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})



turbospectrum_paths = {"turbospec_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/turbospectrum/exec-gf/",  # change to /exec-gf/ if gnu compiler
                       "interpol_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/scripts/model_interpolators/",
                       "model_atom_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/model_atoms/",
                       "departure_file_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/nlte_data/",
                       "model_atmosphere_grid_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/model_atmospheres/",
                       "line_list_path": "/Users/heitor/Desktop/NLTE-code/TSFitPy/input_files/linelists/linelist_for_fitting/"}



#----------------------------------------------

# Function to correct radial velocity
def doppler_shift(wavelength, velocity):
    c = 299792.458  # speed of light in km/s
    return wavelength * np.sqrt((1 + velocity / c) / (1 - velocity / c))



def read_fits(name):
    
    sobject_id = name

    spec1 = '%s1.fits' % sobject_id
    spec2 = '%s2.fits' % sobject_id
    spec3 = '%s3.fits' % sobject_id
    spec4 = '%s4.fits' % sobject_id
      
    #RV = spec['rv_com']
    #if not RV:
        #RV = 0.0
    RV = -89.35298156738281
        
    print('Observed RV:')
    print(RV)

    folder = './specs'

    print('Loading observed spectrum:')
    print(sobject_id)
    
    #selecting one
    spec_name = spec3
                    
    hdulist = fits.open('./specs/%s' % (spec_name), memmap=False) 
    
    header = hdulist[1].header
    y = hdulist[1].data
          
    ws = float(header['CRVAL1'])
    wd = float(header['CDELT1'])       
         
    x = np.linspace(ws, ws+len(y)*wd, num=len(y), endpoint=False)    
    
    x = doppler_shift(x, -RV)         
    
    return [x,y]
    
    

def turbo(teff,logg,met,vmic,lmin,lmax,FWHM,abond):
    
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
    element_abundances = {"Li": abond, "O": 0.0}  # elemental abundances [X/Fe]; if not written solar scaled ones are used
    include_molecules = False  # way faster without them
    
    # plots the data, but can also save it for later use
    wavelength, flux = plot_synthetic_data(turbospectrum_paths, teff, logg, met, vmic, lmin, lmax, ldelta, atmosphere_type, nlte_flag, elements_in_nlte, element_abundances, include_molecules, resolution=0, macro=0, rotation=0, verbose=False)

    #convolution

    #FHWM (0.12A = 12 pixels) = 2.354 * sigma = 2.354 * 5.09 pixels
    #FWHM= 0.12
    
    pix = FWHM/ldelta

    sig= pix/2.354

    z = gaussian_filter1d(flux, sigma=sig)
    
    return wavelength, z




def make_synSB(teff1,logg1,met1,vmic1,FWHM1,abond1,teff2,logg2,met2,vmic2,FWHM2,abond2,RV,ratio):
    
    #making the syn A spectrum     
    template_wavelength_1, template_flux_1 = turbo(teff1,logg1,met1,vmic1,lmin,lmax,FWHM1,abond1)
    
    #making the syn B spectrum     
    template_wavelength_2, template_flux_2 = turbo(teff1,logg1,met1,vmic1,lmin,lmax,FWHM1,abond2)


    #doppler shift for the secondary Use the RV as the diference.
    velocity=RV
    template_wavelength_2_rv = doppler_shift(template_wavelength_2, velocity)
    
    
    #ratio = 0.67371
    #use the ratio to scale the flux 
    
    sint1a = 1./(1.+ratio) * template_flux_1
    sint2a = 1./(1.+(1./ratio)) * np.interp(template_wavelength_1, template_wavelength_2_rv, template_flux_2, 1, 1)

    sint12a = sint1a + sint2a
    
    #return structure wave_1 , flux_1 , wave_2 , flux_2 , wave_sb , flux_sb
    return [template_wavelength_1, sint1a, template_wavelength_2, sint2a, template_wavelength_1, sint12a]



def chi_squared(params, x, y):

    RV = 105
    ratio = 0.67371
    
    mask = (x > lmax) & (x < lmin)
    x = x[mask]  
    y = y[mask]
        
    syb_SB = make_synSB(params['teff_a'],params['logg_a'],params['met_a'],params['vmic_a'],params['FWHM_a'],params['abond_a']
               ,params['teff_b'],params['logg_b'],params['met_b'],params['vmic_b'],params['FWHM_b'],params['abond_b']
               ,RV,ratio)
    
    
    interp_func = interp1d(syb_SB[4], syb_SB[5], kind='linear')
    syn_template_flux = interp_func(x)
    
    chi2 = np.sum((y - syn_template_flux) ** 2)
    return chi2
    


#----------------------------------------------


#define the band
#564u
#if max(observed_wavelength) < 6800 and min(observed_wavelength) > 4800 :
 #   lmin = 5840
 #   lmax = 6000


global lmin 
global lmax 

#Li line
# 6707.7635

lmin = 6705.6
lmax = 6712.6


#-----------
#parameters rought numbers
print('Loading Template 1')

teff_a = 5171
logg_a = 4.1
met_a = -0.2

vmic_a = 1.0

FWHM_a = 0.30

abond_a= 3.00


#-----------

#parameters rought numbers
print('Loading Template 2')

teff_b = 5839
logg_b = 3.5
met_b = -0.2

vmic_b = 1.0

FWHM_b = 0.28

abond_b= 3.00


#-----------
#make template

RV = 105

ratio = 0.67371

print('making SB2 Spectrum')
[wv_a, fl_a, wv_b, fl_b, wv_sb, fl_sb] = make_synSB(teff_a,logg_a,met_a,vmic_a,FWHM_a,abond_a, teff_b,logg_b,met_b,vmic_b,FWHM_b,abond_b,RV,ratio)


#-----------
#reading data

x,y = read_fits('140117001501305')


#-----------
#parameters to fit


params = Parameters()  



params.add('teff_a', teff_a, vary=False)
params.add('logg_a', logg_a, vary=False)
params.add('met_a', met_a, vary=False)

params.add('vmic_a', vmic_a, vary=False)
params.add('FWHM_a', FWHM_a, vary=False)



params.add('teff_b', teff_b, vary=False)
params.add('logg_b', logg_b, vary=False)
params.add('met_b', met_b, vary=False)

params.add('vmic_b', vmic_b, vary=False)
params.add('FWHM_b', FWHM_b, vary=False)



params.add('abond_a', abond_a, min = -5.0, max = 10.0)
params.add('abond_b', abond_b, min = -5.0, max = 10.0)






result = minimize(chi_squared, params, method='leastsq', args=(x, y))
params = result.params







#print result.redchi
#print params['amp'].value, params['sigma'].value, params['lambdac'].value





#-----------
# Plotting the results

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(x, y,'.', color='k',label='observed')

ax.plot(wv_a,fl_a, label='synA', linewidth=2.0, color='k',alpha = 0.2)
ax.plot(wv_b,fl_b, label='synB', linewidth=2.0, color='r',alpha = 0.2)




ax.plot(wv_sb,fl_sb, label='SB-syn', linewidth=4.0, color='green')



ax.legend(loc=3)
ax.set_xlabel('Wavelength')
ax.set_ylabel('Intensity')
#ax.set_title(f"radial velocity: {best_velocity_r} km/s")


plt.savefig('./fig/sb2_test.pdf')


#-----------













































#