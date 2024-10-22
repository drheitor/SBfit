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
import thecannon as tc

disp, g_thetas, g_s2, g_fid, g_sca, chosen_labels, g_order = joblib.load('cannon_model_sme_TEFF_LOGG_FEH_VMIC_VSINI_AK_order2_nobins.dat')
terms = tc.vectorizer.polynomial.terminator(chosen_labels, g_order)
g_vectorizer = tc.vectorizer.polynomial.PolynomialVectorizer(chosen_labels, g_order) 

g_m1 = disp < 5000
g_m2 = (disp > 5000) & (disp < 6000)
g_m3 = (disp > 6000) & (disp < 7000)
g_m4 = disp > 7000  
sint_wav1, sint_wav2, sint_wav3, sint_wav4 = disp[g_m1], disp[g_m2], disp[g_m3], disp[g_m4]
g_s2 = [g_s2[g_m1], g_s2[g_m2], g_s2[g_m3], g_s2[g_m4]]
mask1 = (sint_wav1 > 4715.94+15) & (sint_wav1 < 4896-15)
mask2 = (sint_wav2 > 5650.06+15) & (sint_wav2 < 5868.25-15)
mask3 = (sint_wav3 > 6480.52+15) & (sint_wav3 < 6733.92-15)
mask4 = (sint_wav4 > 7693.50+15) & (sint_wav4 < 7875.55-15)
g_masks = [mask1, mask2, mask3, mask4]
g_sint_wavs = [sint_wav1, sint_wav2, sint_wav3, sint_wav4] 
print(len(g_sint_wavs[0])+len(g_sint_wavs[1])+len(g_sint_wavs[2])+len(g_sint_wavs[3]))



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

from lmfit import Model
from lmfit import minimize, Parameters


from scipy.interpolate import interp1d
#from scipy.optimize import minimize
#from scipy.optimize import minimize, differential_evolution



from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
sns.set_style("white")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})



turbospectrum_paths = {"turbospec_path": "../TSFitPy/turbospectrum/exec-gf/",  # change to /exec-gf/ if gnu compiler
                       "interpol_path": "../TSFitPy/scripts/model_interpolators/",
                       "model_atom_path": "../TSFitPy/input_files/nlte_data/model_atoms/",
                       "departure_file_path": "../TSFitPy/input_files/nlte_data/",
                       "model_atmosphere_grid_path": "../TSFitPy/input_files/model_atmospheres/",
                       "line_list_path": "../TSFitPy/input_files/linelists/linelist_for_fitting/"}


sb2_galah = Table.read('../SB2_catalogue.fits')
dr6table = Table.read('/media/storage/OWNCLOUD/GALAH/obs/reductions/Iraf_6.1/dr6.1.fits')
gaia = Table.read('/data4/travegre/Projects/Galah_binaries/GALAH_iDR3_main_191213.fits')


dic = {
    'RV_a': 'V1_50',
    'teff_a': 'teff1_50',
    'logg_a': 'logg1_50',
    'met': 'feh_50',
    'vmic_a': 'vmic1_50',
    'FWHM_a': 'vbroad1_50',
    'RV_b': 'V2_50',
    'teff_b': 'teff2_50',
    'logg_b': 'logg2_50',
    'vmic_b': 'vmic2_50',
    'FWHM_b': 'vbroad2_50',
    'ratio1': 'ratio1_50',
    'ratio2': 'ratio2_50',
    'ratio3': 'ratio3_50',
    'ratio4': 'ratio4_50'
}
inv_dic = {v: k for k, v in dic.items()}

fitted_pars = ['RV_a', 'teff_a', 'logg_a', 'vmic_a', 'FWHM_a', 'met', 'RV_b', 'teff_b', 'logg_b', 'vmic_b', 'FWHM_b', 'ratio1', 'ratio2', 'ratio3', 'ratio4']

# sobject_iraf_iDR2_180108_cannon.fits
# GALAH_iDR3_main_191213.fits

xc1 = np.linspace(4730, 4880, num=int((4880-4730)/0.04)+1, endpoint=True)
xc2 = np.linspace(5665, 5850, num=int((5850-5665)/0.05)+1, endpoint=True)
xc3 = np.linspace(6495, 6720, num=int((6720-6495)/0.06)+1, endpoint=True)
xc4 = np.linspace(7705, 7860, num=int((7860-7705)/0.07)+1, endpoint=True)

#xc1 = xc1[:10]
#xc2 = xc2[:10]
#xc3 = xc3[:10]
#xc4 = xc4[:10]

c = 299792.458  # speed of light in km/s
#----------------------------------------------



# Function to correct radial velocity
def doppler_shift(wavelength, velocity):    
    return wavelength * np.sqrt((1 + velocity / c) / (1 - velocity / c))

def timer(s):
    
    d =  s/ (24 * 60 * 60)
    h = (d - int(d)) *24
    m = (h-int(h)) *60
    sec= (m-int(m)) *60
    

    print("%d:%02d:%02d:%02d" % (d, h, m, sec))
    
def read_fits(name):
    
    sobject_id = name
    data = sb2_galah[sb2_galah['sobject_id'] == int(sobject_id)]
    data_red = dr6table[dr6table['sobject_id'] == int(sobject_id)]
    data_red_old = gaia[gaia['sobject_id'] == int(sobject_id)]
    RV = float(data_red_old['rv_guess'])
    
    if not RV:
      RV = 0.0
    
    spec1 = '%s1.fits' % sobject_id
    spec2 = '%s2.fits' % sobject_id
    spec3 = '%s3.fits' % sobject_id
    spec4 = '%s4.fits' % sobject_id
    
    #RV = data_red['rv_com']
    

    folder = sobject_id[:6]

    y_arr = []
              
    for i, par in enumerate([[xc1, spec1], [xc2, spec2], [xc3, spec3], [xc4, spec4]]):
        try:
            hdulist = fits.open('/media/storage/OWNCLOUD/GALAH/obs/reductions/Iraf_6.1/%s/spectra/com/%s' % (folder, par[1]), memmap=False) 
        except:
            hdulist = fits.open('/media/storage/HERMES_REDUCED/dr6.1/%s/spectra/com/%s' % (folder, par[1]), memmap=False) 
                     
        header = hdulist[1].header
        y = hdulist[1].data

        ws = float(header['CRVAL1'])
        wd = float(header['CDELT1'])                
        x = np.linspace(ws, ws+len(y)*wd, num=len(y), endpoint=False) 
        x = doppler_shift(x, -RV)
                    
        # INTERPOLATE SPECTRUM
        y = np.interp(par[0], x, y, 1, 1)

        y_arr.append(y)
    
    return data, y_arr
    
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
    element_abundances = {"Li": abond, "O": 0.0}  # elemental abundances [X/Fe]; if not written solar scaled ones are used
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

def make_synSB(teff1,logg1,met1,vmic1,FWHM1,teff2,logg2,met2,vmic2,FWHM2,RV1,RV2,ratios, individual=False):
    
    y_arr = []
    sint1a_arr = []
    sint2a_arr = []

    for i, j in enumerate([xc1, xc2, xc3, xc4]):
        try:
            #making the syn A spectrum     
            template_wavelength_1, template_flux_1 = turbo(teff1,logg1,met1,vmic1,np.min(j),np.max(j),FWHM1,0,0.0)
            
            #making the syn B spectrum     
            template_wavelength_2, template_flux_2 = turbo(teff2,logg2,met2,vmic2,np.min(j),np.max(j),FWHM2,0,0.0)

            #doppler shift
            template_wavelength_1_rv = doppler_shift(template_wavelength_1, RV1)
            template_wavelength_2_rv = doppler_shift(template_wavelength_2, RV2)

            sint1a = 1./(1.+ratios[i]) * np.interp(j, template_wavelength_1_rv, template_flux_1, 1, 1)
            sint2a = 1./(1.+(1./ratios[i])) * np.interp(j, template_wavelength_2_rv, template_flux_2, 1, 1)

            sint1a_arr.append(sint1a)
            sint2a_arr.append(sint2a)

            y_arr.append(sint1a + sint2a)
        except:
            return np.zeros(len(xc1)+len(xc2)+len(xc3)+len(xc4))

    if individual:
        return sint1a_arr, sint2a_arr
    else:
        return y_arr

def chi_squared(params, y):

    #RV = 105
    #ratio = 0.67371
    
    #mask = (x < lmax) & (x > lmin)
    #x = x[mask]  
    #y = y[mask]
            
    
    syb_SB = make_synSB(
        params['teff_a'].value,params['logg_a'].value,params['met'].value,params['vmic_a'].value,params['FWHM_a'].value,
        params['teff_b'].value,params['logg_b'].value,params['met'].value,params['vmic_b'].value,params['FWHM_b'].value,
        params['RV_a'].value,params['RV_b'].value, [params['ratio1'].value, params['ratio2'].value, params['ratio3'].value, params['ratio4'].value]
    )
    
    
    #interp_func = interp1d(syb_SB[4], syb_SB[5], kind='linear')
    #syn_template_flux = interp_func(x)

    dif = []
    for i in range(4):
        dif.extend(y[i] - syb_SB[i])
    
    return dif
    
def fit_one_spectrum(sobject_id, return_dict):
    pars, y_arr = read_fits(sobject_id)

    params = Parameters()

    params.add('RV_a', float(pars['V1_50']), vary=True)
    params.add('teff_a', float(pars['teff1_50']), vary=True)
    params.add('logg_a', float(pars['logg1_50']), vary=True)
    params.add('met', float(pars['feh_50']), vary=True)
    params.add('vmic_a', float(pars['vmic1_50']), vary=True)
    params.add('FWHM_a', 0.5, vary=True)

    params.add('RV_b', float(pars['V2_50']), vary=True)
    params.add('teff_b', float(pars['teff2_50']), vary=True)
    params.add('logg_b', float(pars['logg2_50']), vary=True)
    #params.add('met_b', float(pars['feh_50']), vary=True)
    params.add('vmic_b', float(pars['vmic2_50']), vary=True)
    params.add('FWHM_b', 0.5, vary=True)

    params.add('ratio1', float(pars['ratio1_50']), vary=True)
    params.add('ratio2', float(pars['ratio2_50']), vary=True)
    params.add('ratio3', float(pars['ratio3_50']), vary=True)
    params.add('ratio4', float(pars['ratio4_50']), vary=True)

    #params.add('abond_a', abond_a, min = -5.0, max = 10.0)
    #params.add('abond_b', abond_b, min = -5.0, max = 10.0)

    #just a counter
    result = minimize(chi_squared, params, method='leastsq', args=(y_arr,))

    print(result.params)
    return_dict[sobject_id] = result

def get_cannon(teff, logg, feh, vmic=False, vsini=False, ak=False, njob=1):  
  #start = time.time() 
  sint = g_thetas[:, 0]*0.0
  print(teff, logg, feh, vmic, vsini, 0)
  labs = (np.array([teff, logg, feh, vmic, vsini, 0]) - g_fid) / g_sca
  
  vec = g_vectorizer(labs)[0]
  
  for i, j in enumerate(vec):
    sint += g_thetas[:, i]*j

  return [sint[g_m1], sint[g_m2], sint[g_m3], sint[g_m4]]
    
if False:
    '''[('RV_a', <Parameter 'RV_a', value=0.10130088915411824, bounds=[-inf:inf]>), ('teff_a', <Parameter 'teff_a', value=5921.470010004442, bounds=[-inf:inf]>), ('logg_a', <Parameter 'logg_a', value=3.1912906074133893, bounds=[-inf:inf]>), ('met', <Parameter 'met', value=-0.7225645873380172, bounds=[-inf:inf]>), ('vmic_a', <Parameter 'vmic_a', value=1.3749655489379948, bounds=[-inf:inf]>), ('FWHM_a', <Parameter 'FWHM_a', value=0.32459948568064, bounds=[-inf:inf]>), ('RV_b', <Parameter 'RV_b', value=-56.20739452597636, bounds=[-inf:inf]>), ('teff_b', <Parameter 'teff_b', value=5969.5773051892265, bounds=[-inf:inf]>), ('logg_b', <Parameter 'logg_b', value=3.0795239960923464, bounds=[-inf:inf]>), ('vmic_b', <Parameter 'vmic_b', value=2.889377752483208, bounds=[-inf:inf]>), ('FWHM_b', <Parameter 'FWHM_b', value=0.2981134625319619, bounds=[-inf:inf]>), ('ratio1', <Parameter 'ratio1', value=0.6355627846259799, bounds=[-inf:inf]>), ('ratio2', <Parameter 'ratio2', value=0.6702174339667246, bounds=[-inf:inf]>), ('ratio3', <Parameter 'ratio3', value=0.6004373582439976, bounds=[-inf:inf]>), ('ratio4', <Parameter 'ratio4', value=0.573118822365639, bounds=[-inf:inf]>)]'''
    pars, y_arr = read_fits('171230001601376')
    #making the syn A spectrum     
    template_wavelength_1, template_flux_1 = turbo(5921,3.19,-0.722,1.37496,np.min(xc3),np.max(xc3),0.3245,0,0.0)

    #making the syn B spectrum
    template_wavelength_2, template_flux_2 = turbo(5969.57,3.07,-0.722,2.8893,np.min(xc3),np.max(xc3),0.298,0,0.0)

    #doppler shift
    template_wavelength_1_rv = doppler_shift(template_wavelength_1, 0.1)
    template_wavelength_2_rv = doppler_shift(template_wavelength_2, -56.2)

    sint1a = 1./(1.+0.600) * np.interp(xc3, template_wavelength_1_rv, template_flux_1, 1, 1)
    sint2a = 1./(1.+(1./0.600)) * np.interp(xc3, template_wavelength_2_rv, template_flux_2, 1, 1)



    plt.plot(xc3, y_arr[2])
    plt.plot(xc3, sint1a)
    plt.plot(xc3, sint2a)
    plt.plot(xc3, sint1a+sint2a)
    plt.show()



    #recording the start
    start = time.time()

    #define the band
    #564u
    #if max(observed_wavelength) < 6800 and min(observed_wavelength) > 4800 :
     #   lmin = 5840
     #   lmax = 6000


    #global lmin 
    #global lmax 

    #Li line
    # 6707.7635

    #lmin = 6705.6
    #lmax = 6712.6


    #lmin = 6450
    #lmax = 6750


    #-----------
    #parameters rought numbers
    print('Loading Template 1')

    #teff_a = 5171
    #logg_a = 4.1
    #met_a = -0.2

    #vmic_a = 1.0

    #FWHM_a = 0.34

    #abond_a= 3.00


    #-----------

    #parameters rought numbers
    print('Loading Template 2')

    #teff_b = 5839
    #logg_b = 3.5
    #met_b = -0.2

    #vmic_b = 1.0

    #FWHM_b = 0.34

    #abond_b= 3.00


    #-----------
    #make template

    #RV = 105

    #ratio = 0.67371


    #-----------
    #reading data

listb = '''140117001501305
140314002601234
140710000101175
150409004101048
150411005101219
150607003601283
160106001601393
160814000101099
161008003501003
161210002601106
161217005101155
170108003901226
170109002101098
170109002101301
170418002101087
170615004901298
170713001601271
171031003301266
171205002601359
171230001601376
180101001601081
180101004301382'''


listb = listb.split()


if True:
    pars, y_arr = read_fits('170129002601083')
    #print(pars)
    
    #ratios = [pars['ratio1_50'].value[0], pars['ratio2_50'].value[0], pars['ratio3_50'].value[0], pars['ratio4_50'].value[0]]
    #cannon_sint1 = get_cannon(pars['teff1_50'].value[0], pars['logg1_50'].value[0], pars['feh_50'].value[0], pars['vmic1_50'].value[0], pars['vbroad1_50'].value[0], False, njob=1)
    #cannon_sint2 = get_cannon(pars['teff2_50'].value[0], pars['logg2_50'].value[0], pars['feh_50'].value[0], pars['vmic2_50'].value[0], pars['vbroad2_50'].value[0], False, njob=1)

    
    mcmc = Table.read('../selection_results_dec12all.fits')
    pars = mcmc[mcmc['sobject_id'] == 170129002601083]
    print(pars)
    ratios = [pars['mcmc_ratio1'].value[0], pars['mcmc_ratio2'].value[0], pars['mcmc_ratio3'].value[0], pars['mcmc_ratio4'].value[0]]
    cannon_sint1 = get_cannon(pars['mcmc_teff1'].value[0], pars['mcmc_logg1'].value[0], pars['mcmc_met'].value[0], pars['mcmc_Vmic1'].value[0], 10, False, njob=1)
    cannon_sint2 = get_cannon(pars['mcmc_teff2'].value[0], pars['mcmc_logg2'].value[0], pars['mcmc_met'].value[0], pars['mcmc_Vmic2'].value[0], 10, False, njob=1)
    
    cannon_sint1a = []
    cannon_sint2a = []

    #doppler shift
    for h, xa in enumerate([xc1, xc2, xc3, xc4]):
        #g_sint_wavs_1_rv = doppler_shift(g_sint_wavs[h], pars['V1_50'].value[0])
        #g_sint_wavs_2_rv = doppler_shift(g_sint_wavs[h], pars['V2_50'].value[0])
        g_sint_wavs_1_rv = doppler_shift(g_sint_wavs[h], pars['mcmc_V1'].value[0])
        g_sint_wavs_2_rv = doppler_shift(g_sint_wavs[h], pars['mcmc_V2'].value[0])
        cannon_sint1a.append(1./(1.+ratios[h]) * np.interp(xa, g_sint_wavs_1_rv, cannon_sint1[h], 1, 1))
        cannon_sint2a.append(1./(1.+(1./ratios[h])) * np.interp(xa, g_sint_wavs_2_rv, cannon_sint1[h], 1, 1))

    for h, xa in enumerate([xc1, xc2, xc3, xc4]):

        fig2, ax = plt.subplots(figsize=(30, 7))
        
        ax.plot(xa, y_arr[h], c='black', lw=1, label='observed')
        ax.plot(xa, cannon_sint1a[h], c='green', lw=1, ls='--', label='primary Cannon')
        ax.plot(xa, cannon_sint2a[h], c='blue', lw=1, ls='--', label='secondary Cannon')
        ax.plot(xa, cannon_sint1a[h]+cannon_sint2a[h], c='red', lw=1, ls='--', label='primary+secondary Cannon')
        #plt.title(f"{pars['sobject_id']}: GALAH SB2 result: {', '.join([str(k)+': '+'%.2f'%pars[dic[k]] for k in fitted_pars])}\nCHI2_reduced: {'%.2f'%(pars['chi2_binary_pipeline'].value[0]/14304.)}, ruwe: {'%.2f'%pars['ruwe']}")
        plt.tight_layout()
        #fig2.savefig(f"{i['sobject_id']}_{h}.png")
        plt.show()
        plt.close(fig2)

if False:

    results = Table.read('results.csv')
    results.rename_column('#sobject_id', 'sobject_id')
    print(results)

    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(40, 20))
    yosn = []
    yosl = []
    for j, i in enumerate(results):

        yosn.append(j)
        yosl.append(i['sobject_id'])

        pars, y_arr = read_fits(str(i['sobject_id']))
        ratios = [pars['ratio1_50'].value[0], pars['ratio2_50'].value[0], pars['ratio3_50'].value[0], pars['ratio4_50'].value[0]]

        cannon_sint1 = get_cannon(pars['teff1_50'].value[0], pars['logg1_50'].value[0], pars['feh_50'].value[0], pars['vmic1_50'].value[0], pars['vbroad1_50'].value[0], False, njob=1)
        cannon_sint2 = get_cannon(pars['teff2_50'].value[0], pars['logg2_50'].value[0], pars['feh_50'].value[0], pars['vmic2_50'].value[0], pars['vbroad2_50'].value[0], False, njob=1)
        #doppler shift
        
        cannon_sint1a = []
        cannon_sint2a = []

        for h, xa in enumerate([xc1, xc2, xc3, xc4]):
            g_sint_wavs_1_rv = doppler_shift(g_sint_wavs[h], pars['V1_50'].value[0])
            g_sint_wavs_2_rv = doppler_shift(g_sint_wavs[h], pars['V2_50'].value[0])
            cannon_sint1a.append(1./(1.+ratios[h]) * np.interp(xa, g_sint_wavs_1_rv, cannon_sint1[h], 1, 1))
            cannon_sint2a.append(1./(1.+(1./ratios[h])) * np.interp(xa, g_sint_wavs_2_rv, cannon_sint1[h], 1, 1))
        
        sint1a, sint2a = make_synSB(
            i['teff_a'],i['logg_a'],i['met'],i['vmic_a'],i['FWHM_a'],
            i['teff_b'],i['logg_b'],i['met'],i['vmic_b'],i['FWHM_b'],
            i['RV_a'],i['RV_b'], [i['ratio1'], i['ratio2'], i['ratio3'], i['ratio4']],
            individual=True
        )
        for h, xa in enumerate([xc1, xc2, xc3, xc4]):

            fig2, ax = plt.subplots(figsize=(30, 7))
            
            ax.plot(xa, y_arr[h], c='black', lw=2, label='observed')
            ax.plot(xa, sint1a[h], c='green', lw=1, label='primary')
            ax.plot(xa, sint2a[h], c='blue', lw=1, label='secondary')
            ax.plot(xa, sint1a[h]+sint2a[h], c='red', lw=1, label='primary+secondary')
            ax.plot(xa, cannon_sint1a[h], c='green', lw=1, ls='--', label='primary Cannon')
            ax.plot(xa, cannon_sint2a[h], c='blue', lw=1, ls='--', label='secondary Cannon')
            ax.plot(xa, cannon_sint1a[h]+cannon_sint2a[h], c='red', lw=1, ls='--', label='primary+secondary Cannon')
            plt.title(f"{i['sobject_id']}: {', '.join([str(k)+': '+'%.2f'%i[k] for k in fitted_pars])}\nGALAH SB2 result: {', '.join([str(k)+': '+'%.2f'%pars[dic[k]] for k in fitted_pars])}\nCHI2_reduced: {'%.2f'%(pars['chi2_binary_pipeline'].value[0]/14304.)}, ruwe: {'%.2f'%pars['ruwe']}")
            plt.tight_layout()
            fig2.savefig(f"{i['sobject_id']}_{h}.png")
            plt.close(fig2)

        nrow = 0
        ncol = 0
        for k, par in enumerate(fitted_pars):  

            axes[nrow][ncol].scatter(float(pars[dic[par]]), j, c='black', label=f"{j} = {i['sobject_id']}", s=50)
            axes[nrow][ncol].scatter(i[par], j, c='red', s=50)
            axes[nrow][ncol].arrow(float(pars[dic[par]]), j, 0.8*(i[par]-float(pars[dic[par]])), 0, color='grey', width=0.1, head_width=0.0, head_length=0.0)
            axes[nrow][ncol].set_xlabel(par)
            axes[nrow][ncol].yaxis.set_major_locator(MultipleLocator(1))

            ncol += 1    
            if ncol % 6 == 0:
                nrow += 1
                ncol = 0
            
    plt.suptitle('Comparison of Traven et al. SB2 results (black) with Turbospectrum (red) ')
    axes[0][0].set_yticks(yosn, yosl, color='black', fontsize=10, horizontalalignment='right')    
    plt.tight_layout()
    fig.savefig('SB2_turbo_results.png')
    #plt.show()
    fsdfsdds

if False:
    start = time.time() 

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

    

    with open('results.csv', 'w') as f:
        f.write('#sobject_id,%s\n' % (','.join([k for k in fitted_pars])))

        for i, j in result.items():
            print(i, j)
            f.write('%s,%s\n' % (i, ','.join([str(j.params[k].value) for k in fitted_pars])))

    fdfsfsd
    #print(result.redchi)
    #print(result.chisqr)

if False:

    for i in listb:

        pars, y_arr = read_fits(i)
        data_red = dr6table[dr6table['sobject_id'] == int(i)]
        data_red_old = gaia[gaia['sobject_id'] == int(i)]
        #print(data_red_old['rv_guess'])
        
        RV = data_red['rv_guess']

        #-----------
        #parameters to fit
        Ha = 6562.8#4861.3#6562.8

        print(i, RV, float(pars['V1_50']), float(pars['V2_50']))
        plt.plot(doppler_shift(xc3, -float(data_red_old['rv_guess'])), y_arr[2], c='red', label='old spectrum')
        plt.plot(doppler_shift(xc3, -RV), y_arr[2], c='black', label='new spectrum')
        
        plt.axvline(Ha, c='grey', lw=2, label='Ha rest frame')
        
        plt.axvline(Ha+Ha*float(pars['V1_50'])/c, c='orange', label='SB2 prim. RV')
        plt.axvline(Ha+Ha*float(pars['V2_50'])/c, c='green', label='SB2 sec. RV')

        plt.legend()

        plt.show()

#make the syn based on the results.
print('making SB2 Spectrum')
wv_a, fl_a, wv_b, fl_b, wv_sb, fl_sb = make_synSB(params['teff_a'].value,params['logg_a'].value,params['met'].value,params['vmic_a'].value,params['FWHM_a'].value,params['abond_a'].value
           ,params['teff_b'].value,params['logg_b'].value,params['met'].value,params['vmic_b'].value,params['FWHM_b'].value,params['abond_b'].value
           ,RV,ratio)


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


plt.savefig('../fig/sb2_test.pdf')





#end time recorder and print the time
end = time.time()
    

print('-----------------\n')
print('\n')
timer(end-start)
print('\n')
print('-----------------\n')

print('**********DONE**********')


#-----------













































#
