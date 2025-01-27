from __future__ import annotations

import datetime
import shutil
from collections import OrderedDict
from configparser import ConfigParser
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from warnings import warn
from scripts.convolve import conv_macroturbulence, conv_rotation, conv_res
from scripts.create_window_linelist_function import create_window_linelist
from scripts.turbospectrum_class_nlte import TurboSpectrum
from scripts.m3dis_class import M3disCall
from scripts.synthetic_code_class import SyntheticSpectrumGenerator
from scripts.synthetic_code_class import fetch_marcs_grid
from scripts.TSFitPy import (output_default_configuration_name, output_default_fitlist_name,
                             output_default_linemask_name)
from scripts.auxiliary_functions import (calculate_equivalent_width, apply_doppler_correction, import_module_from_path,
                                         combine_linelists)
from scripts.loading_configs import SpectraParameters, TSFitPyConfig
from scripts.solar_abundances import periodic_table



def check_if_path_exists(path_to_check: str) -> str:
    # check if path is absolute
    if os.path.isabs(path_to_check):
        if os.path.exists(os.path.join(path_to_check, "")):
            return path_to_check
        else:
            raise ValueError(f"Configuration: {path_to_check} does not exist")
    # if path is relative, check if it exists in the current directory
    if os.path.exists(os.path.join(path_to_check, "")):
        # returns absolute path
        return os.path.join(os.getcwd(), path_to_check, "")
    else:
        # if it starts with ../ convert to ./ and check again
        if path_to_check.startswith("../"):
            path_to_check = path_to_check[3:]
            if os.path.exists(os.path.join(path_to_check, "")):
                return os.path.join(os.getcwd(), path_to_check, "")
            else:
                raise ValueError(f"Configuration: {path_to_check} does not exist")
        else:
            raise ValueError(f"Configuration: {path_to_check} does not exist")


def plot_synthetic_data(turbospectrum_paths, teff, logg, met, vmic, lmin, lmax, ldelta, atmosphere_type, nlte_flag,
                        elements_in_nlte, element_abundances, include_molecules, resolution=0, macro=0, rotation=0,
                        verbose=False, return_unnorm_flux=False, do_matplotlib_plot=False):
    print('AVOCADO-17')
    for element in element_abundances:
        element_abundances[element] += met
    temp_directory = f"../temp_directory/temp_directory_{datetime.datetime.now().strftime('%b-%d-%Y-%H-%M-%S')}__{np.random.random(1)[0]}/"

    temp_directory = os.path.join(os.getcwd(), temp_directory, "")
    
    for path in turbospectrum_paths:
        turbospectrum_paths[path] = check_if_path_exists(turbospectrum_paths[path])

    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

    if atmosphere_type == "1D":
        model_atmosphere_grid_path = os.path.join(turbospectrum_paths["model_atmosphere_grid_path"], "1D", "")
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"
    elif atmosphere_type == "3D":
        model_atmosphere_grid_path = os.path.join(turbospectrum_paths["model_atmosphere_grid_path"], "3D", "")
        model_atmosphere_list = model_atmosphere_grid_path + "model_atmosphere_list.txt"

    model_temperatures, model_logs, model_mets, marcs_value_keys, marcs_models, marcs_values = fetch_marcs_grid(
        model_atmosphere_list, TurboSpectrum.marcs_parameters_to_ignore)

    depart_bin_file_dict, depart_aux_file_dict, model_atom_file_dict = {}, {}, {}
    aux_file_length_dict = {}

    if nlte_flag:
        nlte_config = ConfigParser()
        nlte_config.read(os.path.join(turbospectrum_paths["departure_file_path"], "nlte_filenames.cfg"))

        for element in elements_in_nlte:
            if atmosphere_type == "1D":
                bin_config_name, aux_config_name = "1d_bin", "1d_aux"
            else:
                bin_config_name, aux_config_name = "3d_bin", "3d_aux"
            depart_bin_file_dict[element] = nlte_config[element][bin_config_name]
            depart_aux_file_dict[element] = nlte_config[element][aux_config_name]
            model_atom_file_dict[element] = nlte_config[element]["atom_file"]

        for element in model_atom_file_dict:
            aux_file_length_dict[element] = len(np.loadtxt(os.path.join(turbospectrum_paths["departure_file_path"], depart_aux_file_dict[element]), dtype='str'))

    today = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")  # used to not conflict with other instances of fits
    today = f"{today}_{np.random.random(1)[0]}"
    line_list_path_trimmed = os.path.join(f"{temp_directory}", "linelist_for_fitting_trimmed", "")
    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "all", today, '')

    print("Trimming")
    create_window_linelist([lmin - 4], [lmax + 4], turbospectrum_paths["line_list_path"], line_list_path_trimmed, include_molecules, False)
    print("Trimming done")

    line_list_path_trimmed = os.path.join(line_list_path_trimmed, "0", "")

    ts = TurboSpectrum(
        turbospec_path=turbospectrum_paths["turbospec_path"],
        interpol_path=turbospectrum_paths["interpol_path"],
        line_list_paths=line_list_path_trimmed,
        marcs_grid_path=model_atmosphere_grid_path,
        marcs_grid_list=model_atmosphere_list,
        model_atom_path=turbospectrum_paths["model_atom_path"],
        departure_file_path=turbospectrum_paths["departure_file_path"],
        aux_file_length_dict=aux_file_length_dict,
        model_temperatures=model_temperatures,
        model_logs=model_logs,
        model_mets=model_mets,
        marcs_value_keys=marcs_value_keys,
        marcs_models=marcs_models,
        marcs_values=marcs_values)

    ts.configure(t_eff=teff, log_g=logg, metallicity=met,
                 turbulent_velocity=vmic, lambda_delta=ldelta, lambda_min=lmin - 3, lambda_max=lmax + 3,
                 free_abundances=element_abundances, temp_directory=temp_directory, nlte_flag=nlte_flag, verbose=verbose,
                 atmosphere_dimension=atmosphere_type, windows_flag=False, segment_file=None,
                 line_mask_file=None, depart_bin_file=depart_bin_file_dict,
                 depart_aux_file=depart_aux_file_dict, model_atom_file=model_atom_file_dict)
    print("Running TS")
    wave_mod_orig, flux_norm_mod_orig, flux_unnorm = ts.synthesize_spectra()
    print("TS completed")
    if wave_mod_orig is not None:
        if np.size(wave_mod_orig) != 0.0:
            try:
                wave_mod_filled = wave_mod_orig
                flux_norm_mod_filled = flux_norm_mod_orig

                if len(wave_mod_orig) > 0:
                    if resolution != 0.0:
                        wave_mod_conv, flux_norm_mod_conv = conv_res(wave_mod_filled, flux_norm_mod_filled, resolution)
                    else:
                        wave_mod_conv = wave_mod_filled
                        flux_norm_mod_conv = flux_norm_mod_filled

                    if macro != 0.0:
                        wave_mod_macro, flux_norm_mod_macro = conv_macroturbulence(wave_mod_conv, flux_norm_mod_conv, macro)
                    else:
                        wave_mod_macro = wave_mod_conv
                        flux_norm_mod_macro = flux_norm_mod_conv

                    if rotation != 0.0:
                        wave_mod, flux_norm_mod = conv_rotation(wave_mod_macro, flux_norm_mod_macro, rotation)
                    else:
                        wave_mod = wave_mod_macro
                        flux_norm_mod = flux_norm_mod_macro

                    if do_matplotlib_plot:
                        plt.plot(wave_mod, flux_norm_mod)
                        plt.xlim(lmin - 0.2, lmax + 0.2)
                        plt.ylim(0, 1.05)
                        plt.xlabel("Wavelength")
                        plt.ylabel("Normalised flux")
                else:
                    print('TS failed')
                    wave_mod, flux_norm_mod = np.array([]), np.array([])
                    flux_unnorm = np.array([])
            except (FileNotFoundError, ValueError, IndexError) as e:
                print(f"TS failed: {e}")
                wave_mod, flux_norm_mod = np.array([]), np.array([])
                flux_unnorm = np.array([])
        else:
            print('TS failed')
            wave_mod, flux_norm_mod = np.array([]), np.array([])
            flux_unnorm = np.array([])
    else:
        print('TS failed')
        wave_mod, flux_norm_mod = np.array([]), np.array([])
        flux_unnorm = np.array([])
    shutil.rmtree(temp_directory)
    #shutil.rmtree(line_list_path_trimmed)  # clean up trimmed line list
    if return_unnorm_flux:
        return wave_mod, flux_norm_mod, flux_unnorm
    else:
        return wave_mod, flux_norm_mod



if __name__ == '__main__':
    #test_star = Star("150429001101153.spec", ["../output_files/Nov-17-2023-00-23-55_0.1683492858486244_NLTE_Fe_1D/"], "../input_files/linelists/linelist_for_fitting/")
    #test_star.plot_fit_parameters_vs_abundance("ew", "Fe", abund_limits=(-3, 3))
    #test_star.plot_ep_vs_abundance("Fe")
    #test_star.plot_loggf_vs_abundance("Fe", abund_limits=(-3, 3))
    #test_star.plot_abundance_plot(abund_limits=(-3, 3))
    #print(test_star.get_average_abundances())

    test = get_average_abundance_all_stars(["../output_files/Nov-17-2023-00-23-55_0.1683492858486244_NLTE_Fe_1D/", "../output_files/Nov-17-2023-00-23-55_0.1683492858486244_NLTE_Fe_1D/"], "../input_files/linelists/linelist_for_fitting/")
    print(test)
