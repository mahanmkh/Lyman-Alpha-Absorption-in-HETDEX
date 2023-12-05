import sys
import os
import glob
import astropy.units as u
from astropy.coordinates import SkyCoord
import hetdex_api
from hetdex_api.shot import *
from hetdex_api.extract import Extract
import time
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import FiberIndex
import numpy as np
from hetdex_api.config import HDRconfig
from elixer import spectrum_utilities as SU
import tables
from astropy.table import QTable, Table, Column
from elixer import global_config as G
from multiprocessing import Pool
import pickle
from hetdex_api.shot import Fibers
import warnings
warnings.filterwarnings('ignore', 'NOT fine tuning model. TURN OFF')

version = '3.0.1'
config = HDRconfig('hdr3')
catfile = op.join(config.hdr_dir['hdr3'], 'catalogs', 'source_catalog_' + version + '.fits') #source_catalog_ vs detect_hdr
detects_table = Table.read(catfile)

sel = (detects_table['shotid'] > 20180100000)
sel = sel & np.array(detects_table['plya_classification'] > 0.8)
sel = sel & np.array(detects_table['flag_best'] == 1)
sel = sel & np.array(detects_table['z_hetdex'] > 1.9) & np.array(detects_table['z_hetdex'] < 3.5)
sel = sel & np.array(detects_table['source_type'] == 'lae')
sel = sel & np.array(detects_table['selected_det']==True)
sel = sel & np.array(detects_table['best_pz'] > 0.2)
sel = sel & np.array(detects_table['linewidth'] < 5.5)
sel = sel & np.array(detects_table['sn'] > 5) & np.array(detects_table['sn'] < 6)
sel = sel & np.array(detects_table['apcor'] > 0.6)
sel = sel & np.array(detects_table['fwhm'] < 1.5 )
sel = sel & np.array(detects_table['lum_lya'] < 1e43)
detect_list = detects_table['detectid'][sel]

wl = np.linspace(3470,5540,1036)
wl_vac=SU.air_to_vac(wl)
conv=1e-17

def kpc_to_arcsec(distance_kpc, z):
    D_A = cosmo.angular_diameter_distance(z).value 
    angular_size_radian = float(distance_kpc) / (D_A * 1000)  #D_A to kpc
    return angular_size_radian * (180 * 3600) / np.pi  #radian to arcsec

def process_id(iden, F=None):
    waves = []
    flux_densities = []
    flux_err = []
    sel = (detects_table['detectid'] == iden)
    det = detects_table[sel]
    shotid = det['shotid'][0]
    ra = det['ra'][0]
    dec = det['dec'][0]
    z = det['z_hetdex'][0]
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    shot = shotid
    radius_in_arcsec = kpc_to_arcsec(radius_in, z)
    radius_out_arcsec = kpc_to_arcsec(radius_out, z)
    fibers_in = get_fibers_table(shot=shot, coords=coord, radius=radius_in_arcsec*u.arcsec, survey='hdr3', verbose=False)
    fibers_out = get_fibers_table(shot=shot, coords=coord, radius=radius_out_arcsec*u.arcsec, survey='hdr3', verbose=False)
    fibers = np.setdiff1d(fibers_out, fibers_in)
    specs = fibers['calfib_ffsky']
    specs = np.where(specs == 0, np.nan, specs)
    spec_counter = len(specs)
    zone1_means = []
    zone2_means = []
    zone3_means = []
    valid_specs = []
    valid_spec_indices = []
    for i, fib in enumerate(specs):
        zone1, zone2, zone3 = [], [], []
        for w, f in zip(wl_vac, fib):
            if 3500 < w < 4000:
                zone1.append(f)
            elif 4000 < w < 5000:
                zone2.append(f)
            elif 5000 < w < 5500:
                zone3.append(f)
        mean_zone1, std_zone1 = np.nanmean(zone1), np.nanstd(zone1)
        mean_zone2, std_zone2 = np.nanmean(zone2), np.nanstd(zone2)
        mean_zone3, std_zone3 = np.nanmean(zone3), np.nanstd(zone3)
        if -0.02 < mean_zone1 < 2.5 and -0.02 < mean_zone2 < 0.02 and -0.02 < mean_zone3 < 0.02:
            zone1_means.append(mean_zone1)
            zone2_means.append(mean_zone2)
            zone3_means.append(mean_zone3)
            valid_specs.append(fib)
            valid_spec_indices.append(i)
    zone1_avg, zone1_std = np.mean(zone1_means), np.std(zone1_means)
    zone2_avg, zone2_std = np.mean(zone2_means), np.std(zone2_means)
    zone3_avg, zone3_std = np.mean(zone3_means), np.std(zone3_means)
    valid_specs_and_indices = [(spec, i) for i, spec in enumerate(valid_specs) if 
                   np.abs(zone1_means[i]-zone1_avg) <= 3*zone1_std and 
                   np.abs(zone2_means[i]-zone2_avg) <= 3*zone2_std and 
                   np.abs(zone3_means[i]-zone3_avg) <= 3*zone3_std]
    valid_spec_indices = [i for i, spec in enumerate(valid_specs) if
                          np.abs(zone1_means[i]-zone1_avg) <= 3*zone1_std and 
                          np.abs(zone2_means[i]-zone2_avg) <= 3*zone2_std and 
                          np.abs(zone3_means[i]-zone3_avg) <= 3*zone3_std]
    valid_spec_counter = len(valid_specs)
    valid_specs = np.array([spec for spec, i in valid_specs_and_indices])
    valid_spec_indices = [i for spec, i in valid_specs_and_indices]
    if valid_spec_counter == 0:
        #print(f"Warning: No valid specs for iden {iden}.")
        return None, None, None, 0, 0
    else:
        valid_specs = np.array(valid_specs)
        errs = fibers['calfibe']
        valid_errs = errs[valid_spec_indices]
        valid_errs = np.where(valid_errs == 0, np.nan, valid_errs)
        err = np.nanmedian(valid_errs, axis=0)
        valid_specs[np.isnan(valid_errs)] = np.nan
        median_spec = np.nanmedian(valid_specs, axis=0)
        seeing= det['fwhm'][0]
        throughput= det['throughput'][0]
        G.HDR_Version='3'
        correction=SU.interpolate_universal_single_fiber_sky_subtraction_residual(seeing,ffsky=True,hdr=G.HDR_Version,zeroflat=False,response=throughput,xfrac=0.3)
        #if correction is None:
            #return None, None, None, 0, 0
        final_spec=median_spec - correction
    flux_density=final_spec*conv
    eflux=err*conv
    lum,rest_wl,lum_err=SU.shift_flam_to_rest_luminosity_per_aa(z,flux_density,wl,eflux=eflux,apply_air_to_vac=True)
    waves.append(rest_wl)
    flux_densities.append(lum)
    flux_err.append(lum_err)
    return waves, flux_densities, flux_err, spec_counter, valid_spec_counter

def process_shotid(shotid):
    F = Fibers(shotid, survey='hdr3')
    sel_shot = detects_table['shotid'] == shotid
    detect_list = detects_table['detectid'][sel & sel_shot]
    waves = []
    flux_densities = []
    flux_err = []
    spec_counter = 0
    valid_spec_counter = 0
    for det in detect_list:
        r1, r2, r3, spec_count, valid_spec_count = process_id(det, F=F)
        if r1 is None or r2 is None or r3 is None:
            continue
        spec_counter += spec_count
        valid_spec_counter += valid_spec_count
        waves.append(r1)
        flux_densities.append(r2)
        flux_err.append(r3)
    F.close()
    if len(waves) == 0 or len(flux_densities) == 0 or len(flux_err) == 0:
        print(f"Warning: Empty results for shotid {shotid}.")
        return spec_counter, valid_spec_counter
    with open('zstore/fiber_spectra_newway_{}_{}_{}.pickle'.format(shotid, radius_in, radius_out), 'wb') as f:
        pickle.dump((waves, flux_densities, flux_err), f)
    return spec_counter, valid_spec_counter

total_specs = 0
total_valid_specs = 0
def aggregate_counts(result):
    spec_counter, valid_spec_counter = result
    global total_specs
    global total_valid_specs
    total_specs += spec_counter
    total_valid_specs += valid_spec_counter

def main(radius_in, radius_out, shot_id=None):
    radii = [(int(radius_in), int(radius_out))]
    if shot_id:
        shots = [int(shot_id)]
    else:
        shots = np.unique(detects_table['shotid'][sel])
    for radius_in, radius_out in radii:
        total_specs = 0
        total_valid_specs = 0
        num = len(shots)
        for shot in shots[:num]:
            spec_counter, valid_spec_counter = process_shotid(shot)
            total_specs += spec_counter
            total_valid_specs += valid_spec_counter
            #print(f"total fibers of this shot = {total_specs}")
            print(f"total valid fibers of this shot = {total_valid_specs}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 anulli.py <radius_in> <radius_out> [shot_id]")
        sys.exit(1)
    radius_in = sys.argv[1]
    radius_out = sys.argv[2]
    shot_id = sys.argv[3] if len(sys.argv) > 3 else None
    main(radius_in, radius_out, shot_id)