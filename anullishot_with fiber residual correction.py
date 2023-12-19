#This is making an attempt to make the code less memory intensive
import sys
import os
import glob
import gc
import astropy.units as u
from astropy.coordinates import SkyCoord
import hetdex_api
from hetdex_api.shot import *
from hetdex_api.extract import Extract
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
from astropy.cosmology import Planck18 as cosmo
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import FiberIndex
import numpy as np
from hetdex_api.config import HDRconfig
from elixer import spectrum_utilities as SU
import tables
from astropy.table import QTable, Table, Column
from elixer import global_config as G
import pickle
from hetdex_api.shot import Fibers

detects_table = Table.read('anulli_table.fits')
detect_list = detects_table['detectid']

wl = np.linspace(3470,5540,1036)
wl_vac=SU.air_to_vac(wl)
conv=1e-17

G.GLOBAL_LOGGING = True
#G.HDR_Version='4'

def kpc_to_arcsec(distance_kpc, z):
    D_A = cosmo.angular_diameter_distance(z).value 
    angular_size_radian = float(distance_kpc) / (D_A * 1000)  #D_A to kpc
    return angular_size_radian * (180 * 3600) / np.pi  #radian to arcsec

def process_id(iden, F=None):
    
    waves = []
    flux_densities = []
    flux_err = []
    indices = np.where(detects_table['detectid'] == iden)[0]
    det = detects_table[indices]['shotid', 'ra', 'dec', 'z_hetdex', 'fwhm', 'throughput']
    shotid = det['shotid'][0]
    ra = det['ra'][0]
    dec = det['dec'][0]
    z = det['z_hetdex'][0]
    seeing= det['fwhm'][0]
    throughput=det['throughput'][0]
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    radius_in_arcsec = kpc_to_arcsec(radius_in, z)
    radius_out_arcsec = kpc_to_arcsec(radius_out, z)
    
    
    fibtab_in = F.query_region(coords=coord, radius=radius_in_arcsec*u.arcsec, shotid=shotid)['fiber_id','flag']
    fibtab_out = F.query_region(coords=coord, radius=radius_out_arcsec*u.arcsec, shotid=shotid)['fiber_id','flag']
    fibtab =np.setdiff1d(fibtab_out, fibtab_in)
    fibtab_len=len(fibtab)
    del fibtab_in
    del fibtab_out
    
    
    if fibtab_len == 0:
        return None, None, None, 0, 0
    else:
        fibers_in = get_fibers_table(shot=shotid, coords=coord, radius=radius_in_arcsec*u.arcsec, survey='hdr4', verbose=False,add_rescor=True)['fiber_id', 'calfib_ffsky_rescor', 'calfibe']
        fibers_out = get_fibers_table(shot=shotid, coords=coord, radius=radius_out_arcsec*u.arcsec, survey='hdr4', verbose=False,add_rescor=True)['fiber_id', 'calfib_ffsky_rescor', 'calfibe']
        fibers = np.setdiff1d(fibers_out, fibers_in)
        del fibers_in
        del fibers_out
        
        
        super_tab = join(fibers, fibtab, "fiber_id")
        mask = super_tab['flag'] == True
        super_tab = super_tab[mask]
        del super_tab['fiber_id']
        num_zeros_nans = np.sum((super_tab['calfib_ffsky_rescor'] == 0) | np.isnan(super_tab['calfib_ffsky_rescor']), axis=1)
        mask2 = num_zeros_nans <= 100
        super_tab = super_tab[mask2]
        
        
    specs = super_tab['calfib_ffsky_rescor']
    specs = np.where(specs == 0, np.nan, specs)
    spec_counter = len(specs)
    mask_zone1 = (3500 < wl_vac) & (wl_vac < 3860)
    mask_zone2 = (3860 < wl_vac) & (wl_vac < 4270)
    mask_zone3 = (4270 < wl_vac) & (wl_vac < 4860)
    mask_zone4 = (4860 < wl_vac) & (wl_vac < 5090)
    mask_zone5 = (5090 < wl_vac) & (wl_vac < 5500)
    medians = np.array([np.nanmedian(specs[:, mask], axis=1) for mask in [mask_zone1, mask_zone2, mask_zone3, mask_zone4, mask_zone5]])
    valid_mask = np.all((-0.05 < medians) & (medians < 0.05), axis=0)
    valid_specs = specs[valid_mask]
    valid_spec_counter = len(valid_specs)
    if valid_spec_counter == 0:
        return None, None, None, 0, 0
    else:
        valid_errs = super_tab['calfibe'][valid_mask]
        valid_errs = np.where(valid_errs == 0, np.nan, valid_errs)
        err = np.nanmedian(valid_errs, axis=0)
        valid_specs[np.isnan(valid_errs)] = np.nan
        median_spec = np.nanmedian(valid_specs, axis=0)
        res, res_err, res_cont= SU.get_empty_fiber_residual(hdr='4', rtype='t01', shotid=shotid, seeing=seeing,response=throughput,ffsky=True,add_rescor=True ,persist=True)
        final_spec = median_spec - res

    flux_density=final_spec*conv
    eflux=err*conv
    lum,rest_wl,lum_err=SU.shift_flam_to_rest_luminosity_per_aa(z,flux_density,wl,eflux=eflux,apply_air_to_vac=True)
    waves.append(rest_wl)
    flux_densities.append(lum)
    flux_err.append(lum_err)
    return waves, flux_densities, flux_err, spec_counter, valid_spec_counter

def process_shotid(shotid):
    F = FiberIndex("hdr4")
    sel_shot = detects_table['shotid'] == shotid
    detect_list = detects_table['detectid'][sel_shot]
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
        gc.collect()
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
        shots = np.unique(detects_table['shotid'])
    for radius_in, radius_out in radii:
        total_specs = 0
        total_valid_specs = 0
        num = len(shots)
        for shot in shots[:num]:
            spec_counter, valid_spec_counter = process_shotid(shot)
            total_specs += spec_counter
            total_valid_specs += valid_spec_counter
            print(f"total valid fibers of this shot = {total_valid_specs}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 anulli.py <radius_in> <radius_out> [shot_id]")
        sys.exit(1)
    radius_in = sys.argv[1]
    radius_out = sys.argv[2]
    shot_id = sys.argv[3] if len(sys.argv) > 3 else None
    main(radius_in, radius_out, shot_id)
