#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import healpy as hp
from astropy.table import Table, vstack, unique, hstack, Column, join
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from hetdex_api.shot import get_fibers_table, Fibers
from hetdex_api.survey import FiberIndex
from hetdex_tools.phot_tools import get_line_image, get_flux_for_source
from hetdex_tools.source_catalog import plot_source_group
from hetdex_api.config import HDRconfig
from elixer import spectrum_utilities as SU

# In[2]:
def is_too_close_to_any_LAE(ra_fiber, dec_fiber, ra_LAEs, dec_LAEs, threshold_radius):
    for ra_LAE, dec_LAE in zip(ra_LAEs, dec_LAEs):
        dist = SkyCoord(ra=ra_fiber*u.degree, dec=dec_fiber*u.degree).separation(SkyCoord(ra=ra_LAE*u.degree, dec=dec_LAE*u.degree)).degree
        if dist < threshold_radius:
            return True
    return False


def get_rectangle_around_point(ra, dec, dir_vector, perpendicular, width, height, left=False):
    """Generate rectangle vertices around a given RA and Dec."""
    half_height = height / 2.0
    if left:
        return [
            [ra - width * dir_vector[0] - half_height * perpendicular[0], dec - width * dir_vector[1] - half_height * perpendicular[1]], # Bottom left
            [ra - half_height * perpendicular[0], dec - half_height * perpendicular[1]], # Bottom right (below LAE)
            [ra + half_height * perpendicular[0], dec + half_height * perpendicular[1]], # Top right (above LAE)
            [ra - width * dir_vector[0] + half_height * perpendicular[0], dec - width * dir_vector[1] + half_height * perpendicular[1]]  # Top left
        ]
    else:
        return [
            [ra - half_height * perpendicular[0], dec - half_height * perpendicular[1]], # Bottom left (below LAE)
            [ra + width * dir_vector[0] - half_height * perpendicular[0], dec + width * dir_vector[1] - half_height * perpendicular[1]], # Bottom right
            [ra + width * dir_vector[0] + half_height * perpendicular[0], dec + width * dir_vector[1] + half_height * perpendicular[1]], # Top right
            [ra + half_height * perpendicular[0], dec + half_height * perpendicular[1]]  # Top left (above LAE)
        ]


def is_inside_rectangle(ra, dec, vertices):
    def point_inside_triangle(p, A, B, C):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(p, A, B)
        d2 = sign(p, B, C)
        d3 = sign(p, C, A)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)    
    A, B, C, D = vertices
    M = [ra, dec]
    return (point_inside_triangle(M, A, B, C) or 
            point_inside_triangle(M, A, D, C) or
            point_inside_triangle(M, A, B, D) or
            point_inside_triangle(M, B, C, D))


def get_square_vertices(ra, dec, side_length):
    '''Return vertices of a square around given ra and dec'''
    vec_top_left = hp.ang2vec(ra - side_length/2, dec + side_length/2, lonlat=True)
    vec_top_right = hp.ang2vec(ra + side_length/2, dec + side_length/2, lonlat=True)
    vec_bot_left = hp.ang2vec(ra - side_length/2, dec - side_length/2, lonlat=True)
    vec_bot_right = hp.ang2vec(ra + side_length/2, dec - side_length/2, lonlat=True)
    return np.vstack([vec_top_left, vec_top_right, vec_bot_right, vec_bot_left])


def setup_healpix_region(pair, side_length, Nside):
    '''Return the combined healpix indices of two squares around the LAEs'''
    vertices_1 = get_square_vertices(pair['ra_1'], pair['dec_1'], side_length)
    vertices_2 = get_square_vertices(pair['ra_2'], pair['dec_2'], side_length)
    hp_indices_1 = hp.query_polygon(Nside, vertices_1, inclusive=True)
    hp_indices_2 = hp.query_polygon(Nside, vertices_2, inclusive=True)
    return np.union1d(hp_indices_1, hp_indices_2)

def extract_fibers_for_pair(pair, F, Nside, side_length, actual_row_idx):
    ra_LAEs = [pair['ra_1'], pair['ra_2']]
    dec_LAEs = [pair['dec_1'], pair['dec_2']]
    coord_1 = SkyCoord(ra=pair['ra_1']*u.degree, dec=pair['dec_1']*u.degree)
    coord_2 = SkyCoord(ra=pair['ra_2']*u.degree, dec=pair['dec_2']*u.degree)
    dir_vector = np.array([coord_2.ra.degree - coord_1.ra.degree, coord_2.dec.degree - coord_1.dec.degree])
    dir_vector /= np.linalg.norm(dir_vector) 
    perpendicular = np.array([-dir_vector[1], dir_vector[0]])
    rectangle_width = 18/3600.0  # This is parallel to the original rectangle
    rectangle_height = 10/3600.0  # This is perpendicular to the original rectangle
    rectangle_LAE1_left = get_rectangle_around_point(ra_LAEs[0], dec_LAEs[0], dir_vector, perpendicular, rectangle_width, rectangle_height, left=True)
    rectangle_LAE2_right = get_rectangle_around_point(ra_LAEs[1], dec_LAEs[1], dir_vector, perpendicular, rectangle_width, rectangle_height)
    rect_region = setup_healpix_region(pair, side_length, Nside)
    fibers_tab = gather_fibers(rect_region, F)
    #print(len(fibers_tab))
    ra_LAEs = [pair['ra_1'], pair['ra_2']]
    dec_LAEs = [pair['dec_1'], pair['dec_2']]
    mask_LAE1 = [is_inside_rectangle(fiber['ra'], fiber['dec'], rectangle_LAE1_left) and not is_too_close_to_any_LAE(fiber['ra'], fiber['dec'], ra_LAEs, dec_LAEs, threshold_radius= 3/3600) for fiber in fibers_tab]
    mask_LAE2 = [is_inside_rectangle(fiber['ra'], fiber['dec'], rectangle_LAE2_right) and not is_too_close_to_any_LAE(fiber['ra'], fiber['dec'], ra_LAEs, dec_LAEs, threshold_radius= 3/3600) for fiber in fibers_tab]
    fibers_in_LAE1_rect = fibers_tab[mask_LAE1]
    fibers_in_LAE2_rect = fibers_tab[mask_LAE2]
    fibers_tab_filtered = vstack([fibers_in_LAE1_rect, fibers_in_LAE2_rect])
    print(f"outwards fiber number for row {actual_row_idx} = {len(fibers_tab_filtered)}")
    return fibers_tab_filtered


# In[3]:
def gather_fibers(rect_region, F):
    debug_logs = []
    fibers_tab = Table()
    for hpix in rect_region:
        try:
            h_tab, h_tab_index = F.get_fib_from_hp(hpix, return_index=True)
            debug_logs.append((hpix, len(h_tab), len(h_tab_index)))
            fibers_tab = vstack([fibers_tab, h_tab])
            fibers_tab=unique(fibers_tab, keys='fiber_id')
            if len(fibers_tab) == 0:
                continue
        except Exception as e:
            debug_logs.append((hpix, 'Error', str(e)))
    amps_list = unique(fibers_tab['shotid', 'multiframe'])
    fiber_spec = Table()
    for row in amps_list:
        fib_i = get_fibers_table(row['shotid'], multiframe=row['multiframe'])
        fiber_spec = vstack([fiber_spec, fib_i])
        #print(len(fiber_spec))
    return join(fibers_tab, fiber_spec)


# In[4]:
def filter_fibers(fiber_table, z):
    wl = np.linspace(3470, 5540, 1036)
    specs = fiber_table['calfib_ffsky']
    specs = np.where(specs == 0, np.nan, specs)
    zone1_means = []
    zone2_means = []
    zone3_means = []
    valid_specs = []
    for fib in specs:
        zone1, zone2, zone3 = [], [], []
        for w, f in zip(wl, fib):
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
    zone1_avg, zone1_std = np.mean(zone1_means), np.std(zone1_means)
    zone2_avg, zone2_std = np.mean(zone2_means), np.std(zone2_means)
    zone3_avg, zone3_std = np.mean(zone3_means), np.std(zone3_means)
    valid_specs = [spec for i, spec in enumerate(valid_specs) if 
                    np.abs(zone1_means[i]-zone1_avg) <= 3*zone1_std and 
                    np.abs(zone2_means[i]-zone2_avg) <= 3*zone2_std and 
                    np.abs(zone3_means[i]-zone3_avg) <= 3*zone3_std]
    valid_spec_counter = len(valid_specs)
    if valid_spec_counter == 0:
        #print(f"Warning: No valid specs for pair with z = {z}")
        return None, None
    valid_specs = np.array(valid_specs)
    median_spec = np.nanmedian(valid_specs, axis=0)
    errs = fiber_table['calfibe']
    errs = np.where(errs == 0, np.nan, errs)
    err = np.nanmedian(errs, axis=0)
    return median_spec, err


# In[5]:
def main(row):
    input_file = 'filtered_pairs_table4.fits'
    filtered_pairs = Table.read(input_file, format='fits')
    if row is not None:
        filtered_pairs = [filtered_pairs[row]]
    F = FiberIndex('hdr3')
    Nside = 2 ** 15
    side_length = 80/3600.0
    wl = np.linspace(3470, 5540, 1036)
    for i, pair_row in enumerate(filtered_pairs):
        actual_row_idx = i + (row or 0)
        fiber_table = extract_fibers_for_pair(pair_row, F, Nside, side_length, actual_row_idx)
        z = (pair_row['z_hetdex_1'] + pair_row['z_hetdex_2']) / 2
        median_spec, err = filter_fibers(fiber_table, z)
        if median_spec is not None and err is not None:
            conv = 1e-17
            flux_density = median_spec * conv
            fluxe = err * conv
            lum, rest_wl, lum_err = SU.shift_flam_to_rest_luminosity_per_aa(z, flux_density, wl, eflux=fluxe, apply_air_to_vac=True)
            with open(f'lumpairs_outwards_multi_IFU/lum_pairs_{actual_row_idx}.pkl', 'wb') as f:
                pickle.dump({
                    'lum': lum,
                    'rest_wl': rest_wl,
                    'lum_err': lum_err
                }, f)
    F.close() 

                
# In[6]:
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process HETDEX data")
    parser.add_argument('row', type=int, help='Row to process.')
    args = parser.parse_args()
    main(row=args.row)
