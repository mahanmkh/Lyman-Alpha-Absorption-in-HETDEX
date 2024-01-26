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
from elixer import global_config as G

G.GLOBAL_LOGGING = True
G.HDR_Version='4'

wl = np.linspace(3470,5540,1036)
wl_vac=SU.air_to_vac(wl)
conv=1e-17

def is_too_close_to_any_LAE(ra_fiber, dec_fiber, ra_LAEs, dec_LAEs, threshold_radius):
    for ra_LAE, dec_LAE in zip(ra_LAEs, dec_LAEs):
        dist = SkyCoord(ra=ra_fiber*u.degree, dec=dec_fiber*u.degree).separation(SkyCoord(ra=ra_LAE*u.degree, dec=dec_LAE*u.degree)).degree
        if dist < threshold_radius:
            return True
    return False


def get_rectangle_around_point(ra, dec, dir_vector, perpendicular, width, height):
    """Generate rectangle vertices around a given RA and Dec."""
    return [
        [ra - width * perpendicular[0] - height * dir_vector[0], dec - width * perpendicular[1] - height * dir_vector[1]], # Bottom left
        [ra + width * perpendicular[0] - height * dir_vector[0], dec + width * perpendicular[1] - height * dir_vector[1]], # Bottom right
        [ra + width * perpendicular[0] + height * dir_vector[0], dec + width * perpendicular[1] + height * dir_vector[1]], # Top right
        [ra - width * perpendicular[0] + height * dir_vector[0], dec - width * perpendicular[1] + height * dir_vector[1]]  # Top left
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
    rectangle_width = 10/3600.0  # This is parallel to the original rectangle
    rectangle_height = 5/3600.0  # This is perpendicular to the original rectangle
    rectangle_LAE1 = get_rectangle_around_point(ra_LAEs[0], dec_LAEs[0], dir_vector, perpendicular, rectangle_width, rectangle_height)
    rectangle_LAE2 = get_rectangle_around_point(ra_LAEs[1], dec_LAEs[1], dir_vector, perpendicular, rectangle_width, rectangle_height)
    rect_region = setup_healpix_region(pair, side_length, Nside)
    fibers_tab = gather_fibers(rect_region, F)
    #print(len(fibers_tab))
    ra_LAEs = [pair['ra_1'], pair['ra_2']]
    dec_LAEs = [pair['dec_1'], pair['dec_2']]
    
    mask_LAE1 = [is_inside_rectangle(fiber['ra'], fiber['dec'], rectangle_LAE1) and not is_too_close_to_any_LAE(fiber['ra'], fiber['dec'], ra_LAEs, dec_LAEs, threshold_radius= 3/3600) for fiber in fibers_tab]
    mask_LAE2 = [is_inside_rectangle(fiber['ra'], fiber['dec'], rectangle_LAE2) and not is_too_close_to_any_LAE(fiber['ra'], fiber['dec'], ra_LAEs, dec_LAEs, threshold_radius= 3/3600) for fiber in fibers_tab]
    fibers_in_LAE1_rect = fibers_tab[mask_LAE1]
    fibers_in_LAE2_rect = fibers_tab[mask_LAE2]
    fibers_tab_filtered = vstack([fibers_in_LAE1_rect, fibers_in_LAE2_rect])
    print(f"perpendicular fiber number for row {actual_row_idx} = {len(fibers_tab_filtered)}")
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


def filter_fibers(fiber_table,z,seeing,throughput,shot1,shot2):
    specs = fiber_table['calfib_ffsky']#calfib_ffsky_rescor for hdr4
    specs = np.where(specs == 0, np.nan, specs)
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
        return None, None, None, None, 0, 0
    else:
        valid_errs = fiber_table['calfibe'][valid_mask]
        valid_errs = np.where(valid_errs == 0, np.nan, valid_errs)
        err = np.nanmedian(valid_errs, axis=0)
        valid_specs[np.isnan(valid_errs)] = np.nan
        median_spec = np.nanmedian(valid_specs, axis=0)
        #res1, res_err1, res_con1= SU.get_empty_fiber_residual(hdr='4', rtype='trim', shotid=shot1, seeing=seeing,response=throughput,ffsky=True,persist=True)
        #res2, res_err2, res_cont2= SU.get_empty_fiber_residual(hdr='4', rtype='trim', shotid=shot2, seeing=seeing,response=throughput,ffsky=True,persist=True)#add_rescor=True for hdr4 #add the average of the two shot ids maybe?
        #res = (res1 + res2) / 2
        final_spec = median_spec #- res
    return final_spec, err


# In[5]:
def main(row):
    input_file = 'close_pairs_HDR4_10to40arcsec.fits'
    filtered_pairs = Table.read(input_file, format='fits')
    if row is not None:
        filtered_pairs = [filtered_pairs[row]]
    F = FiberIndex("hdr4")
    Nside = 2 ** 15
    side_length = 70/3600.0
    for i, pair_row in enumerate(filtered_pairs):
        actual_row_idx = i + (row or 0)
        fiber_table = extract_fibers_for_pair(pair_row, F, Nside, side_length, actual_row_idx)
        z = (pair_row['z_hetdex_1'] + pair_row['z_hetdex_2']) / 2
        seeing = (pair_row['fwhm_1'] + pair_row['fwhm_2']) / 2
        throughput = (pair_row['throughput_1'] + pair_row['throughput_2']) / 2
        shot1= pair_row['shotid_1']
        shot2= pair_row['shotid_2']                
        final_spec, err = filter_fibers(fiber_table,z,seeing,throughput,shot1,shot2)
        if final_spec is not None and err is not None:
            flux_density = final_spec * conv
            fluxe = err * conv
            lum, rest_wl, lum_err = SU.shift_flam_to_rest_luminosity_per_aa(z, flux_density, wl, eflux=fluxe, apply_air_to_vac=True)
            with open(f'zstore/lum_pairs_{actual_row_idx}.pkl', 'wb') as f:
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
