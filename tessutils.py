# -*- coding: utf-8 -*-
"""
tessutils.py - Luke Bouma (bouma.luke@gmail.com) - May 2019

Miscellaneous utilities used for the TESS reductions.

====================
parallel_bkgd_subtract: estimate sky background (with a median filter)

parallel_plot_median_filter_quad: plot sky bkgd.

    plot_median_filter_quad

parallel_mask_saturated_stars: mask saturated stars given saturation level

    mask_saturated_stars_worker

parallel_mask_dquality_flag_frames: mask entire frames based on DQUALITY flag
    (OUTDATED)

    mask_dquality_flag_frame

parallel_trim_get_single_extension: get single extension image, trim to remove
    virtual columns, append "PROJID" header keywork

    from_CAL_to_fitsh_compatible
    (deprecated) from_ete6_to_fitsh_compatible

are_known_planets_in_field: precursor to measuring known planet properties is
    knowing if there are any

measure_known_planet_SNR: if there are known planets in this frame, estimate
    their SNR through a max-likelihood trapezoidal transit model

    _measure_planet_snr

read_object_reformed_catalog: get objectid, ra, dec from reformed catalog

read_object_catalog: read the Gaia .catalog file into a pandas DataFrame

read_tess_txt_lightcurve: Read grcollect dumped lightcurve file into a
    dictionary (which can then be put into a FITS table).

plot_lc_positions: Given list of lcpaths, plot their on-chip positions.

mask_orbit_start_and_end: Ignore the times near the edges of orbits.
"""
from __future__ import division, print_function

##########################################

import aperturephot as ap
import imageutils as iu
import shared_variables as sv

import os, pickle, re, shutil
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.colors as colors
from glob import glob
from parse import search

from astropy.io import fits
from astroquery.mast import Catalogs
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

from numpy import array as nparr, all as npall

from datetime import datetime
import multiprocessing as mp

from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const
from astropy.io import ascii

from lcstatistics import plot_raw_epd_tfa

from astrobase.periodbase import kbls
from astrobase import lcfit
from astrobase.varbase.transits import get_transit_times
from astrobase import lcmath
from astrobase.lcfit.utils import make_fit_plot

from scipy.ndimage import median_filter
from scipy.ndimage.filters import gaussian_filter
from copy import deepcopy

##########################################

DEBUG = False

SATURATIONMASKCMD = ('fiign {fitsfile} -o {outfitsfile} '
                     '-s {saturationlevel} --an')

DQUALITYMASKCMD = ('fiign {fitsfile} -o {outfitsfile} '
                   '-s {minimumreadout:.10f} --ignore-nonpositive')

# bad times quoted in data release notes are all in TJD (no barycentric
# correction applied). this is b/c they're from POC, not SPOC.
# the following were manually copied in, and are saved for posterity:
badtimewindows = [
    (1347, 1349.5), # sector 1, coarse pointing
    (1338.52153, 1339.65310), # sector 1 downlink, btwn orbits 9->10
    (1367.15347, 1368.59406), # sector 2 downlink, btwn orbits 11->12
    (1382.03986, 1385.896635), # sector 3, orbit 13 start
    (1395.47997, 1396.60497), # sector 3 downlink, btwn orbits 13->14
    (1394.00000, 1396.60497), # sector 3 empirically bad scattered light
    (1406.29246, 1409.388295), # sector 3, orbit 14 end
    (1418.53690, 1421.211685), # sector 4 instrument anomaly
    (1421.21000, 1424.54897), # sector 4 empirically "bad" time, + dl orbits 15->16
    (1465.212615, 1468.269985), # sector 6 PRF time
    (1477.01998, 1478.11304), # sector 6 downlink, btwn orbits 19->20
    (1503.03803, 1504.69775), # sector 7 downlink, btwn orbits 21->22
    (1517.34150, 1517.39566), # sector 8 orbit 23 camera 1 attitude control
    (1529.06510, 1535.00264), # sector 8 gap + attitude + instr anomaly 23->24
    (1543.21648, 1543.75080), # sector 9 camera 1 attitude control issues
    (1555.54148, 1557.00080), # sector 9 downlink, btwn orbits 25->26
    (1569.43176, 1570.87620), # sector 10 orbit 27 "strong scattered light"
    (1581.78453, 1584.72342), # sector 10 downlink + "strong scattered light"
    (1609.69425, 1610.77620), # sector 11 downlink, btwn orbits 29->30
    (1638.99562, 1640.03312), # sector 12 downlink, btwn orbits 31->32
    (1667.69004, 1668.61921), # sector 13 downlink, btwn orbits 33->34
    (1696.38865, 1697.33865), # sector 14 downlink, btwn orbits 35->36
    (1723.8584, 1724.9348), # sector 15 downlink, btwn orbits 37->38
    (1750.3584, 1751.6501), # sector 16 downlink, btwn orbits 39->40
    (1776.2959, 1777.7230), # sector 17 downlink, btwn orbits 41->42
    (1802.4418, 1803.4418), # sector 18 downlink, btwn orbits 43->44
    (1827.9939, 1828.9591), # sector 19 downlink, btwn orbits 45->46
    (1854.8584, 1856.3897), # sector 20 downlink, btwn orbits 47->48
    (1883.9939, 1884.9175), # sector 21 downlink, btwn orbits 49->50
    (1912.5043, 1913.5668), # sector 22 downlink, btwn orbits 51->52
    (1940.8723, 1941.8307), # sector 23 downlink, btwn orbits 53->54
    (1968.3480, 1969.2647), # sector 24 downlink, btwn orbits 55->56
    (1995.6362, 1996.9105), # sector 25 downlink, btwn orbits 57->58
    (2022.1918, 2023.1084), # sector 26 downlink, btwn orbits 59->60
    (2048.1293, 2049.1466), # sector 27 downlink, btwn orbits 61->62
    (2073.9626, 2075.1675), # sector 28 downlink, btwn orbits 63->64
    (2101.3723, 2114.4348), # sector 29 downlink, btwn orbits 65->66
]
# and we can append to this automatically using the orbit times from:
# https://archive.stsci.edu/missions/tess/doc/tess_drn/
# (downloaded on the date noted below... I'll reach out to Scott Fleming to
# see whether it'll be updated past S29)
from paths import DATADIR
badtime_df = pd.read_csv(
    os.path.join(DATADIR, '20210712_tess_orbit_times_by_sector.csv'),
    comment='#'
)
for ix in range(37, 65, 2):
    thistuple = (
        float(badtime_df[badtime_df.Orbit == ix]['End TJD']),
        float(badtime_df[badtime_df.Orbit == ix+1]['Start TJD'])
    )
    badtimewindows.append(thistuple)
# append the mid-sector downlink gaps between orbits 37->38, 39->40, 41->42,
# etc. NOTE that if the downlink schedule ever changes, and the parity changes,
# the opposite gap will need to be done.  this continues through orbit 65->66,
# which finished 2020-09-21 (sector 29).


def mask_dquality_flag_frame(task):
    """
    Mask the entire frame if dquality flag from SPOC is raised. Overwrites
    image with fitsh-aware mask in the header. Note the mask is not
    actually applied to the pixel values until we need to visualize; the
    mask is metadata.
    """

    fitsname, flagvalues = task

    # open the header, check if the quality flag matches the passed value.
    data, hdr = iu.read_fits(fitsname, ext=0)

    isbadtime = False
    if -1 in flagvalues:
        # get frame time in TJD. if its within any known bad times for TESS,
        # then set isbadtime to True, and mask the frame.

        # get frametime in TJD = BTJD - LTT_corr
        tjd = hdr['TSTART'] - hdr['BARYCORR']

        for window in badtimewindows:
            if tjd > min(window) and tjd < max(window):
                isbadtime = True

    # if it matches, mask out everything. otherwise, do nothing.
    if hdr['DQUALITY'] in flagvalues or isbadtime:

        # if you pass `fiign` "0" as the saturation value, you don't get a
        # correct mask. so we mask out non-positive values, and set the
        # saturation value below the lowest positive value.
        minabsdata = np.nanmin(np.abs(data[data>0]))
        minimumreadout = minabsdata - minabsdata/2.

        minimumreadout = max([minimumreadout, 1e-10])
        cmdtorun = DQUALITYMASKCMD.format(fitsfile=fitsname, outfitsfile=fitsname,
                                          minimumreadout=minimumreadout)

        returncode = os.system(cmdtorun)

        if returncode == 0:
            print('%sZ: dquality %s (or badtime). masked %s w/ minread %s' %
                  (datetime.utcnow().isoformat(), repr(hdr['DQUALITY']),
                   fitsname, repr(minimumreadout)))
            return fitsname
        else:
            print('ERR! %sZ: DQUALITY mask construction failed for %s' %
                  (datetime.utcnow().isoformat(), fitsname))
            return None

    else:
        return 1


def parallel_mask_dquality_flag_frames(fitslist, flagvalues=[-1,4,32,36],
                                       nworkers=16, maxworkertasks=1000):
    '''
    OUTDATED. USE parallel_move_badframes INSTEAD.

    Append mask to fits header of an ENTIRE CALIBRATED FRAME if it matches any
    of the passed data quality flag value.

    See e.g., EXP-TESS-ARC-ICD-TM-0014 for a description of the flag values.

    Kwargs:
        flagvalues (list): list of integers that tells you something about the
        data quality. E.g., "32" is a "reaction wheel desaturation event", AKA
        a "momentum dump".  The value "-1" is special because it is the
        manually-imposed "Manual Exclude" flag from within pipe-trex. These
        come from a list of known bad times, which are read from the data
        release notes.
    '''

    print('%sZ: %s files to check for DQUALITY=%s in' %
          (datetime.utcnow().isoformat(), len(fitslist), repr(flagvalues)))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, flagvalues) for x in fitslist]

    # fire up the pool of workers
    results = pool.map(mask_dquality_flag_frame, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    print('%sZ: finished checking for DQUALITY=%s' %
          (datetime.utcnow().isoformat(), flagvalues))

    return {result for result in results}


def move_badframe(task):
    """
    Move the entire frame to "badframe" subdirectory, if it matches particular
    quality flags or time windows.
    """

    fitsname, flagvalues = task

    # open the header, check if the quality flag matches the passed value.
    hdr = iu.read_fits_header(fitsname, ext=0)

    isbadtime = False
    if -1 in flagvalues:
        # get frame time in TJD. if its within any known bad times for TESS,
        # then set isbadtime to True, and mask the frame.

        # get frametime in TJD = BTJD - LTT_corr
        tjd = hdr['TSTART'] - hdr['BARYCORR']

        for window in badtimewindows:
            if tjd > min(window) and tjd < max(window):
                isbadtime = True

    # if it matches, move. otherwise, do nothing.
    if hdr['DQUALITY'] in flagvalues or isbadtime:

        dstdir = os.path.join(os.path.dirname(fitsname),'badframes')
        dstpath = os.path.join(dstdir,os.path.basename(fitsname))

        shutil.move(fitsname, dstpath)
        print('{}Z: BADFRAME moved {} -> {}'.format(
            datetime.utcnow().isoformat(), fitsname, dstpath))
        return dstpath

    else:
        return 1


def verify_badframe_move(fitslist, flagvalues, max_frac_badframes=0.25):
    """
    max_frac_badframes: if you have more than this fraction of the FFIs that
    are "badframes", you should be worried!
    """

    dqualitys = []
    isbadtimes = []
    tjds = []

    astromkeys = ['CRVAL1', 'CRVAL2', 'A_DMAX', 'B_DMAX']
    astromdict = {}
    for k in astromkeys:
        astromdict[k] = []

    for f in fitslist:

        # open the header, check if the quality flag matches the passed value.
        hdr = iu.read_fits_header(f, ext=0)

        dquality = hdr['DQUALITY']
        dqualitys.append(dquality)

        for k in astromkeys:
            try:
                astromdict[k].append(hdr[k])
            except KeyError:
                astromdict[k].append(0)

        # get frametime in TJD = BTJD - LTT_corr.
        # if its within any known bad times for TESS, then set isbadtime to
        # True, and mask the frame.
        tjd = hdr['TSTART'] - hdr['BARYCORR']
        tjds.append(tjd)

        isbadtime = False
        if -1 in flagvalues:
            for window in badtimewindows:
                if tjd > min(window) and tjd < max(window):
                    isbadtime = True

        if dquality in flagvalues or isbadtime:

            isbadtimes.append(True)

        else:

            isbadtimes.append(False)

    df = pd.DataFrame()

    df['fitslist'] = fitslist
    df['isbadtime'] = isbadtimes
    df['DQUALITY'] =  dqualitys
    df['tjd'] = tjds
    for k in astromkeys:
        df[k] = astromdict[k]

    df = df.sort_values(by='fitslist')
    camera = hdr['CAMERA']
    ccd = hdr['CCD']
    outstr = 'cam{}_ccd{}'.format(camera, ccd)
    outpath = os.path.join(os.path.dirname(fitslist[0]),
                           'verify_badframe_move_{}.csv'.format(outstr))
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))

    if not len(df[df['isbadtime']])/len(df) < max_frac_badframes:
        errmsg = (
            'Got {} badtimes of {} FFIs. Too large fraction!'.
            format(len(df[df['isbadtime']]), len(df) )
        )

        # sector 3 has reaction wheel testing.
        # sector 8 has an instrument failure.
        # sector 12, camera 1 has extended scattered light (see Kruse's videos)
        # sector 14 and sector 15, camera 1 also has extended scattered light
        badsectors = ['s0003', 's0008', 's0012', 's0014', 's0015']

        raise_error = True
        for badsector in badsectors:
            if badsector in f:
                print('WRN! {}'.format(errmsg))
                raise_error = False

        if raise_error:
            raise AssertionError(errmsg)

    #
    # plot DQUALITY vs TIME
    #
    plt.close('all')
    f, axs = plt.subplots(nrows=2, ncols=1, figsize=(6,6))
    axs[0].scatter(df['tjd'], df['DQUALITY'])
    axs[0].set_ylabel('dquality')
    axs[1].scatter(df['tjd'], df['isbadtime'])
    axs[1].set_ylabel('is_omitted')
    axs[1].set_xlabel('tjd = BJTD - barycorr')
    outpath = os.path.join(os.path.dirname(fitslist[0]),
                           'verify_badframe_move_{}.png'.format(outstr))
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))

    #
    # plot ASTROMETRY QUANTITIES vs TIME
    # astromkeys = ['CRVAL1', 'CRVAL2', 'A_DMAX', 'B_DMAX']
    #
    plt.close('all')
    f, axs = plt.subplots(nrows=4, ncols=1, figsize=(6,9))

    for ix, k in enumerate(astromkeys):
        axs[ix].scatter(df['tjd'], df[k])
        axs[ix].set_ylabel(k)

    axs[-1].set_xlabel('tjd = BJTD - barycorr')
    outpath = os.path.join(os.path.dirname(fitslist[0]),
                           'verify_frame_astrometric_quantities_{}.png'.
                           format(outstr))
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(outpath))



def parallel_move_badframes(fitslist,
                            flagvalues=[-1,4,32,36,2048,2080,2052,2084],
                            nworkers=16,
                            maxworkertasks=1000):
    '''
    Move the entire frame to "badframe" subdirectory, if it matches particular
    quality flags or time windows.

    Move frame to badframes dir it matches any of the passed data quality flag
    value.  See e.g., EXP-TESS-ARC-ICD-TM-0014 for a description of the flag
    values.

    Kwargs:
        flagvalues (list): list of integers that tells you something about the
        data quality.

            "32" is "reaction wheel desaturation event", AKA "momentum dump".

            "4" is that the spacecraft is in corase point mode.

            "2048" is scattered light. empirically, it's _bad_ scattered light.

            "-1" is special because it is the manually-imposed "Manual Exclude"
            flag from within pipe-trex. These come from a list of known bad
            times, which are read from the data release notes.
    '''

    print('%sZ: %s files to check for DQUALITY=%s in (BADFRAME MOVE)' %
          (datetime.utcnow().isoformat(), len(fitslist), repr(flagvalues)))

    verify_badframe_move(fitslist, flagvalues, max_frac_badframes=0.25)

    # omit the frames.
    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, flagvalues) for x in fitslist]

    # fire up the pool of workers
    results = pool.map(move_badframe, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    print('%sZ: finished checking for DQUALITY=%s (BADFRAME MOVE)' %
          (datetime.utcnow().isoformat(), flagvalues))

    return {result for result in results}


def mask_saturated_stars_worker(task):
    """
    Mask the saturated stars using the calibrated images. Overwrites image with
    fitsh-aware mask in the header. Note the mask is not actually applied to
    the pixel values until we need to visualize; the mask is metadata.

    At first thought, it would be better to do this with raw images because
    once the flat-field is applied, saturated stars do not have a constant
    value.  However inspection of the raw images shows that the saturated stars
    do not have constant values there either. So it's easier to first trim, and
    then apply this.
    """

    fitsname, saturationlevel = task

    cmdtorun = SATURATIONMASKCMD.format(fitsfile=fitsname, outfitsfile=fitsname,
                                        saturationlevel=saturationlevel)

    returncode = os.system(cmdtorun)

    if DEBUG:
        print(cmdtorun)

    if returncode == 0:
        print('%sZ: appended saturated stars mask %s' %
              (datetime.utcnow().isoformat(), fitsname))
        return fitsname
    else:
        print('ERR! %sZ: saturated star mask construction failed for %s' %
              (datetime.utcnow().isoformat(), fitsname))
        return None


def parallel_mask_saturated_stars(fitslist, saturationlevel=65535, nworkers=16,
                                  maxworkertasks=1000):
    '''
    Append mask of saturated stars to fits header using the calibrated TESS
    images, by applying a constant saturation level.

    Kwargs:
        saturationlevel (int): 2**16 - 1 = 65535 is a common choice. This is
        the number that is used to mask out cores of saturated stars.
    '''

    print('%sZ: %s files to mask saturated stars in' %
          (datetime.utcnow().isoformat(), len(fitslist)))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, saturationlevel) for x in fitslist]

    # fire up the pool of workers
    results = pool.map(mask_saturated_stars_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {result for result in results}


def parallel_trim_get_single_extension(fitslist, outdir, projid, nworkers=16,
                                       maxworkertasks=1000):
    '''
    see docstring for from_CAL_to_fitsh_compatible
    '''

    print('%sZ: %s files to trim and get single extension' %
          (datetime.utcnow().isoformat(), len(fitslist)))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    path_exists = []
    for ix, fitsname in enumerate(fitslist):
        outname = (
            os.path.join(outdir, os.path.basename(
                fitsname.replace('-s_ffic.fits', '_cal_img.fits'))
            )
        )
        if os.path.exists(outname):
            path_exists.append(1)
        else:
            path_exists.append(0)

    path_exists = nparr(path_exists)

    if npall(path_exists):
        print(
            'found all {:d} fitsh-compatible files, continuing'.
            format(len(path_exists))
        )
        return 0

    else:
        tasks = [(x, outdir, projid) for x in fitslist]

        # fire up the pool of workers
        results = pool.map(from_CAL_to_fitsh_compatible, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return 1


def from_CAL_to_fitsh_compatible(task):
    '''
    This function takes a calibrated image from MAST and turns it into a
    fitsh-compatible image. This is done BEFORE constructing a mask for the
    image using tessutils.mask_saturated_stars

    This means:
        * get single extension (omit the uncertainty map, for now).
        * trim to remove virtual columns
        * append "PROJID" header keywork.

    Arg:
        task: tuple of (fitspath, outdir), where `fitspath` is a calibrated
        full frame image from MAST. This means its filename should end with
        "-s_ffic.fits".  (Each image is a single CCD, with 4 subarrays).
        outdir (str): directory to write the files to.

    Returns:
        nothing.
    '''
    fitsname, outdir, projid = task

    if fitsname.split('-')[-1] != 's_ffic.fits':
        raise AssertionError('expected calibrated FFI from MAST.')

    outname = (
        os.path.join(outdir, os.path.basename(
            fitsname.replace('-s_ffic.fits', '_cal_img.fits')))
    )

    try:
        data, hdr = iu.read_fits(fitsname, ext=1)
    except TypeError as e:
        print(e)
        print('got read error for {}'.format(fitsname))
        raise TypeError

    # FITS header is 1-based counting, but python is 0-based. To convert
    # from FITS->python from slicing you need to subtract 1.
    rows = hdr['SCIROWS']-1 # row start.
    rowe = hdr['SCIROWE']-1+1 # row end (inclusive)
    cols = hdr['SCCSA']-1 # col start
    cole = hdr['SCCED']-1+1 # col end (inclusive)

    trim = data[slice(rows,rowe),slice(cols,cole)]

    hdr['PROJID'] = projid
    hdr.comments['PROJID'] = 'trex identifier for parameters in this run'

    assert trim.shape == (2048, 2048)

    if not os.path.exists(outname):
        fits.writeto(outname, trim, header=hdr)
        print('{:s} Wrote {:s} to {:s}'.format(
            datetime.utcnow().isoformat(), fitsname, outname)
        )
    else:
        print('{:s} ERR! Found {:s} already exists'.format(
            datetime.utcnow().isoformat(), outname)
        )



def from_ete6_to_fitsh_compatible(fitslist, outdir, projid=42):
    '''
    This function takes a list of ETE6 reduced images and turns them into
    fitsh-compatible images.  Using it is a bad idea, because you shouldn't be
    using ETE6 any more anyway.
    '''

    for fitsname in fitslist:
        from_CAL_to_fitsh_compatible((fitsname, outdir, projid))


def are_known_planets_in_field(ra_center, dec_center, outname, use_NEA=False,
                               use_alerts=False, use_tev=False, use_exofop=True):
    """
    Given a field center, find which known planets are on chip.
    Dependencies: tessmaps, astroquery

    Args:
        ra_center, dec_center (floats): coordinate of field center in degrees

        outname (str): path to which csv file with details of HJs on chip will
        be written

        useNEA (bool): whether to use NASA exoplanet archive to grab the known
        planet properties (in particular, known HJs), or not.

        use_alerts (bool): whether to use the "alert" file, from
        https://archive.stsci.edu/prepds/tess-data-alerts/index.html, to get
        candidate planet host (TOI) properties.

        use_tev (bool): whether to use the TEV TOI list from MIT (i.e.,
        download it directly).  Assumes you are connected to the internet.

    Returns:
        True if any HJs on chip, else False.
    """

    # at most one data getting location should be used
    assert np.sum([use_NEA, use_alerts, use_tev, use_exofop]) == 1

    from tessmaps import get_time_on_silicon as gts

    cam_dirn = SkyCoord(ra_center*u.deg, dec_center*u.deg, frame='icrs')
    cam_tuple = (cam_dirn.barycentrictrueecliptic.lat.value,
                 cam_dirn.barycentrictrueecliptic.lon.value)

    if use_NEA:
        eatab = NasaExoplanetArchive.get_confirmed_planets_table()

        is_jup = (eatab['pl_radj'] > 0.4*u.Rjup)
        is_hot = (eatab['pl_orbper'] < 10*u.day)
        is_hj = is_jup & is_hot
        hj_coords = eatab[is_hj]['sky_coord']

        pl_onchip = gts.given_one_camera_get_stars_on_silicon(
            hj_coords, cam_tuple, withgaps=False)

    elif use_alerts:
        # latest version available
        df = pd.read_csv(
            '/home/lbouma/proj/cdips-pipeline/data/'
            'csv-file-toi-plus-catalog-2019-12-05.csv',
            comment='#'
        )
        kp_coords = SkyCoord(nparr(df['TIC Right Ascension'])*u.deg,
                             nparr(df['TIC Declination'])*u.deg,
                             frame='icrs')

        pl_onchip = gts.given_one_camera_get_stars_on_silicon(
            kp_coords, cam_tuple, withgaps=False)

    elif use_tev:
        # download latest TOI list to local memory; use that instead.
        df = pd.read_csv(
            'https://tev.mit.edu/data/collection/193/csv/6/',
            sep=',', comment='#'
        )
        kp_coords = SkyCoord(nparr(df['TIC Right Ascension'])*u.deg,
                             nparr(df['TIC Declination'])*u.deg,
                             frame='icrs')
        pl_onchip = gts.given_one_camera_get_stars_on_silicon(
            kp_coords, cam_tuple, withgaps=False)

    elif use_exofop:
        # download latest TOI list to local memory; use that instead.
        df = pd.read_csv(
            'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv',
            sep=',', comment='#'
        )
        kp_coords = SkyCoord(nparr(df['RA']),
                             nparr(df['Dec']),
                             frame='icrs', unit=(u.hourangle, u.deg))
        pl_onchip = gts.given_one_camera_get_stars_on_silicon(
            kp_coords, cam_tuple, withgaps=False)

    else:
        raise NotImplementedError('use_alerts or use_NEA must be true.')


    if np.any(pl_onchip) and use_NEA:
        desiredcols = ['pl_hostname', 'pl_letter', 'pl_name', 'pl_orbper',
                       'pl_radj', 'st_optmag', 'gaia_gmag','sky_coord']

        outtab = eatab[is_hj][desiredcols]
        outtab = outtab[pl_onchip.astype(bool)]

        ascii.write(outtab, output=outname, format='ecsv',
                    overwrite=True)
        print('wrote {}'.format(outname))

        return True

    elif np.any(pl_onchip) and (use_alerts or use_tev or use_exofop):

        outdf = df[pl_onchip.astype(bool)]
        outdf.to_csv(outname, index=False)
        print('wrote {}'.format(outname))

        return True

    else:
        return False


def are_clusters_in_field(ra_center, dec_center, outname, dist_cut=2000,
                          sectornum=1):
    """
    Given a field center, find which clusters from Kharchenko+ 2013 are on chip.
    Limit to clusters within some distance, say 2kpc.
    Dependencies: tessmaps

    Args:
        ra_center, dec_center (floats): coordinate of field center in degrees

        outname (str): path to which csv file with details of HJs on chip will
        be written

    Kwargs:

        dist_cut (float): number of parsecs beyond which we are not interested

        sectornum (int): 1-based "science sector" count.

    Returns:
        True if any HJs on chip, else False.
    """

    from tessmaps import get_time_on_silicon as gts

    # get the coordinates and properties of the clusters
    k13path = ('/home/lbouma/proj/pipe-trex/data/'
               'Kharchenko_2013_MWSC_tess_sectors.csv')
    k13 = pd.read_csv(k13path)

    sel_clusters = (k13['sector_{:d}'.format(sectornum-1)]==1)
    sel_clusters &= (k13['d'] < dist_cut)

    df = k13[sel_clusters]
    desiredcols = ['MWSC', 'Name', 'Type', 'n_Type', 'RAJ2000', 'DEJ2000',
                   'r0', 'r1', 'r2', 'N1sr0', 'N1sr1', 'N1sr2', 'd', 'logt',
                   'ra', 'dec']
    df = df[desiredcols]

    # get coordinates
    cluster_coords = SkyCoord(nparr(df['ra'])*u.deg, nparr(df['dec'])*u.deg,
                              frame='icrs')

    ##########################################
    cam_dirn = SkyCoord(ra_center*u.deg, dec_center*u.deg, frame='icrs')
    cam_tuple = (cam_dirn.barycentrictrueecliptic.lat.value,
                 cam_dirn.barycentrictrueecliptic.lon.value)

    ##########################################
    cluster_onchip = gts.given_one_camera_get_stars_on_silicon(
        cluster_coords, cam_tuple, withgaps=False)

    if np.any(cluster_onchip):

        outdf = df
        outdf = outdf[cluster_onchip.astype(bool)]

        outdf.to_csv(outname, index=False)
        print('wrote {}'.format(outname))

        return True

    else:
        return False


def cluster_cutout_jpg_worker(task):

    (bkgdsubfile, subconvfile, calfitsfile, wcsfile, fitsdir,
     cname, ctype, nstar, dist, logt,
     xmin, xmax, ymin, ymax, clusterdistancecutoff
    ) = task

    outdir = os.path.join(fitsdir, 'CUT-'+str(cname))
    outcalpath = os.path.join(
        outdir, 'CUT-{}_{}_CAL.jpg'.
        format(cname,
               os.path.basename(calfitsfile.replace('.fits','')))
    )
    outbkgdsubpath = os.path.join(
        outdir, 'CUT-{}_{}_BKGDSUB.jpg'.
        format(cname,
               os.path.basename(bkgdsubfile.replace('.fits','')))
    )
    outsubpath = os.path.join(
        outdir, 'CUT-{}_{}_SUBCONV.jpg'.
        format(cname,
               os.path.basename(subconvfile.replace('.fits','')))
    )

    if dist > clusterdistancecutoff:
        print('skipping {}, d>{}pc'.
              format(cname, clusterdistancecutoff))
        return 1

    if (os.path.exists(outcalpath) and
        os.path.exists(outsubpath) and
        os.path.exists(outbkgdsubpath)
    ):
        return 1

    thisctype, thisnstar, thislogt = str(ctype), int(nstar), float(dist)

    annotatestr = '{:s}-{:s}, {:s}*, logt={:s}, d={:.1f}'.format(
        cname, ctype, repr(int(nstar)),
        '{:.1f}'.format(logt), dist/1000
    )

    # logscale of the CAL image (no background subtraction)
    img, _ = iu.read_fits(calfitsfile, ext=0)
    trimmed_img = img[ymin:ymax, xmin:xmax]
    iu.mplplot_logscale_img_w_colorbar(trimmed_img, outcalpath,
                                       cmap='binary_r', vmin=10, vmax=1000,
                                       titlestr=annotatestr)

    # diffscale image of the BKGDSUB image
    img, _ = iu.read_fits(bkgdsubfile, ext=0)
    trimmed_img = img[ymin:ymax, xmin:xmax]
    iu.mplplot_diffscale_img_w_colorbar(trimmed_img, outbkgdsubpath,
                                        cmap='RdBu_r', vmin=-1000, vmax=1000,
                                        titlestr=annotatestr)

    # diffscale image of the SUBCONV image
    img, _ = iu.read_fits(subconvfile, ext=0)
    trimmed_img = img[ymin:ymax, xmin:xmax]
    iu.mplplot_diffscale_img_w_colorbar(trimmed_img, outsubpath,
                                        cmap='RdBu_r', vmin=-1000, vmax=1000,
                                        titlestr=annotatestr)

    return 1


def parallel_cluster_cutout_jpgs(bkgdsubfiles, subconvfiles, calimgfiles,
                                 wcsfiles, fitsdir, cname, ctype, nstar, dist,
                                 logt, xmin, xmax, ymin, ymax, clusterdistancecutoff,
                                 nworkers=16, maxworkertasks=1000):
    """
    cutout boxes from the images, to make movies of individual clusters.

    we use a fixed box (xmin,xmax,ymin,ymax) because determining specific boxes
    on each frame leads to jitter over the movie.
    """

    print('%sZ: %s files to make cluster cutouts for %s' %
          (datetime.utcnow().isoformat(), len(wcsfiles), cname))

    outdir = os.path.join(fitsdir, 'CUT-'+str(cname))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    tasks = [(
        x, y, z, w,
        fitsdir, cname, ctype, nstar, dist, logt,
        xmin, xmax, ymin, ymax,
        clusterdistancecutoff
    )
        for x,y,z,w in zip(bkgdsubfiles, subconvfiles, calimgfiles, wcsfiles)
    ]

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(cluster_cutout_jpg_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {result for result in results}


def make_cluster_cutout_jpgs(sectornum, fitsdir, racenter, deccenter, field,
                             camera, ccd, statsdir, clusterdistancecutoff=2000,
                             nworkers=16):
    """
    cluster cutouts from three types of images:
        CAL images
        BKGDSUB images (background subtracted)
        SUBCONV images (difference image -- convolved + subtracted)

    (kw)args:
        clusterdistancecutoff (float): distance beyond which you are not making
        cutouts for clusters.
    """

    calimgdir = (
        '/nfs/phtess2/ar0/TESS/FFI/RED/sector-{:d}/cam{:d}_ccd{:d}/'.
        format(int(str(field[1:]).lstrip('0')), camera, ccd)
    )

    calimgfiles = glob(os.path.join(calimgdir, 'tess*cal_img.fits'))
    wcsfiles = glob(os.path.join(fitsdir,'*.wcs'))
    bkgdsubfiles = [w.replace('.wcs','.fits') for w in wcsfiles]
    subconvfiles = glob(os.path.join(fitsdir,'[r|n]sub-*-tess*-xtrns.fits'))

    if not (
        len(wcsfiles) == len(subconvfiles) ==
        len(bkgdsubfiles) == len(calimgfiles)
    ):
        if len(wcsfiles) - len(subconvfiles) < 10:
            # some wonky wcs's
            shiftok = np.array([os.path.exists(f.replace('.wcs','.fits'))
                                for f in wcsfiles])
            for w in np.array(wcsfiles)[~shiftok]:
                dst = os.path.join(os.path.dirname(w), 'badframes',
                                   os.path.basename(w))
                shutil.move(w,dst)
                print('BAD SHIFT moving {} -> {}'.format(w, dst))

        else:
            raise AssertionError(
                'something wrong in astrometric shift.'+
                '\nN_WCS: {:d}'.format(len(wcsfiles))+
                '\nN_calfits: {:d}'.format(len(calimgfiles))+
                '\nN_subconvfits: {:d}'.format(len(subconvfiles))+
                '\nN_bkgdsubfits: {:d}'.format(len(bkgdsubfiles))
            )


    outcsv = os.path.join(
        statsdir, '{:s}_cam{:d}_ccd{:d}_kharchenko13_clusters.csv'.
        format(field, camera, ccd)
    )

    if are_clusters_in_field(racenter, deccenter, outcsv, sectornum=sectornum):
        pass
    else:
        return -1

    # columns: Index([u'MWSC', u'Name', u'Type', u'n_Type', u'RAJ2000', u'DEJ2000',
    # u'r0', u'r1', u'r2', u'N1sr0', u'N1sr1', u'N1sr2', u'd', u'logt', u'ra',
    # u'dec'], dtype='object')
    df = pd.read_csv(outcsv)

    names = np.array(df['Name'])
    ctype = np.array(df['Type'].fillna(''))
    nstar = np.array(df['N1sr2'])
    dist = np.array(df['d'])
    logt = np.array(df['logt'])
    ras = np.array(df['ra'])
    decs = np.array(df['dec'])
    angrads = np.array(df['r2'])

    outcalmatches = glob(os.path.join( fitsdir, 'CUT*', 'CUT-*_CAL.jpg' ))
    if len(outcalmatches) > 1000:
        print('found cutouts were already made, return')
        return

    for name, ct, ns, d, age, ra, dec, r2 in zip(names, ctype, nstar, dist,
                                                 logt, ras, decs, angrads):

        outdir = os.path.join(fitsdir, 'CUT-'+str(name))
        outcalmatches = glob(
            os.path.join(outdir, 'CUT-{}*_CAL.jpg'.format(name))
        )
        if len(outcalmatches) > 10:
            print('found cuts for {}, continue'.format(name))
            continue

        ra, dec = float(ra), float(dec)
        clusterangradius = (float(r2)*u.deg).value
        boxwidth, boxheight = 5*clusterangradius, 5*clusterangradius

        radeccenter = [ra, dec, boxwidth, boxheight]

        middle_index = 200 # used to define the box pixel limits for the movie
        _img, _hdr = iu.read_fits(calimgfiles[middle_index])
        xmin,xmax,ymin,ymax = (
            iu._given_radecbox_get_xybox(wcsfiles[middle_index],
                                         calimgfiles[middle_index], None,
                                         radeccenter, _hdr, _img,
                                         do_spoc_trim_shift=True,
                                         forcesquare=True)
        )

        if not xmax - xmin > 10:
            continue
        if not ymax - ymin > 10:
            continue

        parallel_cluster_cutout_jpgs(bkgdsubfiles,
                                     subconvfiles,
                                     calimgfiles,
                                     wcsfiles,
                                     fitsdir, name, ct, ns, d, age,
                                     xmin, xmax, ymin, ymax,
                                     clusterdistancecutoff, nworkers=nworkers)


def measure_known_planet_SNR(kponchippath, projcatalogpath, lcdirectory,
                             statsdir, sectornum, minxmatchsep=3, nworkers=20,
                             use_NEA=False, use_alerts=False, use_tev=False,
                             use_exofop=True, skipepd=False):
    """
    Args:

        kponchippath (str): path to csv that tells you which HJs are expected
        to be on (or near) chip.

        projcatalogpath (str): path to projected catalog (from e.g., 2MASS or
        GAIA).

        lcdirectory (str): path to directory with lightcurves. Globs are
        constructed to search for LCs of the HJ matches.

        minxmatchsep (float): minimum separation, in arcseconds, for
        crossmatching of hot Jupiter exoarchive catalog positions to 2mass
        projected plate positions.
    """

    # at most one data getting location should be used
    assert np.sum([use_NEA, use_alerts, use_tev, use_exofop]) == 1

    minxmatchsep = minxmatchsep*u.arcsec

    if use_NEA:
        # read information table from `tessutils.are_known_planets_in_field`
        tab = ascii.read(kponchippath)

        # if it's not HAT/WASP/KELT, it's probably not a HJ that this tuning test
        # cares about. if so, scrap it.
        wantstrs = ['HAT','WASP','KELT']
        is_wanted = []
        for hn in nparr(tab['pl_hostname']):
            want = False
            for wantstr in wantstrs:
                if wantstr in hn:
                    want = True
                    break
                else:
                    pass
            is_wanted.append(want)
        is_wanted = nparr(is_wanted)

        tab = tab[is_wanted]
    elif use_alerts or use_tev:
        tab = pd.read_csv(kponchippath)
        tab['pl_name'] = tab['Full TOI ID']
        tab['pl_hostname'] = tab['TIC']
    elif use_exofop:
        tab = pd.read_csv(kponchippath)
        tab['pl_name'] = tab['TOI']
        tab['pl_hostname'] = tab['TIC ID']
    else:
        raise NotImplementedError

    # get the starids that correspond to the HJs on-chip
    projcat = read_object_reformed_catalog(projcatalogpath, isgaiaid=True,
                                           gaiafull=False)

    proj_ra, proj_dec = projcat['ra']*u.deg, projcat['dec']*u.deg

    proj_coords = SkyCoord(proj_ra, proj_dec, frame='icrs')
    if use_NEA:
        kp_coords = tab['sky_coord']
    elif use_alerts or use_tev:
        kp_coords = SkyCoord(nparr(tab['TIC Right Ascension'])*u.deg,
                             nparr(tab['TIC Declination'])*u.deg,
                             frame='icrs')
    elif use_exofop:
        kp_coords = SkyCoord(nparr(tab['RA']),
                             nparr(tab['Dec']),
                             frame='icrs', unit=(u.hourangle, u.deg))

    for colname in [
        'match_proj_ra','match_proj_dec','match_sep_arcsec'
    ]:
        tab[colname] = np.nan

    starids = []
    for kp_coord, name in zip(kp_coords, tab['pl_name']):

        seps = kp_coord.separation(proj_coords)
        projsorted = proj_coords[np.argsort(seps)]
        sepssorted = proj_coords[np.argsort(seps)]

        # now get the starids of the match
        if np.any(seps < minxmatchsep):
            sel = (seps < minxmatchsep)

            if len(sel[sel]) > 1:
                print('got {} matches'.format(len(sel[sel])))
                print('separations are {}'.format(
                    repr(seps[sel].to(u.arcsec)))
                )
                print('taking minimum separation...')

            projcatmatch = projcat[np.argmin(seps)]

            thispl = (tab['pl_name']==name)
            starids.append(projcatmatch[0].encode('ascii'))
            tab[thispl]['match_proj_ra'] = projcatmatch[1]
            tab[thispl]['match_proj_dec'] = projcatmatch[2]
            match_sep = np.min(seps).to(u.arcsec).value
            tab[thispl]['match_sep_arcsec'] = match_sep

            print('{}: got projcatalog match with separation {} arcsec'.
                  format(name, match_sep))

        else:
            starids.append(np.nan)
            print('{}: did not get projcatalog match with separation < {}'.
                  format(name, minxmatchsep))
            continue

    tab['starids'] = starids

    # check if the HJs on chip with known starids have lightcurves
    rle, ele, tle, rlcs, elcs, tlcs = [],[],[],[],[],[]
    for starid in tab['starids']:

        if starid=='nan' or pd.isnull(starid):
            rawlc = str(np.nan)
            epdlc = str(np.nan)
            tfalc = str(np.nan)
        else:
            if type(starid) == bytes:
                starid = starid.decode('utf-8')
            rawlc = glob(os.path.join(lcdirectory, starid+'_llc.fits'))
            epdlc = glob(os.path.join(lcdirectory, starid+'_llc.fits'))
            tfalc = glob(os.path.join(lcdirectory, starid+'_llc.fits'))

            for lc in [rawlc, epdlc, tfalc]:
                if len(lc) > 1:
                    raise ValueError('expected at most 1 lightcurve match')

            rawlc = rawlc[0] if len(rawlc)==1 else str(np.nan)
            epdlc = epdlc[0] if len(epdlc)==1 else str(np.nan)
            tfalc = tfalc[0] if len(tfalc)==1 else str(np.nan)

        rawlcexists = os.path.exists(rawlc)
        epdlcexists = os.path.exists(epdlc)
        tfalcexists = os.path.exists(tfalc)

        rle.append(rawlcexists)
        ele.append(epdlcexists)
        tle.append(tfalcexists)
        rlcs.append(rawlc)
        elcs.append(epdlc)
        tlcs.append(tfalc)

    tab['rawlcexists'] = nparr(rle)
    tab['epdlcexists'] = nparr(ele)
    tab['tfalcexists'] = nparr(tle)
    tab['rawlc'] = nparr(rlcs)
    tab['epdlc'] = nparr(elcs)
    tab['tfalc'] = nparr(tlcs)

    print(tab[['pl_name', 'starids', 'rawlcexists', 'epdlcexists',
               'tfalcexists']])
    print(tab['rawlc'])

    if not np.any(tab['tfalcexists']):

        print('WRN! did not find any known hot jupiters that had TFA '
              'lightcurves')

        return 0

    elif not np.any(tab['tfalcexists']) and np.any(tab['epdlcexists']):

        print('WRN! but found EPD lcs for some HJs. Perhaps TFA is failing.')

        return 0

    elif not np.any(tab['tfalcexists']) and not np.any(tab['epdlcexists']):
        print('WRN! and did not find any EPD lightcurves either.')

        return 0

    # otherwise, continue to by measuring the HJ's SNR
    for tfalc, plname in zip(
        tab['tfalc'][tab['tfalcexists']], tab['pl_name'][tab['tfalcexists']]
    ):
        _measure_planet_snr(plname, tfalc, statsdir, sectornum,
                            nworkers=nworkers, use_NEA=use_NEA,
                            use_alerts=use_alerts, skipepd=skipepd)


def _write_nan_df(plname, tfalc, snrfit_savfile):
    outdf = pd.DataFrame({'plname':plname,
                          'trapz_snr':np.nan,
                          'trapz_transitdepth':np.nan,
                          'oot_rms':np.nan,
                          'npoints_in_transit':np.nan,
                          'ntransits':np.nan,
                          'tfalc':tfalc}, index=[0])
    outdf.to_csv(snrfit_savfile, index=False)
    print('WRN! SNR calculation failed, but made {} anyway with nan'.
          format(snrfit_savfile))
    print('{} made {}'.format(
        datetime.utcnow().isoformat(), snrfit_savfile))


def _measure_planet_snr(plname, tfalc, statsdir, sectornum,
                        timename='TMID_BJD', nworkers=20,
                        n_transit_durations=4, use_NEA=False, use_alerts=True,
                        skipepd=False):

    plname = str(plname).replace(' ','')
    plname = str(plname).replace('.','-')

    # we only care about TFA lightcurves for planet SNR measurements.
    dtrstages = ['TFA']
    aps = [str(r) for r in range(1,4)]
    magnames = [dtrstage+ap for dtrstage in dtrstages for ap in aps]
    errnames = ['IRE'+ap for dtrstage in dtrstages for ap in aps]

    # read the lc
    hdulist = fits.open(tfalc)
    lc = hdulist[1].data

    time = lc[timename]

    plotpath = os.path.join(statsdir, str(plname)+'_AP2_lightcurve.png')
    if skipepd:
        plot_raw_epd_tfa(time, lc['IRM2'], np.zeros_like(lc['IRM2']),
                         lc['TFA2'], 1, savpath=plotpath,
                         xlabel='BJD$_\mathrm{TDB}$ [days]', skipepd=skipepd)
    else:
        plot_raw_epd_tfa(time, lc['IRM2'], lc['EP2'], lc['TFA2'], 1,
                         savpath=plotpath, xlabel='BTJD = BJD - 2457000',
                         skipepd=skipepd)


    outdir = os.path.join(statsdir, plname)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for magname, errname in zip(magnames, errnames):

        blsfit_savfile = os.path.join(
            outdir, str(plname)+'_'+str(magname)+'_BLS_fit.png')
        trapfit_savfile = os.path.join(
            outdir, str(plname)+'_'+str(magname)+'_trapezoidal_fit.png')
        snrfit_savfile = os.path.join(
            outdir, str(plname)+'_'+str(magname)+'_snr_of_fit.csv')

        mag = lc[magname]
        err_mag = lc[errname]

        mag_0 = 10  # these numbers do not matter; we care about relative flux
        flux_0 = 1000
        flux = flux_0 * 10**(-0.4 * (mag - mag_0))
        # sigma_flux = dg/d(mag) * sigma_mag, for g=f0 * 10**(-0.4*(mag-mag0)).
        err_flux = np.abs(
            -0.4 * np.log(10) * flux_0 * 10**(-0.4*(mag-mag_0)) * err_mag
        )

        fluxmedian = np.nanmedian(flux)
        flux /= fluxmedian
        err_flux /= fluxmedian

        finds = np.isfinite(flux) & np.isfinite(err_flux)
        flux = flux[finds]
        err_flux = err_flux[finds]
        if len(time) > len(flux):
            # hack to match indices on first iteration
            time = time[finds]

        # fit BLS model; plot resulting phased LC.
        if len(time)==0:
            _write_nan_df(plname, tfalc, snrfit_savfile)
            return 1
        endp = 1.05*(np.nanmax(time) - np.nanmin(time))/2
        blsdict = kbls.bls_parallel_pfind(time, flux, err_flux,
                                          magsarefluxes=True, startp=0.1,
                                          endp=endp, maxtransitduration=0.15,
                                          nworkers=nworkers, sigclip=None)
        fitd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                           blsdict['bestperiod'],
                                           maxtransitduration=0.15,
                                           magsarefluxes=True, sigclip=None,
                                           perioddeltapercent=5)
        if not isinstance(fitd,dict):
            _write_nan_df(plname, tfalc, snrfit_savfile)
            return 1

        bls_period = fitd['period']
        make_fit_plot(fitd['phases'], fitd['phasedmags'], None,
                      fitd['blsmodel'], fitd['period'], fitd['epoch'],
                      fitd['epoch'], blsfit_savfile, magsarefluxes=True)
        ingduration_guess = fitd['transitduration']*0.2
        transitparams = [fitd['period'], fitd['epoch'], fitd['transitdepth'],
                         fitd['transitduration'], ingduration_guess ]

        # fit initial trapezoidal model; plot resulting phased LC.
        trapfit = lcfit.traptransit_fit_magseries(time, flux, err_flux,
                                                  transitparams,
                                                  magsarefluxes=True,
                                                  sigclip=None,
                                                  plotfit=trapfit_savfile)

        # isolate each transit to within +/- n_transit_durations
        tmids, t_starts, t_ends = (
            get_transit_times(fitd, time, n_transit_durations, trapd=trapfit)
        )

        fitmags = trapfit['fitinfo']['fitmags']
        fitflux = fitmags

        # estimate snr direct from the trapezoidal fit
        t_full_durn_phase = trapfit['fitinfo']['finalparams'][3]
        t_ing_durn_phase = trapfit['fitinfo']['finalparams'][4]
        period = trapfit['fitinfo']['finalparams'][0]

        t_full_durn = t_full_durn_phase * period
        t_ing_dur = t_ing_durn_phase * period

        per_point_cadence=30.*u.min

        npoints_in_transit = (
            float(((t_full_durn*u.day)/per_point_cadence).cgs.value)
        )

        transitdepth = np.abs(trapfit['fitinfo']['finalparams'][2])

        # get the indices in transit, for RMS measurement
        (in_fluxs, time_list,
         intra_inds_list, windowinds ) = [], [], [], []
        for t_start,t_end in zip(t_starts, t_ends):
            this_window_inds = (time > t_start) & (time < t_end)
            tmid = t_start + (t_end-t_start)/2
            # flag out slightly more than expected "in transit" points
            prefactor = 1.05
            transit_start = tmid - prefactor*t_full_durn/2
            transit_end = tmid + prefactor*t_full_durn/2

            this_window_intra = (
                (time[this_window_inds] > transit_start) &
                (time[this_window_inds] < transit_end)
            )
            this_window_oot = ~this_window_intra

            windowinds.append(this_window_inds)
            time_list.append( time[this_window_inds] )
            in_fluxs.append( flux[this_window_inds] )
            intra_inds_list.append( (time[this_window_inds]>transit_start) &
                                    (time[this_window_inds]<transit_end) )

        try:
            intra = np.hstack(np.array(intra_inds_list))
        except ValueError:
            outdf = pd.DataFrame({'plname':plname, 'trapz_snr':np.nan,
                                  'tfalc':tfalc}, index=[0])
            outdf.to_csv(snrfit_savfile, index=False)
            print('WRN! SNR calculation failed, but made {} anyway with nan'.
                  format(snrfit_savfile))
            print('{} made {}'.format(
                datetime.utcnow().isoformat(), snrfit_savfile))
            return 0

        sel_time = np.array(time_list)
        sel_flux = np.array(in_fluxs)
        windowinds = np.bitwise_or.reduce( np.array(windowinds), axis=0 )

        try:
            subtractedrms = np.std(flux[windowinds][~intra] -
                                   fitflux[windowinds][~intra] )

            ntransits = len(t_starts)
            trapz_snr = (
                np.sqrt(npoints_in_transit*ntransits) *
                transitdepth/subtractedrms
            )

            outdf = pd.DataFrame({'plname':plname,
                                  'trapz_snr':trapz_snr,
                                  'trapz_transitdepth':transitdepth,
                                  'oot_rms':subtractedrms,
                                  'npoints_in_transit':npoints_in_transit,
                                  'ntransits':ntransits,
                                  'tfalc':tfalc}, index=[0])
            outdf.to_csv(snrfit_savfile, index=False)
            print('made {}'.format(snrfit_savfile))

        except IndexError:

            outdf = pd.DataFrame({'plname':plname, 'trapz_snr':np.nan,
                                  'tfalc':tfalc}, index=[0])
            outdf.to_csv(snrfit_savfile, index=False)
            print('WRN! SNR calculation failed, but made {} anyway with nan'.
                  format(snrfit_savfile))
            print('{} made {}'.format(
                datetime.utcnow().isoformat(), snrfit_savfile))

    return 1


def explore_ccd_temperature_timeseries():
    """
    This is a one-off exploration to understand the engineering files below
    better. Some take-aways:

        * 'tess2018331094053_sector01-eng.fits' and
        'tess2018323111417_sector01-eng.fits' have identical temperature time
        series. It's unclear why there are two of these data files, but it
        doesn't matter for our purposes. This was confirmed by email from Scott
        Fleming (2019/01/04).

        * We can extract the time-series of on-chip CCD temperature using the
        S_CAM{:d}_ALCU_sensor_CCD{:d} pattern.

        * The cadence of the above time-series is different from FFIs, and will
        need to be binned to match the FFI cadence and assign an average
        temperature to each FFI.

        * There is one other type of CCD-specific temperature available:
            * Board temperature S_CAM[1-4]_CCD[1-4]_board_temp

        * There are also a variety of camera-specific temperatures (see TESS
        Instrument Handbook, Appendix A.5 Engineering Data).
    """

    engdatadir = '/nfs/phtess1/ar1/TESS/FFI/ENGINEERING/'
    s01_first = 'tess2018331094053_sector01-eng.fits'
    s01_second = 'tess2018323111417_sector01-eng.fits'
    s02 = 'tess2018330083923_sector02-eng.fits'

    d = {}
    for dfile in [s01_first, s01_second, s02]:

        d[dfile] = {}

        hdulist = fits.open(os.path.join(engdatadir, dfile))

        ccds, cams = list(range(1,5)), list(range(1,5))

        # NOTE: unclear if "COOKED" is actually the temperature... plausible
        # but unclear.
        for cam in cams:
            for ccd in ccds:
                TEMPERATURE_HDR_NAME = 'S_CAM{:d}_ALCU_sensor_CCD{:d}'.format(cam, ccd)
                this_time = hdulist[TEMPERATURE_HDR_NAME].data['TIME']
                this_temperature = hdulist[TEMPERATURE_HDR_NAME].data['COOKED']

                d[dfile][TEMPERATURE_HDR_NAME] = {}
                d[dfile][TEMPERATURE_HDR_NAME]['time'] = this_time
                d[dfile][TEMPERATURE_HDR_NAME]['temperature'] = this_temperature


    temperature_hdr_names = ['S_CAM{:d}_ALCU_sensor_CCD{:d}'.format(cam, ccd)
                             for cam in cams for ccd in ccds]

    times = [d[k][thn]['time'] for k in d.keys() for thn in
             temperature_hdr_names]

    temperatures = [d[k][thn]['temperature'] for k in d.keys() for thn in
                    temperature_hdr_names]

    f,ax = plt.subplots()
    for k in d.keys():
        for temperature_hdr_name in temperature_hdr_names:
            if 'S_CAM1_ALCU_sensor_CCD1' not in temperature_hdr_name:
                continue
            if 'tess2018331094053_sector01-eng.fits' in k:
                ax.plot(d[k][temperature_hdr_name]['time'],
                        d[k][temperature_hdr_name]['temperature']+3,
                        label='{:s}_{:s}'.format(k, temperature_hdr_name))
            else:
                ax.plot(d[k][temperature_hdr_name]['time'],
                        d[k][temperature_hdr_name]['temperature'],
                        label='{:s}_{:s}'.format(k, temperature_hdr_name))

    ax.set_xlabel('BTJD', fontsize='small')
    ax.set_ylabel('On-chip CCD temperature sensor ("COOKED") [deg C]',
                  fontsize='small')

    ax.legend(loc='best',fontsize='xx-small')

    f.savefig('temperature_sanity_check.png', dpi=300)

    # ok, now check if they're actually THE SAME, for every single ccd. If so,
    # then having these two engineering files for sector 1 is pointless.
    for temperature_hdr_name in temperature_hdr_names:
        temperature_1 = d[s01_first][temperature_hdr_name]['temperature']
        temperature_2 = d[s01_second][temperature_hdr_name]['temperature']

        time_1 = d[s01_first][temperature_hdr_name]['time']
        time_2 = d[s01_second][temperature_hdr_name]['time']

        np.testing.assert_array_almost_equal(temperature_1, temperature_2,
                                             decimal=6)
        np.testing.assert_array_almost_equal(time_1, time_2,
                                             decimal=6)

    print('Test passed: {:s} and {:s} have identical temperature timeseries'.
          format(s01_first, s01_second))


def append_ccd_temperature_to_hdr_worker(task):

    fitsname, d = task

    framekey = os.path.splitext(os.path.basename(fitsname))[0]

    # get the temperature time series appropriate for this CAM/CCD pair using
    # the file name.  match: tess2018206192942-s0001-4-3-0120_cal_img.fits
    sr = search('{}/tess2{}-{}-{}-{}-{}_cal_img_bkgdsub.fits', fitsname)
    cam = sr[3]
    ccd = sr[4]

    thiskey = 'S_CAM{:d}_ALCU_sensor_CCD{:d}'.format(int(cam), int(ccd))

    time = d[thiskey]['time'] # engineering data time in "TJD = BJD-2457000".
    temperature = d[thiskey]['temperature'] # ditto, for CCD temperature

    # take the mean of the temperature values within this time window. append
    # it to the header.
    data, hdr = iu.read_fits(fitsname, ext=0)
    tstart = hdr['TSTART'] # start as BTJD = BJD-2457000.
    tstop = hdr['TSTOP'] # stop in BTJD, ditto.

    sel_times = (time>=tstart) & (time<tstop)

    mean_temp = np.mean( temperature[sel_times] )
    n_temps = len(temperature[sel_times])

    if n_temps >= 1:
        hdr['CCDTEMP'] = mean_temp
        hdr['NTEMPS'] = n_temps
    else:
        hdr['CCDTEMP'] = np.nan
        hdr['NTEMPS'] = 0

    hdr.comments['CCDTEMP'] = (
        'mean temperature (S_CAM{:d}_ALCU_sensor_CCD{:d})'.
        format(int(cam), int(ccd))
    )
    hdr.comments['NTEMPS'] = (
        'number of temperatures avgd for CCDTEMP'
    )

    fits.writeto(fitsname, data, header=hdr, output_verify='silentfix+ignore',
                 overwrite=True)

    if n_temps>=1:
        print('{} Wrote CCDTEMP to {:s}'.format(
            datetime.utcnow().isoformat(), fitsname))
        return framekey, mean_temp, n_temps
    else:
        print('{} WRN! Wrote NAN CCDTEMP to {:s}'.format(
            datetime.utcnow().isoformat(), fitsname))
        return framekey, np.nan, 0


def parallel_append_ccd_temperature_to_hdr(fitslist, temperaturepklpath,
                                           nworkers=16, maxworkertasks=1000):
    # pass a list of trim, cut cal images:
    # tess2018206195942-s0001-4-3-0120_cal_img.fits

    d = pickle.load(open(temperaturepklpath, 'rb'))

    print('%sZ: %s files to append CCD temperature to header' %
          (datetime.utcnow().isoformat(), len(fitslist)))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, d) for x in fitslist]

    # fire up the pool of workers
    results = pool.map(append_ccd_temperature_to_hdr_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    print('%sZ: finished appending CCD temperatures to headers' %
          (datetime.utcnow().isoformat()))

    # write CCDTEMP, NTEMPS, and FRAMEKEY csv file for later quick-read during
    # FITS LC creation.
    outdf = pd.DataFrame(results, columns=['framekey','ccdtemp','ntemps'])

    framekey = outdf['framekey'].iloc[0]
    res = search('tess{}-{}_cal_img_bkgdsub', framekey)
    keystring = res[-1]
    outname = '{}_key_temperature_count.csv'.format(keystring)

    # output csv path example:
    # /nfs/phtess1/ar1/TESS/FFI/ENGINEERING/s0002-3-3-0121_key_temperature_count.csv
    outdir = os.path.dirname(temperaturepklpath)
    outpath = os.path.join(outdir, outname)

    outdf.to_csv(outpath, index=False)
    print('{}: made {}'.format(datetime.utcnow().isoformat(), outpath))

    return {result for result in results}


def make_ccd_temperature_timeseries_pickle(sectornum):
    """
    convert the engineering data on MAST to a pickled time-series of on-chip
    CCD temperature.

    outputs:
        '/nfs/phtess1/ar1/TESS/FFI/ENGINEERING/sector0001_ccd_temperature_timeseries.pickle'

        which contains a dictionary, with keys 'S_CAM{:d}_ALCU_sensor_CCD{:d}',
        each of which contains a dictionary of time and temperature.
    """

    engdatadir = '/nfs/phtess1/ar1/TESS/FFI/ENGINEERING/'
    if sectornum==1:
        # identical to the other one on mast. LGB tested this in
        # explore_ccd_temperature_timeseries
        fname = os.path.join(engdatadir,'tess2018323111417_sector01-eng.fits')
    elif sectornum>1:
        fnames = glob(os.path.join(
            engdatadir,'tess2*_sector{:s}-eng.fits'.
            format(str(sectornum).zfill(2))
        ))
        if len(fnames) > 1:
            raise AssertionError('expected one engineering file per sector')
        elif len(fnames)==0:
            raise AssertionError('got no engineering files for this sector!!')
        fname = fnames[0]
    else:
        raise NotImplementedError

    hdulist = fits.open(fname)
    ccds, cams = list(range(1,5)), list(range(1,5))

    # NOTE: unclear if "COOKED" is actually the temperature... plausible
    # but unclear.
    d = {}
    temperature_hdr_names = ['S_CAM{:d}_ALCU_sensor_CCD{:d}'.format(cam, ccd)
                             for cam in cams for ccd in ccds]

    for temperature_hdr_name in temperature_hdr_names:
        this_time = hdulist[temperature_hdr_name].data['TIME']
        try:
            this_temperature = hdulist[temperature_hdr_name].data['COOKED']
        except KeyError:
            # in sectors >=13, the engineering data changed the key from
            # "COOKED" to "VALUE".
            this_temperature = hdulist[temperature_hdr_name].data['VALUE']
        d[temperature_hdr_name] = {}
        d[temperature_hdr_name]['time'] = this_time
        d[temperature_hdr_name]['temperature'] = this_temperature

    # note: the read out times on each camera are not identical. therefore we
    # can't make this extremely simple as just temperatures in the 16 readouts,
    # vs time. it needs to be: for each readout, what is the temperature
    # time-series?

    pklpath = os.path.join(
        engdatadir,
        'sector{:s}_ccd_temperature_timeseries.pickle'.
        format(str(sectornum).zfill(4))
    )
    with open(pklpath, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    print('{} made {}'.format(datetime.utcnow().isoformat(), pklpath))


def read_tess_txt_lightcurve(
    lcfile,
    catalogdf,
    infokeys=['Gaia-ID', 'RA[deg]', 'Dec[deg]', 'Parallax[mas]',
              'PM_RA[mas/yr]', 'PM_Dec[mas/year]', 'Ref_Epoch[yr]',
              'AstExcNoise[mas]', 'AstExcNoiseSig', 'AstPriFlag',
              'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
              'radial_velocity', 'phot_variable_flag', 'teff_val', 'a_g_val',
              'e_bp_min_rp_val', 'radius_val', 'lum_val']
    ):
    """
    Read grcollect dumped lightcurve file into a dictionary (which can then be
    put into a FITS table, or some other data structure better than text
    files).

    Args:
        lcfile (str): path to *.grcollectilc, matching format below.

        catalogdf (pd.DataFrame): output from read_object_catalog(catfile)

    catfile

    Format details:

    00 tmid_utc exp mid-time in JD_UTC (from DATE-OBS,DATE-END)
    01 rstfc    Unique frame key ({STID}-{FRAMENUMBER}_{CCDNUM})
    02 starid   GAIA ID of the object
    03 xcc      original X coordinate on CCD on photref frame
    04 ycc      original y coordinate on CCD on photref frame
    05 xic      shifted X coordinate on CCD on subtracted frame
    06 yic      shifted Y coordinate on CCD on subtracted frame
    07 fsv      Measured S value
    08 fdv      Measured D value
    09 fkv      Measured K value
    10 bgv      Background value
    11 bge      Background measurement error

    12 ifl1     Flux in aperture 1 (ADU)
    13 ife1     Flux error in aperture 1 (ADU)
    14 irm1     Instrumental magnitude in aperture 1
    15 ire1     Instrumental magnitude error for aperture 1
    16 irq1     Instrumental magnitude quality flag for aperture 1 (0/G OK, X bad)

    17 ifl2     Flux in aperture 2 (ADU)
    18 ife2     Flux error in aperture 2 (ADU)
    19 irm2     Instrumental magnitude in aperture 2
    20 ire2     Instrumental magnitude error for aperture 2
    21 irq2     Instrumental magnitude quality flag for aperture 2 (0/G OK, X bad)

    22 ifl3     Flux in aperture 3 (ADU)
    23 ife3     Flux error in aperture 3 (ADU)
    24 irm3     Instrumental magnitude in aperture 3
    25 ire3     Instrumental magnitude error for aperture 3
    26 irq3     Instrumental magnitude quality flag for aperture 3 (0/G OK, X bad)
    """

    # read the LC into a numpy recarray
    lccolnames = ['tmid_utc','rstfc','starid',
                  'xcc','ycc','xic','yic','fsv','fdv','fkv',
                  'bgv','bge',
                  'ifl1', 'ife1', 'irm1', 'ire1', 'irq1',
                  'ifl2', 'ife2', 'irm2', 'ire2', 'irq2',
                  'ifl3', 'ife3', 'irm3', 'ire3', 'irq3' ]
    dtypelist = ('f8,U48,U19,'+
                 'f8,f8,f8,f8,f8,f8,f8,'+
                 'f8,f8,'+
                 'f8,f8,f8,f8,U1,'+
                 'f8,f8,f8,f8,U1,'+
                 'f8,f8,f8,f8,U1' )

    recarr = np.genfromtxt(lcfile,
                           usecols=tuple(range(len(lccolnames))),
                           names=lccolnames,
                           dtype=dtypelist)

    # generate the objectid by stripping the filename
    objectid = os.path.basename(lcfile).rstrip(os.path.splitext(lcfile)[-1])

    # look up this object in the catalog
    objind = catalogdf['Gaia-ID'] == int(objectid)

    if objind.size > 0:

        objectinfo = catalogdf[infokeys].ix[objind]
        objectinfo = objectinfo.to_dict('records')[0]

    else:

        objectinfo = {'objectid': objectid}

    # this is the lcdict we need
    lcdict = {'objectid': objectid,
              'objectinfo':objectinfo}
    for col in lccolnames:
        lcdict[col] = recarr[col]

    return lcdict


def read_object_reformed_catalog(reformedcatalogfile, isgaiaid=False,
                                 gaiafull=True):
    """
    you often need an objectid, ra, dec, and/or 2mass magnitudes.  get these
    from the "reformed" catalog with a small sub-read.
    """

    if not isgaiaid:
        columns='id,ra,dec,xi,eta,2massJ,2massK,2massqlt,2massI,2massr,2massi,2massz'
        columns = columns.split(',')

        catarr = np.genfromtxt(reformedcatalogfile,
                               comments='#',
                               usecols=list(range(len(columns))),
                               names=columns,
                               dtype='U15,f8,f8,f8,f8,f8,f8,U3,f8,f8,f8,f8',
                               delimiter=' '
                               )
    else:
        if gaiafull:
            columns='id,ra,dec,xi,eta,G,Rp,Bp,plx,pmra,pmdec,varflag'
            dtypes='U19,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,<U16'
        else:
            columns='id,ra,dec'
            dtypes='U19,f8,f8'
        columns = columns.split(',')

        catarr = np.genfromtxt(reformedcatalogfile,
                               comments='#',
                               usecols=list(range(len(columns))),
                               names=columns,
                               dtype=dtypes,
                               delimiter=' '
                               )
    return catarr


def median_filter_frame(task):

    imgfile, outbkgdpath, outbkgdsubpath, n_sigma, k, k_sigma = task

    if os.path.exists(outbkgdpath) and os.path.exists(outbkgdsubpath):
        print('{} found and skipped {}'.format(
            datetime.utcnow().isoformat(), outbkgdpath))
        return 1

    hdulist = fits.open(imgfile)
    hdr, data = hdulist[0].header, hdulist[0].data

    std_data = np.std(data)
    med_data = np.median(data)

    # overwrite stars with the median value... (because otherwise the scipy
    # median filter faults out)
    img = deepcopy(data)

    sel_inds = data > (med_data + n_sigma*std_data)
    data[sel_inds] = med_data

    thisbkgd = median_filter(data, size=(k, k), mode='reflect')
    thisbkgd = gaussian_filter(thisbkgd, k_sigma, order=0, output=None,
                               mode='reflect', truncate=4.0)

    hdu_bkgd = fits.PrimaryHDU(thisbkgd)
    hdulist_bkgd = fits.HDUList([hdu_bkgd])

    hdu_sub = fits.PrimaryHDU(data = img - thisbkgd,
                              header=hdr)
    hdulist_sub = fits.HDUList([hdu_sub])

    outdir = os.path.dirname(outbkgdpath)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for outhdu, outfile in zip(
        [hdu_bkgd, hdu_sub],
        [outbkgdpath, outbkgdsubpath]
    ):
        if not os.path.exists(outfile):
            outhdu.writeto(outfile)
            print('{}: made {}'.
                  format(datetime.utcnow().isoformat(), outfile))
        else:
            print('{}: found & skipped {}'.
                  format(datetime.utcnow().isoformat(), outfile))

    hdulist_bkgd.close()


def none_filter_frame(task):

    imgfile, outbkgdpath, outbkgdsubpath, n_sigma, k, k_sigma = task

    if os.path.exists(outbkgdpath) and os.path.exists(outbkgdsubpath):
        print('{} found and skipped {}'.format(
            datetime.utcnow().isoformat(), outbkgdpath))
        return 1

    hdulist = fits.open(imgfile)
    hdr, data = hdulist[0].header, hdulist[0].data

    thisbkgd = np.zeros_like(data)

    hdu_bkgd = fits.PrimaryHDU(thisbkgd)
    hdulist_bkgd = fits.HDUList([hdu_bkgd])

    img = deepcopy(data)
    hdu_sub = fits.PrimaryHDU(data = img - thisbkgd,
                              header=hdr)
    hdulist_sub = fits.HDUList([hdu_sub])

    outdir = os.path.dirname(outbkgdpath)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for outhdu, outfile in zip(
        [hdu_bkgd, hdu_sub],
        [outbkgdpath, outbkgdsubpath]
    ):
        if not os.path.exists(outfile):
            outhdu.writeto(outfile)
            print('{}: made {}'.
                  format(datetime.utcnow().isoformat(), outfile))
        else:
            print('{}: found & skipped {}'.
                  format(datetime.utcnow().isoformat(), outfile))

    hdulist_bkgd.close()


def parallel_bkgd_subtract(fitslist, method='boxblurmedian', isfull=True, k=32,
                           k_sigma=32, outdir=None, nworkers=32,
                           maxworkertasks=1000, n_sigma=2):
    """
    Given list of FITS files, estimate the bulk sky background. Two additional
    files are made for each FITS image passed:

        1) "{fitsimg}_bkgd.fits": background image.
        2) "{fitsimg}_bkgdsub.fits": target image minus the background image.
        This is what should be used in subsequent image processing.

    This subtraction step is needed because the local background model
    attempted by `ficonv` (the "b" term in the kernel) is a poor model for the
    scattered light in TESS.  A box median filter (e.g., 48 x 48 on each
    frame), convolved with a gaussian kernel, does a good job.  (The same will
    probably be true for HATPI.)

    kwargs:

        method (str): 'boxblurmedian', or 'none'.

        k (int): box median size.

        k_sigma (int): standard deviation of gaussian blur kernel

        n_sigma (float): in background estimation, the first step is to replace
        all pixels > n_sigma away from the median with the median background
        value. This masks stars, but without leaving nans. The median filter is
        then run on this masked image.

        outdir (str): if passed, writes images to this directory. Else, writes
        to directory implicit in fitslist.

        isfull (bool): if a "FULL" reduction, expects 2 orbits (TESS-specific).

    NOTES:
    ----------
    The call is to scipy.ndimage.median_filter, which runs in O(N*k) time, for
    N the number of pixels in the image, k the kernel size.  Though faster
    algorithms exist (Perreault & Hebert 2007), I could not find any that were
    implemented in python (I checked PIL, astropy, and scikit learn).

    The faster algorithms seem to mostly rely on smart cacheing and updating
    histograms of the pixels in the kernel.  Some algorithms also rely on
    assuming the image is in UINT8 or UINT16, which allows some binary tree
    speedups too. The latter won't work for us.

    Possible future method to implement: "boxblurmedianplustime", for a spatial box
    median, plus a continuity condition in time (e.g., 32 x 32 spatial, plus a
    window of [-3 cadences, 3 cadences]).
    """
    if method not in ['boxblurmedian','none']:
        raise NotImplementedError(
            'boxblurmedian is only implemented bkgd subtraction method, for now'
        )

    # Pass a N x N median filter over the FFIs. N x N is the physical box
    # size in pixels. The Earth/Moon blobs move over a timescale of an
    # orbit (27.8 days). Adding a time dimension is a possible future todo.

    fitslist = np.sort(fitslist) # ensure we're time-ordered.

    if isinstance(outdir,str):
        pass
    else:
        outdir = os.path.dirname(fitslist[0])

    outbkgdnames = [ os.path.join(outdir, n.replace(".fits", "_bkgd.fits")) for
                    n in list(map(os.path.basename, fitslist)) ]

    outbkgdsubnames = [ os.path.join(outdir, n.replace(
        ".fits", "_bkgdsub.fits")) for n in
        list(map(os.path.basename, fitslist))
    ]

    # TESS-specific: split by orbits, because of sharp features at
    # beginning/end of orbits. 
    tstarts, tstops = [], []
    print('getting times for bkgd subtraction')
    for img in fitslist:
        d = iu.get_header_keyword_list(img, ['TSTART','TSTOP'], ext=0)
        tstarts.append(d['TSTART'])
        tstops.append(d['TSTOP'])
    print('got times for bkgd subtraction')
    tstarts, tstops = nparr(tstarts), nparr(tstops)
    times = tstarts + (tstops-tstarts)/2.

    from parse import parse
    res = parse('{}/sector-{}/{}',fitslist[0])
    sectornum = int(res[1])
    if sectornum in [1,2,4,5,6,7,8,9,10,11,17]:
        orbitgap = 1. # days
    elif sectornum in [3]:
        orbitgap = 0.15 # days
    elif sectornum in [12,13,14,15]:
        orbitgap = 0.5 # days
    else:
        errmsg = 'need manual orbitgap to be implemented in bkgdsub'
        raise NotImplementedError(errmsg)

    if isfull and sectornum in [1,2,5,6,7,9,10,11,12,13,14,15,17]:
        expected_norbits = 2
    elif not isfull:
        expected_norbits = 1
    elif sectornum in [4,8]:
        expected_norbits = 3 # (quality cuts down to 2 afterwards)
    else:
        expected_norbits = 2

    norbits, groups = lcmath.find_lc_timegroups(times, mingap=orbitgap)

    if norbits != expected_norbits:
        outmsg = (
            'bkgd estimation assumes given two orbits. {} orbits. Time {}'.
            format(norbits, repr(times))
        )
        raise AssertionError(outmsg)

    # Do background subtraction for each orbit separately.
    for group_ix, group in enumerate(groups):

        groupind = np.zeros_like(times)
        groupind[group.start : group.stop] += 1
        groupind = groupind.astype(np.bool)

        tg_times = times[groupind]
        tg_fitsimgs = fitslist[groupind]
        tg_outbkgdnames = nparr(outbkgdnames)[groupind]
        tg_outbkgdsubnames = nparr(outbkgdsubnames)[groupind]

        print('{} starting median filter for orbit {}'.
              format(datetime.utcnow().isoformat(), group_ix))

        print('%sZ: %s files to median filter' %
              (datetime.utcnow().isoformat(), len(tg_fitsimgs)))

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        tasks = [(x, y, z, n_sigma, k, k_sigma) for x, y, z in
                 zip(tg_fitsimgs, tg_outbkgdnames, tg_outbkgdsubnames) ]

        if method == 'boxblurmedian':
            results = pool.map(median_filter_frame, tasks)
        elif method == 'none':
            results = pool.map(none_filter_frame, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        print('%sZ: finished median filtering' %
              (datetime.utcnow().isoformat()))

    return 1


def plot_apertures_on_frame(fitsframe, photrefprojcat, xlim=None, ylim=None):

    df = pd.read_csv(
        photrefprojcat,
        sep=' ',
        names='id,ra,dec,xi,eta,G,Rp,Bp,plx,pmra,pmdec,varflag,x_proj,y_proj'.split(',')
    )

    vmin, vmax = 10, int(1e3)

    xstr = '_x_{}t{}'.format(xlim[0],xlim[1]) if isinstance(xlim, list) else ''
    ystr = '_y_{}t{}'.format(ylim[0],ylim[1]) if isinstance(ylim, list) else ''

    outdir = os.path.dirname(fitsframe)
    outpath = (
        os.path.join(
            outdir,
            os.path.basename(fitsframe).replace(
                '.fits','_apertures_on_frame{}{}.png'.format(xstr,ystr))
        )
    )

    img, _ = iu.read_fits(fitsframe)

    plt.close('all')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots(figsize=(6,6))

    import matplotlib.colors as colors
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    img[img<0] = 0.01

    cset = ax.imshow(img, cmap='binary_r', vmin=vmin, vmax=vmax, norm=norm)

    # -1 total because image origin
    ax.scatter(df['x_proj']-.5, df['y_proj']-.5, s=6, facecolors='none',
               edgecolors='r', linewidths=0.5)

    if isinstance(xlim, list):
        ax.set_xlim(xlim)
    if isinstance(ylim, list):
        ax.set_ylim(ylim)

    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.get_yaxis().set_tick_params(which='both', direction='in')

    cb = fig.colorbar(cset, ax=ax, extend='both', fraction=0.046, pad=0.04)
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')

    fig.tight_layout()

    fig.savefig(outpath, bbox_inches='tight', dpi=450)
    print('{}: made {}'.format(datetime.utcnow().isoformat(), outpath))


def plot_median_filter_quad(task):

    bkgdfile, calfile, outdir = task

    vmin, vmax = 10, int(1e3)

    outpath = (
        os.path.join(outdir,
                     os.path.basename(bkgdfile).replace('.fits','.png'))
    )

    if os.path.exists(outpath):
        print('found {}. continue'.format(outpath))
        return 0

    bkgd_img, _ = iu.read_fits(bkgdfile)
    cal_img, _ = iu.read_fits(calfile)

    plt.close('all')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, axs = plt.subplots(ncols=2, nrows=2)

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    cset1 = axs[0,1].imshow(cal_img, cmap='binary_r', vmin=vmin, vmax=vmax,
                            norm=norm)

    diff_vmin, diff_vmax = -1000, 1000

    diffnorm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=diff_vmin,
                                 vmax=diff_vmax)

    axs[0,0].imshow(bkgd_img - np.median(cal_img), cmap='RdBu_r',
                    vmin=diff_vmin, vmax=diff_vmax, norm=diffnorm)

    cset2 = axs[1,0].imshow(cal_img - bkgd_img, cmap='RdBu_r', vmin=diff_vmin,
                            vmax=diff_vmax, norm=diffnorm)

    axs[1,1].imshow(cal_img - np.median(cal_img), cmap='RdBu_r',
                    vmin=diff_vmin, vmax=diff_vmax, norm=diffnorm)

    for ax in axs.flatten():
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

    cb1 = fig.colorbar(cset1, ax=axs[0,1], extend='both')
    cb2 = fig.colorbar(cset2, ax=axs[1,1], extend='both')
    cb2.set_ticks([-1e3,-1e2,-1e1,0,1e1,1e2,1e3])
    cb2.set_ticklabels(['-$10^3$','-$10^2$','-$10^1$','0',
                        '$10^1$','$10^2$','$10^3$'])

    fig.tight_layout(h_pad=0.1, w_pad=-7.3, pad=0)

    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    print('{}: made {}'.format(datetime.utcnow().isoformat(), outpath))


def parallel_plot_median_filter_quad(fitsdir, nworkers=16, maxworkertasks=1000):
    """
    make 2x2 plots of:
        calibrated image          |     bkgd estimate
        cal - bkgd                |     cal - constant median
    """
    bkgdfits = glob(os.path.join(fitsdir, '*_cal_img_bkgd.fits'))
    _fitslist = [b.replace('_bkgd','') for b in bkgdfits]

    outdir = fitsdir

    print('%sZ: %s files to plot median filter quad' %
          (datetime.utcnow().isoformat(), len(bkgdfits)))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x,y,outdir) for x,y in zip(bkgdfits, _fitslist)]

    results = pool.map(plot_median_filter_quad, tasks)

    pool.close()
    pool.join()

    return 1


def read_object_catalog(catalogfile):
    """
    read the Gaia .catalog file into a pandas DataFrame
    """

    df = pd.read_csv(catalogfile, header=0, delim_whitespace=True)

    # columns are read in looking like "teff_val[35]". strip the [integer], and
    # the pre-pended "#" on the first column.
    cols = [re.sub('\[\d+\]', '', c) for c in df.columns]
    cols = [c.lstrip('#') for c in cols]

    df = df.rename(index=str,
                   columns={old:new for old, new in zip(df.columns, cols)})

    # columns are:
    # ['Gaia-ID', 'RA[deg]', 'Dec[deg]', 'RAError[mas]', 'DecError[mas]',
    # 'Parallax[mas]', 'Parallax_error[mas]', 'PM_RA[mas/yr]',
    # 'PM_Dec[mas/year]', 'PMRA_error[mas/yr]', 'PMDec_error[mas/yr]',
    # 'Ref_Epoch[yr]', 'AstExcNoise[mas]', 'AstExcNoiseSig', 'AstPriFlag',
    # 'phot_g_n_obs', 'phot_g_mean_flux', 'phot_g_mean_flux_error',
    # 'phot_g_mean_flux_over_error', 'phot_g_mean_mag', 'phot_bp_n_obs',
    # 'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
    # 'phot_bp_mean_flux_over_error', 'phot_bp_mean_mag', 'phot_rp_n_obs',
    # 'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
    # 'phot_rp_mean_flux_over_error', 'phot_rp_mean_mag',
    # 'phot_bp_rp_excess_factor', 'radial_velocity', 'radial_velocity_error',
    # 'phot_variable_flag', 'teff_val', 'teff_percentile_lower',
    # 'teff_percentile_upper', 'a_g_val', 'a_g_percentile_lower',
    # 'a_g_percentile_upper', 'e_bp_min_rp_val',
    # 'e_bp_min_rp_percentile_lower', 'e_bp_min_rp_percentile_upper',
    # 'radius_val', 'radius_percentile_lower', 'radius_percentile_upper',
    # 'lum_val', 'lum_percentile_lower', 'lum_percentile_upper', 'xi[deg]',
    # 'eta[deg]']

    return df


def merge_object_catalog_vs_cdips(
    in_reformed_cat_file, out_reformed_cat_file,
    cdips_cat_file=('/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
                    'cdips_targets_v0.6_gaiasources_Rplt16_orclose.csv'),
    G_Rp_cut=14):
    """
    the CDIPS project defines a cluster star sample for which we should make
    lightcurves no matter what.

    this function:

        * takes a reformed_catalog (created by calling
        ap.reform_gaia_fov_catalog(catalog_file, reformed_cat_file))

        * applies a secondary magnitude cut (G_Rp < G_Rp_cut), except if the
        star is a cluster star, in which case it is always included (provided
        it was within whatever first magnitude cut was used to create the
        reformed_gaia_fov_catalog).

        * writes to out_reformed_cat_file. if in_reformed_cat_file ==
        out_reformed_cat_file, will overwrite.
    """

    catarr = read_object_reformed_catalog(in_reformed_cat_file, isgaiaid=True)
    field_df = pd.DataFrame(catarr)
    del catarr

    cdips_df = pd.read_csv(cdips_cat_file, sep=',')

    field_ids = np.array(field_df['id']).astype(str)
    cdips_ids = np.array(cdips_df['source_id']).astype(str)
    del cdips_df

    print('{} begin source_id xmatch DR2 to CDIPS...'.
          format(datetime.utcnow().isoformat()))
    is_cdips = np.in1d(field_ids, cdips_ids)
    print('{} completed source_id xmatch DR2 to CDIPS...'.
          format(datetime.utcnow().isoformat()))

    sel = (field_df['Rp'] < G_Rp_cut) | is_cdips

    outdf = field_df[sel]

    if in_reformed_cat_file == out_reformed_cat_file:
        print('[WRN!] will overwrite {}'.format(in_reformed_cat_file))
        print('[WRN!] N_stars for LCs before cut: {}'.format(len(field_df)))
        print('[WRN!] N_stars for LCs after: {}'.format(len(outdf)))

    outdf.to_csv(out_reformed_cat_file, sep=' ', index=False, header=False,
                 na_rep='n/a')
    print('wrote {}'.format(out_reformed_cat_file))

    if np.any(is_cdips):
        print('[INFO!] got {} CDIPS stars in field, including...'.
              format(len(is_cdips[is_cdips])))
        print('{}'.format(repr(field_df[is_cdips].head(n=20))))
    else:
        print('[INFO!][WRN!] got 0 CDIPS stars in field')


def plot_lc_positions(lcdir, lcglob, statsdir, outname=None, N_desired=20000):
    """
    Given list of lcpaths, plot their on-chip positions.

    args:
        lcpaths (list): of lightcurve paths.

        outname: outpath is join of statsdir and outname, if given.
    """

    lcpaths = glob(os.path.join(lcdir, lcglob))

    if len(lcpaths) > N_desired:
        selpaths = np.random.choice(lcpaths, size=N_desired, replace=False)
    else:
        selpaths = lcpaths

    print('beginning plot LC positions on {} LCs'.format(len(selpaths)))

    if not outname:
        outname = 'lc_position_scatter_plot.png'
    outpath = os.path.join(statsdir, outname)
    if os.path.exists(outpath):
        print('found {}, skip'.format(outpath))
        return

    xs, ys, ras, decs = [], [], [], []
    for selpath in selpaths:
        hdul = fits.open(selpath)
        xs.append(hdul[0].header['XCC'])
        ys.append(hdul[0].header['YCC'])
        ras.append(hdul[0].header['RA[deg]'])
        decs.append(hdul[0].header['Dec[deg]'])
    xs, ys, ras, decs = nparr(xs), nparr(ys), nparr(ras), nparr(decs)

    f, ax = plt.subplots(figsize=(4,4))
    ax.scatter(xs, ys, c='k', alpha=0.5, s=0.5, rasterized=True, linewidths=0)
    ax.set_title(os.path.basename(lcdir))

    ax.set_xlabel('x on photref')
    ax.set_ylabel('y on photref')

    f.savefig(outpath, bbox_inches='tight', dpi=350)
    print('made {}'.format(outpath))

    outdf = pd.DataFrame({'x':xs,'y':ys,'ra':ras,'dec':decs})
    outdf.to_csv(outpath.replace('.png', '.csv'), index=False)
    print('made {}'.format(outpath.replace('.png', '.csv')))


def mask_orbit_start_and_end(time, flux, orbitgap=1, expected_norbits=2,
                             orbitpadding=6/(24), raise_error=True):
    """
    Ignore the times near the edges of orbits.

    args:
        time, flux
    returns:
        time, flux: with `orbitpadding` days trimmed out
    """
    norbits, groups = lcmath.find_lc_timegroups(time, mingap=orbitgap)

    if norbits != expected_norbits:
        errmsg = 'got {} orbits, expected {}. groups are {}'.format(
            norbits, expected_norbits, repr(groups))
        if raise_error:
            raise AssertionError(errmsg)
        else:
            print(errmsg)
            print('returning what was passed')
            return time, flux

    sel = np.zeros_like(time).astype(bool)
    for group in groups:
        tg_time = time[group]
        start_mask = (np.min(tg_time), np.min(tg_time) + orbitpadding)
        end_mask = (np.max(tg_time) - orbitpadding, np.max(tg_time))
        sel |= (
            (time > max(start_mask)) & (time < min(end_mask))
        )

    return_time = time[sel]
    return_flux = flux[sel]

    return return_time, return_flux
