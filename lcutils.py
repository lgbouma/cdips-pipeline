#!/usr/bin/env python

"""
lcutils.py - Luke Bouma (luke@astro.princeton.edu) - Jan 2019

Tools for working with FITS lightcurves. Including: converting text LCs (from
grcollect) to FITS lightcurves. Running EPD on FITS lightcurves. Getting
barycentric time correction for FITS lightcurves. And making input lists TFA
needs, from FITS LCs.

plot_fits_lightcurve

parallel_convert_grcollect_to_fits_lc
    convert_grcollect_to_fits_lc_worker

parallel_run_epd_imagesub_fits
    epd_fitslightcurve_imagesub_worker
    epd_fitslightcurve_imagesub

make_ascii_files_for_vartools
    _make_tfa_lc_list
    _make_trendlist_tfa
    _make_dates_tfa

initial_tfa

round_two_tfa_selection

run_tfa

parallel_merge_tfa_lcs
    merge_tfa_lc_worker

parallel_apply_barycenter_time_correction
    apply_barycenter_time_correction_worker
    astropy_utc_time_to_bjd_tdb

test_pca
"""

import os, sys, shutil
import os.path
import logging
from glob import glob
from parse import search
from datetime import datetime

import numpy as np, pandas as pd, matplotlib.pyplot as plt

from numpy import array as nparr, all as npall, isfinite as npisfinite
np.seterr(all='ignore')

import multiprocessing as mp
from functools import partial

import tessutils as tu, imageutils as iu, imagesubphot as ism

from astropy.io import fits
from astropy.time import Time
from astropy import units as u
import astropy
import astropy.coordinates as coord

DEBUG = False

def plot_fits_lightcurve(filename, apstr, outdir, customstr='',
                         maskbadtimes=True):
    """
    given path to a *_llcs.fits lightcurve, plot it.

    args:
        filename: /dir/to/*_llcs.fits
        apstr: e.g., 'TFA2'
        outdir: to write the lightcurve image

    kwargs:
        customstr: string to print, under the RMS info. e.g., "\nhidude"
        maskbadtimes: needed only if you want to try masking times that weren't
        already masked in badframes during LC construction.
    """

    # define outpath
    outext = '_{}.png'.format(apstr)
    outpath = os.path.join(
        outdir, os.path.basename(filename).replace('.fits',outext))

    # load lc
    hdulist = fits.open(filename)
    hdr = hdulist[0].header
    time_bjd = hdulist[1].data['TMID_BJD']
    barycorr = hdulist[1].data['BJDCORR']
    mag = hdulist[1].data[apstr]
    hdulist.close()

    from matplotlib.ticker import FormatStrFormatter

    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,2))

    if maskbadtimes:
        from tessutils import badtimewindows
        # BJD = TJD + LTT_corr
        tjd = time_bjd - barycorr
        toffset = 2457000
        tjd -= toffset

        good_inds = np.ones_like(tjd).astype(bool)
        for window in badtimewindows:
            bad_window_inds = (tjd > min(window)) & (tjd < max(window))
            good_windows_inds = ~bad_window_inds
            good_inds &= good_windows_inds

        time = time_bjd[good_inds]
        mag = mag[good_inds]

    else:
        time = time_bjd

    ax.scatter(time, mag, c='black', alpha=0.9, zorder=2, s=3, rasterized=True,
               linewidths=0)

    txt_x, txt_y = 0.99, 0.98
    stdmmag = np.nanstd(mag)*1e3
    if stdmmag > 0.1:
        stattxt = '$\sigma$ = {:.1f} mmag{}'.format(stdmmag, customstr)
        ndigits = 2
    elif stdmmag > 0.01:
        stattxt = '$\sigma$ = {:.2f} mmag{}'.format(stdmmag, customstr)
        ndigits = 3
    else:
        stattxt = '$\sigma$ = {:.3f} mmag{}'.format(stdmmag, customstr)
        ndigits = 4
    _ = ax.text(txt_x, txt_y, stattxt, horizontalalignment='right',
            verticalalignment='top', fontsize='small', zorder=3,
            transform=ax.transAxes)
    ax.get_yaxis().set_tick_params(which='both', direction='in',
                                   labelsize='x-small')
    ax.get_xaxis().set_tick_params(which='both', direction='in',
                                   labelsize='x-small')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim), min(ylim)))

    ax.set_xlabel('BJD-tdb', fontsize='small')
    ax.set_ylabel(apstr, fontsize='small')

    fig.tight_layout(h_pad=-0.5)
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), outpath))




def _map_key_to_format(key):
    """
    given key, get FITS format code. Works for arbitrary number of apertures.  see
    http://docs.astropy.org/en/stable/io/fits/usage/table.html for
    details.
    """

    # example keys being mapped for TESS:
    # ['bge', 'bgv', 'fdv', 'fkv', 'fsv',
    #  'ife1', 'ife2', 'ife3', 'ifl1', 'ifl2',
    #  'ifl3', 'ire1', 'ire2', 'ire3', 'irm1', 'irm2', 'irm3', 'irq1', 'irq2',
    #  'irq3', 'tmid_utc', 'rstfc', 'xic', 'yic']

    if key in ['bge','bgv','fdv','fkv','fsv','xic','yic']:
        return 'D'
    elif 'ife' in key: # flux err
        return 'D'
    elif 'ifl' in key: # flux
        return 'D'
    elif 'ire' in key: # raw mag err
        return 'D'
    elif 'irm' in key: # raw mag
        return 'D'
    elif 'irq' in key: # quality flag
        return '1A'
    elif 'tmid_utc' in key: # timestamp
        return 'D'
    elif 'rstfc' in key: # frame id
        return '48A'

def _map_key_to_comment(k):
    kcd = {
        "tmid_utc": "exp mid-time in JD_UTC (from DATE-OBS,DATE-END)",
        "rstfc" : "Unique frame key",
        "starid": "GAIA ID of the object",
        "xcc"   : "original X coordinate on CCD on photref frame",
        "ycc"   : "original y coordinate on CCD on photref frame",
        "xic"   : "shifted X coordinate on CCD on subtracted frame",
        "yic"   : "shifted Y coordinate on CCD on subtracted frame",
        "fsv"   : "Measured S value",
        "fdv"   : "Measured D value",
        "fkv"   : "Measured K value",
        "bgv"   : "Background value",
        "bge"   : "Background measurement error",
        "ifl1"  : "Flux in aperture 1 (ADU)",
        "ife1"  : "Flux error in aperture 1 (ADU)",
        "irm1"  : "Instrumental mag in aperture 1",
        "ire1"  : "Instrumental mag error for aperture 1",
        "irq1"  : "Instrumental quality flag ap 1, 0/G OK, X bad",
        "ifl2"  : "Flux in aperture 2 (ADU)",
        "ife2"  : "Flux error in aperture 2 (ADU)",
        "irm2"  : "Instrumental mag in aperture 2",
        "ire2"  : "Instrumental mag error for aperture 2",
        "irq2"  : "Instrumental quality flag ap 2, 0/G OK, X bad",
        "ifl3"  : "Flux in aperture 3 (ADU)",
        "ife3"  : "Flux error in aperture 3 (ADU)",
        "irm3"  : "Instrumental mag in aperture 3",
        "ire3"  : "Instrumental mag error for aperture 3",
        "irq3"  : "Instrumental quality flag ap 3, 0/G OK, X bad",
        "ccdtemp" : "mean CCD temperature S_CAM_ALCU_sensor_CCD",
        "ntemps"  : "number of temperatures avgd to get ccdtemp",
        'dtr_isub': "img subtraction photometry performed",
        'dtr_epd' : "EPD detrending performed",
        'dtr_tfa' : "TFA detrending performed",
        'dtr_pca' : "PCA detrending performed",
        'projid' :  "PIPE-TREX identifier for software version",
        'btc_ra' : "Right ascen in barycentric time correction",
        'btc_dec' : "Declination in barycentric time correction",
        'rdistpx': "Distance from pixel (1024,1024) [pixels]",
        'thetadeg': "Azimuth angle from pixel center [degrees]"
    }
    return kcd[k]


def convert_grcollect_to_fits_lc_worker(task, **kwargs):
    """
    Given grcollect lightcurve, make a FITS binary table with the lightcurve
    data. Sort the lightcurve in time order. Collect header information from
    the frames that created the lightcurve.  (TESS-specific): get the CCD
    temperature timeseries.
    """

    (grclcpath, outpath) = task
    observatory = kwargs.get('observatory')
    lcdir = kwargs.get('lcdir')
    fitsdir = kwargs.get('fitsdir')
    projectid = kwargs.get('projectid')
    temperaturedf = kwargs.get('temperaturedf')

    if observatory != 'tess':
        # if not TESS, you may need to modify the header reads, and the overall
        # lightcurve format 
        raise NotImplementedError

    # use the "catdf" GAIADR2-*.catalog DataFrame from global memory.
    lcd = tu.read_tess_txt_lightcurve(grclcpath, catdf)

    # sort the lightcurve, and the X, Y, temperature, etc timeseries, time.
    # every entry in "sortkeys" is a np.ndarray.
    times = lcd['tmid_utc']
    sortkeys = [k for k in np.sort(list(lcd.keys())) if k not in
                ['objectid','objectinfo']]
    timesortedind = np.argsort(times)
    for k in sortkeys:
        lcd[k] = lcd[k][timesortedind]

    # read the fits header from the earliest frame in the sequence; inherit
    # various header properties from this frame.
    earliestframename = lcd['rstfc'][np.argmin(times)]
    earliestframepath = os.path.join(fitsdir, earliestframename+'.fits')

    kwlist=['SIMPLE', 'SIMDATA', 'TELESCOP', 'INSTRUME', 'CAMERA', 'CCD',
            'EXPOSURE', 'TIMEREF', 'TASSIGN', 'TIMESYS', 'BJDREFF',
            'TIMEUNIT', 'TELAPSE', 'LIVETIME', 'TSTART', 'TSTOP', 'DATE-OBS',
            'DATE-END', 'DEADC', 'TIMEPIXR', 'TIERRELA',
            'INT_TIME', 'FRAMETIM',
            'NUM_FRM', 'TIMEDEL', 'NREADOUT']

    hdict = iu.get_header_keyword_list(earliestframepath, kwlist, ext=0)
    cdict = iu.get_header_comment_list(earliestframepath, kwlist, ext=0)

    # make the primary header
    hdr = fits.Header()
    for k in np.sort(list(hdict.keys())):
        hdr[k.replace('_','')] = ( hdict[k], cdict[k] )

    primary_hdu = fits.PrimaryHDU(header=hdr)

    # make the lightcurve data extension, and make list of time-series columns
    datakeys = [k for k in np.sort(list(lcd.keys()))
                if k not in ['starid','xcc','ycc','objectid','objectinfo']
               ]

    outnames = [k.upper() for k in datakeys]

    formats = [_map_key_to_format(k) for k in datakeys]

    datacols = [lcd[k] for k in datakeys]

    fitscollist = [fits.Column(name=n, format=f, array=a) for n,f,a in
                   zip(outnames, formats, datacols)]

    # (TESS-specific): get the CCD temperature timeseries. This is stored
    # in each frame, as identified by the 'rstfc' data key.
    if observatory=='tess':

        # Only pick the frame keys that have flux values in the lightcurve. A
        # few nans from the original FITS files are expected, because of the
        # momentum dumps.
        inds = nparr(temperaturedf['framekey'].isin(lcd['rstfc']))

        tdfframekeys = nparr(temperaturedf['framekey'])[inds]
        ccdtemp = nparr(temperaturedf['ccdtemp'])[inds]
        ntemps = nparr(temperaturedf['ntemps'])[inds]

        # Sort the temperature timeseries to be in the same order as
        # the lightcurve, using the framekeys. Requires a double argsort, for
        # example:
        #
        # a = array([4, 2, 5, 6])
        # b = array([5, 2, 6, 4])
        # b.argsort()[a.argsort()] # array([3, 1, 0, 2])
        #
        # where we're matching b to the array a.
        matchsortinds = tdfframekeys.argsort()[lcd['rstfc'].argsort()]
        tdfframekeys = tdfframekeys[matchsortinds]
        ccdtemp = ccdtemp[matchsortinds]
        ntemps = ntemps[matchsortinds]

        np.testing.assert_array_equal(
            tdfframekeys, lcd['rstfc'],
            err_msg='got tdfframekeys != lcd keys. bad temperature sort?'
        )

        fitscollist.append(
            fits.Column(name='CCDTEMP', format='D', array=ccdtemp)
        )
        fitscollist.append(
            fits.Column(name='NTEMPS', format='J', array=ntemps)
        )

    hdutimeseries = fits.BinTableHDU.from_columns(fitscollist)

    # update the header comments for the timeseries data. to do this, since the
    # timeseries data is listed in sequential "TTYPE1", "TTYPE2", etc columns
    # (this is forced), need to map those column names to the keys first.
    tfields = hdutimeseries.header['TFIELDS'] # number of table fields
    tshdrkv = {}
    for ind in range(1,tfields+1):
        key = 'TTYPE{}'.format(ind)
        tshdrkv[key] = hdutimeseries.header[key]
    for k,v in tshdrkv.items():
        hdutimeseries.header.comments[k] = _map_key_to_comment(v.lower())

    # ditto, for the primary header data. include xcc and ycc. FITS does not
    # allow nans.
    lcd['objectinfo'] = {
        (k if not pd.isnull(v) else k):
        (v if not pd.isnull(v) else 'NaN')
        for k,v in lcd['objectinfo'].items()
    }

    for k in lcd['objectinfo'].keys():
        primary_hdu.header['HIERARCH '+k] = lcd['objectinfo'][k]

    kwlist=['SIMPLE', 'SIMDATA', 'TELESCOP', 'INSTRUME', 'CAMERA', 'CCD',
            'EXPOSURE', 'TIMEREF', 'TASSIGN', 'TIMESYS', 'BJDREFI', 'BJDREFF',
            'TIMEUNIT', 'TELAPSE', 'LIVETIME', 'TSTART', 'TSTOP', 'DATE-OBS',
            'DATE-END', 'DEADC', 'TIMEPIXR', 'TIERRELA', 'BTC_PIX1',
            'BTC_PIX2', 'BARYCORR', 'INT_TIME', 'READ_TIME', 'FRAMETIM',
            'NUM_FRM', 'TIMEDEL', 'NREADOUT']

    xcc, ycc = np.mean(lcd['xcc']), np.mean(lcd['ycc'])

    # get polar coordinates from the field center. arctan2 is the numpy
    # function appropriate for quadrant-specific arctangent.
    x_mid, y_mid = 1024, 1024
    r = np.sqrt((xcc - x_mid)**2 + (ycc - y_mid)**2)
    theta_deg = np.arctan2( ycc-y_mid, xcc-x_mid ) * 180 / np.pi

    primary_hdu.header['XCC'] = xcc
    primary_hdu.header['YCC'] = ycc
    primary_hdu.header['RDISTPX'] = r
    primary_hdu.header['THETADEG'] = theta_deg
    primary_hdu.header['PROJID'] = projectid
    primary_hdu.header['DTR_ISUB'] = True
    primary_hdu.header['DTR_EPD'] = False
    primary_hdu.header['DTR_TFA'] = False
    primary_hdu.header['DTR_PCA'] = False
    for k in ['xcc','ycc','PROJID',
              'DTR_ISUB','DTR_EPD','DTR_TFA','DTR_PCA',
              'THETADEG','RDISTPX'
    ]:
        primary_hdu.header.comments[k] = _map_key_to_comment(k.lower())

    hdulist = fits.HDUList([primary_hdu, hdutimeseries])

    if os.path.exists(outpath):
        os.remove(outpath)
    hdulist.writeto(outpath, overwrite=False)
    print('{} wrote {}'.format(datetime.utcnow().isoformat(), outpath))


def parallel_convert_grcollect_to_fits_lc(lcdirectory,
                                          fitsdir,
                                          projectid,
                                          catfile='/nfs/phtess1/ar1/TESS/FFI/RED_IMGSUB/TUNE/s0002/RED_4-4-1028_ISP/GAIADR2-RA98.4609411225734-DEC-58.5412164069738-SIZE12.catalog',
                                          ilcglob='*.grcollectilc',
                                          nworkers=16,
                                          maxworkertasks=1000,
                                          observatory='tess',
                                          temperaturedfpath=None,
                                          overwrite=False):
    """
    Parallelizes convert_grcollect_to_fits_lc_worker.

    the TESS SPOC lc pattern is:
        tessyyyydddhhmmss-ssctr-tid-scid-cr_lc.fits.gz

    we will eventually use:
        projid1234_s000X_gaiaid_llc.fits.gz

    Non-obvious kwargs:
        catfile (str): path to non-reformed Gaia catalog, projected onto the
        frame. Various star parameters are read to the FITS headers.

        temperaturedfpath (str): if TESS, path to the engineering file with
        temperature information (CCDTEMP, NTEMPS, FRAMEKEY). E.g.,
        /nfs/phtess1/ar1/TESS/FFI/ENGINEERING/s0002-3-3-0121_key_temperature_count.csv
    """

    # Load the catalog file as a GLOBAL to prevent copying it (~few Gb) when
    # spawning processes below. nb. multiprocessing.Pool also has an
    # "initializer" arg that can work for this, but this is such an easy
    # implementation that it's what I'm going for. NB. it is deleted at the end
    # of this function.
    if observatory=='tess':
        global catdf
        catdf = tu.read_object_catalog(catfile)
        temperaturedf = pd.read_csv(temperaturedfpath)
    else:
        raise NotImplementedError

    ilclist = glob(os.path.join(lcdirectory, ilcglob))
    outlist = [i.replace(os.path.splitext(ilcglob)[-1], '_llc.fits')
               for i in ilclist]

    print('%sZ: %s files to convert grcollect -> FITS lightcurves' %
          (datetime.utcnow().isoformat(), len(ilclist)))

    if not overwrite:
        path_exists = []
        for outpath in outlist:
            if os.path.exists(outpath):
                path_exists.append(1)
            else:
                path_exists.append(0)

        path_exists = nparr(path_exists)

        if npall(path_exists):
            print(
                'found all {:d} FITS lightcurves, continuing'.
                format(len(path_exists))
            )
            return 0

    tasks = [(x, y) for x,y in zip(ilclist, outlist)]
    kwargs = {}
    kwargs['observatory'] = observatory
    kwargs['lcdirectory'] = lcdirectory
    kwargs['fitsdir'] = fitsdir
    kwargs['projectid'] = projectid
    kwargs['temperaturedf'] = temperaturedf

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
    results = pool.map(
        partial(convert_grcollect_to_fits_lc_worker, **kwargs), tasks
    )

    # wait for the processes to complete work
    pool.close()
    pool.join()

    del catdf
    return 1



##################
# EPD DETRENDING #
##################

def parallel_run_epd_imagesub_fits(fitsilcfiles, outfiles, smooth=21,
                                   sigmaclip=3.0, minndet=200,
                                   observatory='tess', nworkers=16,
                                   maxworkertasks=1000):

    print('%sZ: %s lcs to run EPD on ' %
          (datetime.utcnow().isoformat(), len(fitsilcfiles)))

    tasks = [(x, y, smooth, sigmaclip, minndet, observatory)
             for x,y in zip(fitsilcfiles, outfiles)]

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
    results = pool.map(epd_fitslightcurve_imagesub_worker, tasks)

    pool.close()
    pool.join()

    return {result for result in results}


def epd_fitslightcurve_imagesub_worker(task):

    x, y, smooth, sigmaclip, minndet, observatory = task

    return epd_fitslightcurve_imagesub(x, y, smooth=smooth,
                                       sigmaclip=sigmaclip, minndet=minndet,
                                       observatory=observatory)


def epd_fitslightcurve_imagesub(fitsilcfile, outfile, smooth=21, sigmaclip=3.0,
                                minndet=200, observatory='tess'):
    """
    Runs the EPD process on fitsilcfile, using columns specified to get the
    required parameters.

    The recommended use case is to overwrite the fitsilcfile with added EPD
    columns and an updated "DTR_EPD" flag in the primary HDU, by passing
    identical paths for fitsilcfile and outfile.

    The FITS lightcurves can have arbitrary columns, but should include:

        [ 'FDV', 'FKV', 'FSV', 'XIC', 'YIC']

    and apertures labelled "IRM1", "IRM2", etc for instrumental magnitudes.

    (This function is similar to epd_lightcurve_imagesub, but different I/O and
    generalizes to arbitrary aperture numbers, hence a new function).
    """

    # read the lightcurve in. check if EPD has already been performed, or if
    # there are insufficient data points to perform it.
    inhdulist = fits.open(fitsilcfile)
    try:
        ilc, primary_hdu = inhdulist[1].data, inhdulist[0]
    except (IndexError, TypeError) as e:
        print('{} ERR! {} found corrupted header/data. removing lc file.'.
              format(datetime.utcnow().isoformat(), fitsilcfile))
        print('ERR! error was {}'.format(
            datetime.utcnow().isoformat(), repr(e)))
        os.remove(fitsilcfile)
        return 0

    if primary_hdu.header['DTR_EPD']:
        print('WRN! {} found EPD had been performed; skipping'.
              format(fitsilcfile))
        return 0

    if len(ilc['XIC']) < minndet:
        print('not running EPD for %s, ndet = %s < min ndet = %s' %
              (fitsilcfile, len(ilc['XIC']), minndet))
        return None

    # checks passed; let's perform EPD. first get the number of apertures.
    names = ilc.dtype.names
    n_apertures = len([n for n in names if 'IRM' in n])
    irm_ap_keys = ['IRM{}'.format(i) for i in range(1,n_apertures+1)]

    # get the indices where all columns are non-nan
    combinedok = (npisfinite(ilc['XIC']) &
                  npisfinite(ilc['YIC']) &
                  npisfinite(ilc['FSV']) &
                  npisfinite(ilc['FDV']) &
                  npisfinite(ilc['FKV']))
    for irm_ap_key in irm_ap_keys:
        combinedok &= npisfinite(ilc[irm_ap_key])

    # get temperatures (optional)
    if observatory == 'tess':
        temperatures = ilc['CCDTEMP'][combinedok]
    elif observatory == 'hatpi':
        temperatures = None
    else:
        raise NotImplementedError('observatory must be "tess" or "hatpi"')

    # get the EPD diff mags
    isfull = True if 'FULL' in fitsilcfile else False
    epddiffmags, epddetails = {}, {}
    for irm_ap_key in irm_ap_keys:

        if observatory=='hatpi':
            epddiffmags[irm_ap_key] = ism.epd_magseries_imagesub(
                ilc[irm_ap_key][combinedok],
                ilc['FSV'][combinedok],
                ilc['FDV'][combinedok],
                ilc['FKV'][combinedok],
                ilc['XIC'][combinedok],
                ilc['YIC'][combinedok],
                smooth=smooth, sigmaclip=sigmaclip,
                observatory=observatory, temperatures=temperatures,
                isfull=isfull
            )
        elif observatory=='tess':
            giveallnan = True
            if len(ilc[irm_ap_key][combinedok])>0:
                try:
                    (epddiffmags[irm_ap_key],
                     epddetails[irm_ap_key]) = ism.epd_magseries_imagesub(
                        ilc[irm_ap_key][combinedok],
                        ilc['FSV'][combinedok],
                        ilc['FDV'][combinedok],
                        ilc['FKV'][combinedok],
                        ilc['XIC'][combinedok],
                        ilc['YIC'][combinedok],
                        smooth=smooth, sigmaclip=sigmaclip,
                        observatory=observatory,
                        temperatures=temperatures[combinedok],
                        times=ilc['TMID_BJD'][combinedok],
                        isfull=isfull
                    )
                    giveallnan = False
                except IndexError:
                    # sometimes temperature array gets wonky. for the few such
                    # cases, give an all-nan epd column.
                    giveallnan = True
                    pass

            if giveallnan:
                print('WRN! for {}, all nan EPD columns'.format(fitsilcfile))
                # all nans. vartools TFA needs the column populated (and it
                # should be anyway).
                epddiffmags[irm_ap_key] = (
                    np.nan * np.ones_like(ilc[irm_ap_key])
                )
                epddetails[irm_ap_key] = {}

                orbitinds = [0,1] if isfull else [0]

                for orbitind in orbitinds:
                    epddetails[irm_ap_key][orbitind] = {}

                    for k in ['tckp', 'u', 'n_knots', 'chisq', 'n_data',
                              'k_freeparam', 'BIC', 'x_pred', 'y_pred',
                              'T_pred', 'mag_pred']:

                        epddetails[irm_ap_key][orbitind][k] = str(np.nan)

                    epddetails[irm_ap_key][orbitind]['mag_pred'] = (
                        np.nan * np.ones_like(ilc[irm_ap_key])
                    )
        else:
            raise NotImplementedError(
                'EPD must be worked out on per-observatory basis.'
            )

    # assert that epddiffmags and epddetails have same keys
    err_msg = ('ERR on {}. epddiffmags and epddetails must have same keys'.
               format(fitsilcfile))

    np.testing.assert_array_equal(
        np.sort(list(epddiffmags.keys())), np.sort(list(epddetails.keys())),
        err_msg=err_msg
    )

    # add the EPD diff mags back to the median mag to get the EPD mags
    epdmags = {}
    for irm_ap_key in irm_ap_keys:

        if irm_ap_key not in epddiffmags:
            print('ERR! EPD failed for {} b/c epddiffmags had no key.'.
                  format(fitsilcfile))
            return 42
        else:
            pass

        if epddiffmags[irm_ap_key] is not None:
            mag_median = np.nanmedian(ilc[irm_ap_key])
            epdmags[irm_ap_key] = epddiffmags[irm_ap_key] + mag_median
        else:
            epdmags[irm_ap_key] = np.array(
                [np.nan for x in ilc[irm_ap_key][combinedok]]
            )
            print('WRN! %sZ: no %s mags available for %s!' %
                  (datetime.utcnow().isoformat(), irm_ap_key, fitsilcfile))

    # write the EPD LCs out to the outfile if given, else default is overwrite
    if not outfile:
        outfile = fitsilcfile

    # create the "EP1", "EP2", "EPN" keys, format keys, and data columns.
    epdnames = [k.replace('IRM','EP') for k in irm_ap_keys]
    epdformats = ['D'] * len(epdnames)
    epddatacols = [epdmags[k] for k in irm_ap_keys]

    epdcollist = [fits.Column(name=n, format=f, array=a) for n,f,a in
                  zip(epdnames, epdformats, epddatacols)]

    epdhdu = fits.BinTableHDU.from_columns(epdcollist)

    new_columns = inhdulist[1].columns + epdhdu.columns
    new_timeseries_hdu = fits.BinTableHDU.from_columns(new_columns)

    # Update the new timeseries HDU header with the info from the EPD fit:
    # 'n_knots', 'chisq', 'n_data', 'k_freeparam'. Give it for each orbit, for
    # each aperture, and include comments.
    # Note that we are NOT including sufficient info to reconstruct the fit.
    # The predicted vectors would give an extra (Ndim-1)*(3 apertures)
    # time-series vectors. And the "tck, u" representation would require an
    # annoying extra two extensions.
    orbitnums = [0,1] if isfull else [0]
    for orbitnum in orbitnums:
        for irm_ap_key in irm_ap_keys:
            knotkey = 'nknot_o{}_{}'.format(orbitnum, irm_ap_key)
            chisqkey = 'chisq_o{}_{}'.format(orbitnum, irm_ap_key)
            ndatakey = 'ndata_o{}_{}'.format(orbitnum, irm_ap_key)
            nfp = 'nfp_o{}_{}'.format(orbitnum, irm_ap_key)
            bickey = 'BIC_o{}_{}'.format(orbitnum, irm_ap_key)

            knotc = 'N knots in EPD spline fit (orbit/aper)'
            chisqc = 'chi^2 in EPD spline fit (orbit/aper)'
            ndatac = 'N data pts in EPD spline fit (orbit/aper)'
            nfpc = 'N free params in EPD spline fit (orbit/aper)'
            chisqc = 'BIC in EPD spline fit (orbit/aper)'

            tempkeys = ['n_knots', 'chisq', 'n_data', 'k_freeparam', 'BIC']

            for k, c, tk in zip(
                [knotkey, chisqkey, ndatakey, nfp],
                [knotc, chisqc, ndatac, nfpc],
                tempkeys
            ):
                new_timeseries_hdu.header[k] = (
                    epddetails[irm_ap_key][orbitnum][tk], c
                )

    # update the flag for whether detrending has been performed
    primary_hdu.header['DTR_EPD'] = True

    outhdulist = fits.HDUList([primary_hdu, new_timeseries_hdu])
    if os.path.exists(outfile):
        os.remove(outfile)
    outhdulist.writeto(outfile, overwrite=False)

    inhdulist.close()

    if outfile == fitsilcfile:
        n_epd_mags = len(
            epdmags[irm_ap_keys[0]][npisfinite(epdmags[irm_ap_keys[0]])]
        )
        print('{}: overwrote {} with {} EPD mags'.format(
            datetime.utcnow().isoformat(), outfile, n_epd_mags))
    else:
        n_epd_mags = len(
            epdmags[irm_ap_keys[0]][npisfinite(epdmags[irm_ap_keys[0]])]
        )
        print('{}: wrote {} with {} EPD mags'.format(
            datetime.utcnow().isoformat(), outfile, n_epd_mags)
        )

    return 1


##################
# TFA DETRENDING #
##################

def _make_tfa_lc_list(lcfiles, statsdir):
    """
    lc_list_tfa = List of input light curve files to process. The filenames are
    in the first column, the X coordinates of the stars (on the reference
    image) are in the second column, and the Y coordinates are in the third
    column. e.g.
    """

    # silly way to populate the lists, but we're I/O limited here anyway
    xcc, ycc = [], []
    for lcfile in lcfiles:
        d = iu.get_header_keyword_list(lcfile, ['XCC','YCC'], ext=0)
        xcc.append(d['XCC'])
        ycc.append(d['YCC'])
    xcc, ycc = nparr(xcc), nparr(ycc)

    df = pd.DataFrame({'fname':lcfiles,'xcc':xcc,'ycc':ycc})

    outpath = os.path.join(statsdir, 'lc_list_tfa.txt')
    df.to_csv(outpath, index=False, header=False, sep=' ')
    print('made {}'.format(outpath))

    return outpath


def _make_trendlist_tfa(templatefiles, statsdir):
    """
    trendlist_tfa.txt columns are:
        path to template lightcurve; XCC of template LC; YCC ditto.

    function returns:
        list of paths to trendlist_tfa_ap{}.txt files
    """

    n_apertures = len(templatefiles)

    # iterate over "templatefiles", which for each aperture have the selected
    # template stars, plus some other info (made by
    # aperturephot.choose_tfa_template)
    outpaths = []
    for templatefile, apind in zip(
        np.sort(templatefiles), range(1,n_apertures+1)):

        lcpaths, xcc, ycc = [], [], []

        df = pd.read_csv(templatefile, sep=' ',
                         names=['gaiaid', 'lcpath', 'aperture_mag','rms',
                                'ndet', 'xi', 'eta'])

        for lcpath in df['lcpath']:
            d = iu.get_header_keyword_list(lcpath, ['XCC','YCC'], ext=0)
            xcc.append(d['XCC'])
            ycc.append(d['YCC'])
            lcpaths.append(lcpath)

        xcc, ycc = nparr(xcc), nparr(ycc)

        outdf = pd.DataFrame({'fname':lcpaths,'xcc':xcc,'ycc':ycc})

        outpath = os.path.join(statsdir, 'trendlist_tfa_ap{}.txt'.
                               format(apind))
        outdf.to_csv(outpath, index=False, header=False, sep=' ')
        print('made {}'.format(outpath))
        outpaths.append(outpath)

    return outpaths


def _make_dates_tfa(fitsdir, fitsimgglob, statsdir):
    """
    dates_tfa.txt: list of all the image identifiers and times in the data set.
    these are the SAME for every (complete) lightcurve.

    the most obvious (and complete) way to generate them is directly from the
    images.

    tess2018241202941-s0002-4-4-0121_cal_img_bkgdsub 1360.3540539999999
    """

    imgfiles = glob(os.path.join(fitsdir, fitsimgglob))

    framekeyarr = nparr([os.path.basename(f).rstrip('.fits')
                         for f in imgfiles])

    # in BTJD
    tstartarr = nparr(
        [iu.get_header_keyword(f, 'TSTART', ext=0) for f in imgfiles]
    )

    outdf = pd.DataFrame({'framekey':framekeyarr,'tstart':tstartarr})

    outpath = os.path.join(statsdir, 'dates_tfa.txt')
    outdf.to_csv(outpath, index=False, header=False, sep=' ')
    print('made {}'.format(outpath))

    return outpath


def make_ascii_files_for_vartools(lcfiles, templatefiles, statsdir, fitsdir,
                                  fitsimgglob, fixedtfatemplate=None):
    """
    lc_list_tfa: List of input light curve files to process.

    trendlist_tfa: path to template lightcurve; XCC of template LC; YCC ditto.
    (so that they're not too close to target star).

    dates_tfa: list of all the image identifiers and times in the data set.

    Kwargs:
        fixedtfatemplate (str / None): if str, this is a path to a directory
        already containing stats_files/trendlist_tfa_ap[1-3].txt files. These
        files are symlinked to the current working directory and used for the
        template stars.
    """

    tfalclist_path = _make_tfa_lc_list(lcfiles, statsdir)

    # either make new lists of template stars, or use old ones if passed
    # fixedtfatemplate
    if not fixedtfatemplate:
        trendlisttfa_paths = _make_trendlist_tfa(templatefiles, statsdir)

    elif fixedtfatemplate and isinstance(fixedtfatemplate,str):
        pathstolink = glob(os.path.join(
            fixedtfatemplate,'trendlist_tfa_ap*.txt'))

        if len(pathstolink) == 0 :
            raise AssertionError('got no paths to link')

        dsts = []
        for src in pathstolink:

            dst = os.path.join(statsdir, os.path.basename(src))
            dsts.append(dst)

            # now open the file, and replace all instances of e.g.,
            # "/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/ISP_1-2-1163/2321703984136309248_llc.fits"
            # with for example:
            # "/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/ISP_1-2-1192/2321703984136309248_llc.fits"
            with open(src, 'r') as f:
                inlines = f.readlines()

            pres = search('{}/ISP_{}/{}', inlines[0])
            srcprojidstr = pres[1]

            pres = search('{}/ISP_{}/{}', statsdir)
            dstprojidstr = pres[1]

            outlines = []
            print('replacing {} with {}...'.format(srcprojidstr, dstprojidstr))
            for l in inlines:
                outlines.append(l.replace(srcprojidstr, dstprojidstr))

            with open(dst, 'w') as f:
                f.writelines(outlines)
            print('made {} from {} source'.format(dst, src))

        trendlisttfa_paths = dsts

    else:
        raise NotImplementedError

    datestfa_path = _make_dates_tfa(fitsdir, fitsimgglob, statsdir)

    return tfalclist_path, trendlisttfa_paths, datestfa_path


# should work with public vartools version
tfablslskillharmcmd = (
    '/home/jhartman/SVN/HATpipe/bin/vartools -l {lc_list_tfa} -matchstringid '
    ' -inputlcformat BGE:BGE:double,BGV:BGV:double,FDV:FDV:double,'
    'FKV:FKV:double,FSV:FSV:double,IFE1:IFE1:double,IFE2:IFE2:double,'
    'IFE3:IFE3:double,IFL1:IFL1:double,IFL2:IFL2:double,IFL3:IFL3:double,'
    'IRE1:IRE1:double,IRE2:IRE2:double,IRE3:IRE3:double,IRM1:IRM1:double,'
    'IRM2:IRM2:double,IRM3:IRM3:double,IRQ1:IRQ1:string,IRQ2:IRQ2:string,'
    'IRQ3:IRQ3:string,id:RSTFC:string,RSTFC:RSTFC:string,TMID_UTC:TMID_UTC:double,'
    'XIC:XIC:double,YIC:YIC:double,CCDTEMP:CCDTEMP:double,'
    'NTEMPS:NTEMPS:int,'
    't:TMID_BJD:double,TMID_BJD:TMID_BJD:double,BJDCORR:BJDCORR:double,'
    'EP1:EP1:double,EP2:EP2:double,EP3:EP3:double '
    '-expr \'mag=EP1\' -expr \'err=IRE1\' '
    '-TFA {trendlist_tfa_ap1} readformat 0 RSTFC EP1 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA1=mag\' '
    '-expr \'mag=EP2\' -expr \'err=IRE2\' '
    '-TFA {trendlist_tfa_ap2} readformat 0 RSTFC EP2 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA2=mag\' '
    '-expr \'mag=EP3\' -expr \'err=IRE3\' '
    '-TFA {trendlist_tfa_ap3} readformat 0 RSTFC EP3 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA3=mag\' '
    '-stats TFA1,TFA2,TFA3 stddev,MAD '
    '-if \'(STATS_TFA1_MAD_12<=STATS_TFA2_MAD_12)'
        '*(STATS_TFA1_MAD_12<=STATS_TFA3_MAD_12)\' '
        '-expr \'mag=TFA1\' -expr \'err=IRE1\' '
        '-BLS q {blsqmin} {blsqmax} {blsminper} {blsmaxper} {blsnfreq} {blsnbins} '
        '0 5 0 1 {outblsmodeldir_iter1} 0 fittrap '
        '-LS {lsminp} {lsmaxp} {lssubsample} 5 0 '
        '-Killharm ls {killharmnharm} 0 0 '
        '-expr \'HARMFILTER=mag\' '
        '-BLS q {blsqmin} {blsqmax} {blsminper} {blsmaxper} {blsnfreq} {blsnbins} '
        '0 5 0 1 {outblsmodeldir_iter2} 0 fittrap '
    '-elif \'(STATS_TFA2_MAD_12<=STATS_TFA1_MAD_12)'
        '*(STATS_TFA2_MAD_12<=STATS_TFA3_MAD_12)\' '
        '-expr \'mag=TFA2\' -expr \'err=IRE2\' '
        '-BLS q {blsqmin} {blsqmax} {blsminper} {blsmaxper} {blsnfreq} {blsnbins} '
        '0 5 0 1 {outblsmodeldir_iter1} 0 fittrap '
        '-LS {lsminp} {lsmaxp} {lssubsample} 5 0 '
        '-Killharm ls {killharmnharm} 0 0 '
        '-expr \'HARMFILTER=mag\' '
        '-BLS q {blsqmin} {blsqmax} {blsminper} {blsmaxper} {blsnfreq} {blsnbins} '
        '0 5 0 1 {outblsmodeldir_iter2} 0 fittrap '
    '-else '
        '-expr \'mag=TFA3\' -expr \'err=IRE3\' '
        '-BLS q {blsqmin} {blsqmax} {blsminper} {blsmaxper} {blsnfreq} {blsnbins} '
        '0 5 0 1 {outblsmodeldir_iter1} 0 fittrap '
        '-LS {lsminp} {lsmaxp} {lssubsample} 5 0 '
        '-Killharm ls {killharmnharm} 0 0 '
        '-expr \'HARMFILTER=mag\' '
        '-BLS q {blsqmin} {blsqmax} {blsminper} {blsmaxper} {blsnfreq} {blsnbins} '
        '0 5 0 1 {outblsmodeldir_iter2} 0 fittrap '
    '-fi '
    '-stats HARMFILTER stddev,MAD '
    ''
    '-o {outlcdir_tfa} columnformat RSTFC:\"Image\",'
    'TMID_BJD:\"BJDTDB midexp time\",TFA1:mag,TFA2:mag,TFA3:mag,HARMFILTER:mag '
    'fits copyheader logcommandline -header -numbercolumns -parallel {nproc} '
    '> {outstatsfile}'
)


tfaonlycmd = (
    '/home/jhartman/SVN/HATpipe/bin/vartools -l {lc_list_tfa} -matchstringid '
    ' -inputlcformat BGE:BGE:double,BGV:BGV:double,FDV:FDV:double,'
    'FKV:FKV:double,FSV:FSV:double,IFE1:IFE1:double,IFE2:IFE2:double,'
    'IFE3:IFE3:double,IFL1:IFL1:double,IFL2:IFL2:double,IFL3:IFL3:double,'
    'IRE1:IRE1:double,IRE2:IRE2:double,IRE3:IRE3:double,IRM1:IRM1:double,'
    'IRM2:IRM2:double,IRM3:IRM3:double,IRQ1:IRQ1:string,IRQ2:IRQ2:string,'
    'IRQ3:IRQ3:string,id:RSTFC:string,RSTFC:RSTFC:string,TMID_UTC:TMID_UTC:double,'
    'XIC:XIC:double,YIC:YIC:double,CCDTEMP:CCDTEMP:double,'
    'NTEMPS:NTEMPS:int,'
    't:TMID_BJD:double,TMID_BJD:TMID_BJD:double,BJDCORR:BJDCORR:double,'
    'EP1:EP1:double,EP2:EP2:double,EP3:EP3:double '
    '-expr \'mag=EP1\' -expr \'err=IRE1\' '
    '-TFA {trendlist_tfa_ap1} readformat 0 RSTFC EP1 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA1=mag\' '
    '-expr \'mag=EP2\' -expr \'err=IRE2\' '
    '-TFA {trendlist_tfa_ap2} readformat 0 RSTFC EP2 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA2=mag\' '
    '-expr \'mag=EP3\' -expr \'err=IRE3\' '
    '-TFA {trendlist_tfa_ap3} readformat 0 RSTFC EP3 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA3=mag\' '
    '-stats TFA1,TFA2,TFA3 stddev,MAD '
    '-o {outlcdir_tfa} columnformat RSTFC:\"Image\",'
    'TMID_BJD:\"BJDTDB midexp time\",TFA1:mag,TFA2:mag,TFA3:mag '
    'fits copyheader logcommandline -header -numbercolumns -parallel {nproc} '
    '> {outstatsfile}'
)


tfafromirmcmd = (
    '/home/jhartman/SVN/HATpipe/bin/vartools -l {lc_list_tfa} -matchstringid '
    ' -inputlcformat BGE:BGE:double,BGV:BGV:double,FDV:FDV:double,'
    'FKV:FKV:double,FSV:FSV:double,IFE1:IFE1:double,IFE2:IFE2:double,'
    'IFE3:IFE3:double,IFL1:IFL1:double,IFL2:IFL2:double,IFL3:IFL3:double,'
    'IRE1:IRE1:double,IRE2:IRE2:double,IRE3:IRE3:double,IRM1:IRM1:double,'
    'IRM2:IRM2:double,IRM3:IRM3:double,IRQ1:IRQ1:string,IRQ2:IRQ2:string,'
    'IRQ3:IRQ3:string,id:RSTFC:string,RSTFC:RSTFC:string,TMID_UTC:TMID_UTC:double,'
    'XIC:XIC:double,YIC:YIC:double,CCDTEMP:CCDTEMP:double,'
    'NTEMPS:NTEMPS:int,'
    't:TMID_BJD:double,TMID_BJD:TMID_BJD:double,BJDCORR:BJDCORR:double '
    '-expr \'mag=IRM1\' -expr \'err=IRE1\' '
    '-TFA {trendlist_tfa_ap1} readformat 0 RSTFC IRM1 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA1=mag\' '
    '-expr \'mag=IRM2\' -expr \'err=IRE2\' '
    '-TFA {trendlist_tfa_ap2} readformat 0 RSTFC IRM2 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA2=mag\' '
    '-expr \'mag=IRM3\' -expr \'err=IRE3\' '
    '-TFA {trendlist_tfa_ap3} readformat 0 RSTFC IRM3 {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA3=mag\' '
    '-stats TFA1,TFA2,TFA3 stddev,MAD '
    '-o {outlcdir_tfa} columnformat RSTFC:\"Image\",'
    'TMID_BJD:\"BJDTDB midexp time\",TFA1:mag,TFA2:mag,TFA3:mag '
    'fits copyheader logcommandline -header -numbercolumns -parallel {nproc} '
    '> {outstatsfile}'
)

inittfafromirmcmd = (
    '/home/jhartman/SVN/HATpipe/bin/vartools -l {init_lc_list_tfa} -matchstringid '
    ' -inputlcformat BGE:BGE:double,BGV:BGV:double,FDV:FDV:double,'
    'FKV:FKV:double,FSV:FSV:double,IFE1:IFE1:double,IFE2:IFE2:double,'
    'IFE3:IFE3:double,IFL1:IFL1:double,IFL2:IFL2:double,IFL3:IFL3:double,'
    'IRE1:IRE1:double,IRE2:IRE2:double,IRE3:IRE3:double,IRM1:IRM1:double,'
    'IRM2:IRM2:double,IRM3:IRM3:double,IRQ1:IRQ1:string,IRQ2:IRQ2:string,'
    'IRQ3:IRQ3:string,id:RSTFC:string,RSTFC:RSTFC:string,TMID_UTC:TMID_UTC:double,'
    'XIC:XIC:double,YIC:YIC:double,CCDTEMP:CCDTEMP:double,'
    'NTEMPS:NTEMPS:int,'
    't:TMID_BJD:double,TMID_BJD:TMID_BJD:double,BJDCORR:BJDCORR:double '
    '-expr \'mag=IRM{apnum}\' -expr \'err=IRE{apnum}\' '
    '-TFA {trendlist_tfa_ap_N} readformat 0 RSTFC IRM{apnum} {dates_tfa} '
    '{npixexclude} xycol 2 3 1 0 0 '
    '-expr \'TFA{apnum}=mag\' '
    '-stats TFA{apnum} stddev,MAD '
    '-o {outlcdir_tfa} columnformat RSTFC:\"Image\",'
    'TMID_BJD:\"BJDTDB midexp time\",TFA{apnum}:mag '
    'fits copyheader logcommandline -header -numbercolumns -parallel {nproc} '
    '> {outstatsfile}'
)


def run_tfa(tfalclist_path, trendlisttfa_paths, datestfa_path, lcdirectory,
            statsdir,
            nworkers=16, do_bls_ls_killharm=True, npixexclude=10,
            blsqmin=0.002, blsqmax=0.1, blsminper=0.2, blsmaxper=30.0,
            blsnfreq=20000, blsnbins=1000, lsminp=0.1, lsmaxp=30.0,
            lssubsample=0.1, killharmnharm=10, tfafromirm=False,
            outlcdir_tfa=None):
    """
    Run TFA on all apertures. Optionally, if do_bls_ls_killharm, include a
    sequence of BLS, then Lomb-Scargle, then harmonic killing, then BLS on the
    harmonic-subtracted residual. If tfafromirm is True, TFA is called on IRM
    mags, otherwise it is called on EPD mags.

    If running TFA alone, of order ~10k lightcurves per minute are created in
    os.path.join(lcdirectory, 'TFA_LCS'). If also doing BLS etc, it's of order
    50-100 processed lightcurves per minute. Periodograms are expensive.

    TFA parameters:
        npixexclude: trend stars exclusion radius, in units of pixels.

    BLS parameters
        blsqmin: minimum transit duration in phase units
        blsqmax
        blsminper
        blsmaxper
        blsnfreq
        blsnbins

    GLS parameters
        lsminp
        lsmaxp
        lssubsample

    Killharm parameters
        killharmnharm
    """

    trendlist_tfa_ap1 = [t for t in trendlisttfa_paths if 'ap1' in t][0]
    trendlist_tfa_ap2 = [t for t in trendlisttfa_paths if 'ap2' in t][0]
    trendlist_tfa_ap3 = [t for t in trendlisttfa_paths if 'ap3' in t][0]

    outblsmodeldir_iter1 = os.path.join(lcdirectory,'BLS_MODEL_ITER1')
    outblsmodeldir_iter2 = os.path.join(lcdirectory,'BLS_MODEL_ITER2')
    if not isinstance(outlcdir_tfa,str):
        outlcdir_tfa = os.path.join(lcdirectory,'TFA_LCS')
    outstatsfile = os.path.join(statsdir, 'vartools_tfa_stats.txt')

    for d in [outblsmodeldir_iter1, outblsmodeldir_iter2, outlcdir_tfa]:
        if not os.path.exists(d):
            os.mkdir(d)

    if do_bls_ls_killharm:
        cmdtorun = tfablslskillharmcmd.format(
            lc_list_tfa = tfalclist_path,
            trendlist_tfa_ap1 = trendlist_tfa_ap1,
            trendlist_tfa_ap2 = trendlist_tfa_ap2,
            trendlist_tfa_ap3 = trendlist_tfa_ap3,
            dates_tfa = datestfa_path,
            npixexclude = npixexclude,
            outblsmodeldir_iter1 = outblsmodeldir_iter1,
            outblsmodeldir_iter2 = outblsmodeldir_iter2,
            outlcdir_tfa = outlcdir_tfa,
            outstatsfile = outstatsfile,
            nproc = nworkers,
            blsqmin = blsqmin,
            blsqmax = blsqmax,
            blsminper = blsminper,
            blsmaxper = blsmaxper,
            blsnfreq = blsnfreq,
            blsnbins = blsnbins,
            lsminp = lsminp,
            lsmaxp = lsmaxp,
            lssubsample = lssubsample,
            killharmnharm = killharmnharm
        )
    else:

        if tfafromirm:
            cmdtorun = tfafromirmcmd.format(
                lc_list_tfa = tfalclist_path,
                trendlist_tfa_ap1 = trendlist_tfa_ap1,
                trendlist_tfa_ap2 = trendlist_tfa_ap2,
                trendlist_tfa_ap3 = trendlist_tfa_ap3,
                dates_tfa = datestfa_path,
                npixexclude = npixexclude,
                outblsmodeldir_iter1 = outblsmodeldir_iter1,
                outblsmodeldir_iter2 = outblsmodeldir_iter2,
                outlcdir_tfa = outlcdir_tfa,
                outstatsfile = outstatsfile,
                nproc = nworkers
            )

        else:
            cmdtorun = tfaonlycmd.format(
                lc_list_tfa = tfalclist_path,
                trendlist_tfa_ap1 = trendlist_tfa_ap1,
                trendlist_tfa_ap2 = trendlist_tfa_ap2,
                trendlist_tfa_ap3 = trendlist_tfa_ap3,
                dates_tfa = datestfa_path,
                npixexclude = npixexclude,
                outblsmodeldir_iter1 = outblsmodeldir_iter1,
                outblsmodeldir_iter2 = outblsmodeldir_iter2,
                outlcdir_tfa = outlcdir_tfa,
                outstatsfile = outstatsfile,
                nproc = nworkers
            )

    print(cmdtorun)

    returncode = os.system(cmdtorun)

    if returncode == 0:
        print('{}Z: TFA+BLS+LS+KILLHARM cmd ran'.format(
              datetime.utcnow().isoformat()))
        return 1
    else:
        print('ERR! {}Z: TFA+BLS+LS+KILLHARM cmd failed'.format(
              datetime.utcnow().isoformat()))
        raise AssertionError
        return 0


def initial_tfa(trendlisttfa_paths, datestfa_path, lcdirectory, statsdir,
                nworkers=16, npixexclude=10, tfafromirm=True,
                outlcdir_tfa=None, N_to_remove=100):
    """
    Run initial TFA on candidate template stars. Then remove the "most
    variable" N_to_remove stars from the template star lists, as measured by
    their variability in the resulting TFA LCs.

    TFA parameters:
        npixexclude: trend stars exclusion radius, in units of pixels.
    """

    trendlist_tfa_ap1 = [t for t in trendlisttfa_paths if 'ap1' in t][0]
    trendlist_tfa_ap2 = [t for t in trendlisttfa_paths if 'ap2' in t][0]
    trendlist_tfa_ap3 = [t for t in trendlisttfa_paths if 'ap3' in t][0]

    outlcdir_tfa = os.path.join(lcdirectory,'INITIAL_TFA_LCS')
    outstatsfile = os.path.join(statsdir, 'initial_vartools_tfa_stats.txt')
    outlcdir_tfa_ap1 = os.path.join(lcdirectory,'INITIAL_TFA_LCS','ap1')
    outlcdir_tfa_ap2 = os.path.join(lcdirectory,'INITIAL_TFA_LCS','ap2')
    outlcdir_tfa_ap3 = os.path.join(lcdirectory,'INITIAL_TFA_LCS','ap3')

    outlcdirs = [outlcdir_tfa_ap1, outlcdir_tfa_ap2, outlcdir_tfa_ap3]

    for d in [outlcdir_tfa, outlcdir_tfa_ap1,
              outlcdir_tfa_ap2, outlcdir_tfa_ap3]:
        if not os.path.exists(d):
            os.mkdir(d)

    for ix, apnum in enumerate([1,2,3]):
        if tfafromirm:

            cmdtorun = inittfafromirmcmd.format(
                init_lc_list_tfa = trendlisttfa_paths[ix],
                trendlist_tfa_ap_N = trendlisttfa_paths[ix],
                apnum = apnum,
                dates_tfa = datestfa_path,
                npixexclude = npixexclude,
                outlcdir_tfa = outlcdirs[ix],
                outstatsfile = outstatsfile,
                nproc = nworkers
            )

        else:
            raise NotImplementedError

        print(cmdtorun)

        returncode = os.system(cmdtorun)

        if returncode == 0:
            print('{}Z: initial TFA{} cmd ran'.format(
                  datetime.utcnow().isoformat(), apnum))
        else:
            print('ERR! {}Z: initial TFA{} cmd failed'.format(
                  datetime.utcnow().isoformat(), apnum))
            raise AssertionError


def round_two_tfa_selection(cand_template_lcpaths, apnum,
                            lcdirectory,
                            trendlisttfa_path,
                            max_fap=0.001,
                            period_max=10.5,
                            n_min_tfa_templates=120,
                            n_desired_tfa_templates=200):
    """
    given a list of candidate TFA template star LC paths, with initial TFA
    applied to them (in lcutils.initial_tfa), construct a list of their
    eta-normal indices, and their LS FAPs.

    sort it. make raw/dtr/bkgd plots to verify the sort is sensible. then cut
    the lists down to ~200 template stars.

    args:

        cand_template_lcpaths: list of candidate TFA template star LC paths.
        these are LCs output by the vartools tfa call, and contain only RSTFC,
        TMID_BJD, and TFA{N}, for {N} the aperture number.

        apnum: integer aperture number

        lcdirectory: str used to get the reduced mags and bkgd measurements
        corresponding to each cand_template_lcpath

        period_max (float): maximum period at which LS peaks will be considered
        against max_fap.
    """

    from astrobase.varclass.varfeatures import lightcurve_ptp_measures
    from astrobase.lcmath import sigclip_magseries
    from astropy.stats import LombScargle
    from lcstatistics import plot_raw_tfa_bkgd

    from cdips.lcproc import mask_orbit_edges as moe

    eta_normals, eta_robusts, faps = [],[],[]
    print('beginning lomb scargle for ap{}'.format(apnum))
    for ix, cand_template_lcpath in enumerate(cand_template_lcpaths):

        if ix%10==0:
            print('{}/{}'.format(ix,len(cand_template_lcpaths)))

        data, hdr = iu.read_fits(cand_template_lcpath, ext=1)

        tfa_mag = data['TFA{}'.format(apnum)]
        tfa_time = data['TMID_BJD']

        sel = (
            np.isfinite(tfa_mag) &
            np.isfinite(tfa_time)
        )

        d = lightcurve_ptp_measures(tfa_time[sel], tfa_mag[sel],
                                    np.ones_like(tfa_time[sel]))

        #
        # Lomb scargle w/ uniformly weighted points.
        #
        try:
            _time, _mag = moe.mask_orbit_start_and_end(
                tfa_time[sel], tfa_mag[sel], orbitgap=1, expected_norbits=2,
                orbitpadding=6/(24), raise_expectation_error=False
            )

            _time, _mag, _ = sigclip_magseries(_time, _mag, _mag*1e-3,
                                               sigclip=[3,3], iterative=False,
                                               niterations=None,
                                               meanormedian='median',
                                               magsarefluxes=False)

            ls = LombScargle(_time, _mag, _mag*1e-3)
            freq, power = ls.autopower()
            period = 1/freq

            #
            # Say FAP = 0.4%. Then "under the assumption that there is no periodic
            # signal in the data, we will observe a peak this high or higher
            # approximately 0.4% of the time, which gives a strong indication that
            # a periodic signal is present in the data."
            #
            # Lots of power at the TESS orbital period (13.7 days) is bad. So only
            # consider the FAP of the peak below period_max, which empirically
            # looks like about 10.5 days.
            fap = ls.false_alarm_probability(power[period < period_max].max())

            eta_normals.append(d['eta_normal'])
            eta_robusts.append(d['eta_robust'])
            faps.append(fap)

        except AssertionError as e:
            # trimming failed because of bad gaps. assign a FAP to ensure this
            # isn't selected as a template star. then continue.
            print('AssertionError: failed to mask orbit for {}'.
                  format(cand_template_lcpath))
            eta_normals.append(-1)
            eta_robusts.append(-1)
            faps.append(0)

    df = pd.DataFrame({'eta_normal':np.array(eta_normals),
                       'eta_robust':np.array(eta_robusts),
                       'fap':np.array(faps),
                       'log10_fap':np.log10(faps),
                       'cand_template_lcpath':cand_template_lcpaths
                      })
    # 0.1% chance. Means some strong periodic variability that the TFA missed.
    # Seems to be 20-25% of TESS stars typically.
    throw_out = df['fap'] < max_fap
    df['throw_out'] = throw_out

    df = df.sort_values(by='log10_fap', ascending=True)

    print('making plots for tfa template check')
    for cand_template_lcpath, fap, eta_r, throw_out in zip(
        df['cand_template_lcpath'], df['fap'],
        df['eta_robust'], df['throw_out']
    ):

        if fap > 0.01:
            print('plotting: skip b/c high FAP {}'.
                  format(cand_template_lcpath))
            continue

        data, hdr = iu.read_fits(cand_template_lcpath, ext=1)
        tfa_mag = data['TFA{}'.format(apnum)]
        tfa_time = data['TMID_BJD']

        rawlcpath = os.path.join(lcdirectory,
                                 os.path.basename(cand_template_lcpath))
        rawdata, hdr = iu.read_fits(rawlcpath, ext=1)
        raw_mag = rawdata['IRM{}'.format(apnum)]
        raw_time = rawdata['TMID_BJD']
        bkgd_val = rawdata['BGV']

        savpath = cand_template_lcpath.replace('.fits','_raw_tfa_bkgd.png')
        if os.path.exists(savpath):
            continue

        customstr = 'FAP: {:.1e}, ETA: {:.1e}, OMIT? {}'.format(
            fap, eta_r, throw_out)

        plot_raw_tfa_bkgd(raw_time, raw_mag, tfa_mag, bkgd_val, apnum,
                          savpath=savpath, customstr=customstr,
                          tfatime=tfa_time)

    # create a backup of the original template stars, since we're going to
    # overwrite it.
    dst = trendlisttfa_path.replace('trendlist','stage1-trendlist')
    shutil.copyfile(trendlisttfa_path,dst)
    print('copied {}->{}'.format(trendlisttfa_path,dst))

    # generally should have enough stars.
    throw_out = df['fap'] < max_fap
    if not len(df[~throw_out]) > n_min_tfa_templates:
        errmsg = (
            'lcutils.round_two_tfa_selection: in selecting tfa template stars,'
            'found {} templates, but need at least {}'.
            format(len(df[~throw_out]), n_min_tfa_templates)
        )
        raise AssertionError(errmsg)

    if len(df[~throw_out]) > n_desired_tfa_templates:

        # pick template stars randomly from dataframe of stars that are not
        # variable.
        outdf = df[~throw_out].sample(n=n_desired_tfa_templates)

    else:

        # same, but default to fewer stars.
        outdf = df[~throw_out].sample(n=n_min_tfa_templates)

    currentdf = pd.read_csv(trendlisttfa_path, sep=' ',
                            names=['templatelcpath','x','y'])

    outdf['cand_template_lcpath'] = list(map(
        lambda x: x.replace('INITIAL_TFA_LCS/ap{}/'.format(apnum),''),
        outdf['cand_template_lcpath']
    ))

    outdf = outdf.merge(currentdf, how='left', left_on='cand_template_lcpath',
                        right_on='templatelcpath', suffixes=('_left','_right'))

    outcols = ['cand_template_lcpath', 'x', 'y']
    odf = outdf[outcols]

    odf.to_csv(
        trendlisttfa_path, header=None, index=False, sep=' '
    )
    print('rewrote {}'.format(trendlisttfa_path))



def merge_tfa_lc_worker(task):
    """
    overwrite the given FITS LC at lcpath with appended TFA magnitudes.
    """
    lcpath, tfalcdir = task

    lcbase = os.path.basename(lcpath)
    tfalcpath = os.path.join(tfalcdir, lcbase)

    # note: you can only merge things that have TFA LCs. so check if it exists.
    tfalcexists = os.path.exists(tfalcpath)
    lcexists = os.path.exists(lcpath)

    if tfalcexists and lcexists:

        tfahdulist = fits.open(tfalcpath)
        tfaprimaryhdr, tfahdr, tfadata = (
            tfahdulist[0].header, tfahdulist[1].header, tfahdulist[1].data
        )

        inhdulist = fits.open(lcpath)
        primaryhdr, hdr, data = (
            inhdulist[0].header, inhdulist[1].header, inhdulist[1].data
        )

        if primaryhdr['DTR_TFA']:
            print('WRN! {} found TFA had been performed; skipping'.
                  format(lcpath))
            return 0

        if np.array_equal(tfadata['RSTFC'],data['RSTFC']):
            tfadatacols = None
            pass
        elif (not np.array_equal(tfadata['RSTFC'],data['RSTFC']) and
            len(tfadata['RSTFC']) < len(data['RSTFC'])
        ):
            # TFA LC has too few frame stamps (for whatever reason -- often
            # because NaNs are treated differently by VARTOOLS TFA, so they are
            # omitted). The following code finds entries in the TFA lightcurve
            # that are missing frameids, and put nans in those cases. It does
            # this by first making an array of NaNs with length equal to the
            # original RAW data. It then puts the TFA values at the appropriate
            # indices, through the frame-id matching ("RSTFC" ids).
            inarr = np.in1d(data['RSTFC'], tfadata['RSTFC'])

            inds_to_put = np.argwhere(inarr).flatten()

            tfakeys = ['TFA1','TFA2','TFA3']
            tfadatacols = []
            for k in tfakeys:
                thiscol_short = tfadata[k]

                thiscol_full = (
                    np.ones_like(np.arange(len(data['RSTFC']),
                                           dtype=np.float))*np.nan
                )

                thiscol_full[ inds_to_put ] = thiscol_short

                tfadatacols.append(thiscol_full)

            wrn_msg = (
                'WRN! {}: found {} missing frameids. added NaNs'.
                format(lcpath, len(inarr[~inarr]))
            )
            print(wrn_msg)

        else:
            err_msg = (
                'ERR! {}: got bad frameids in TFA/MAIN LC. need to debug!'.
                format(lcpath)
            )
            print(err_msg)
            raise AssertionError

        # create the "TF1", "TF2", "TFN" keys, format keys, and data columns.
        n_apertures = len([t[1] for t in hdr.cards if 'IRM' in str(t[1])])
        irm_ap_keys = ['IRM{}'.format(i) for i in range(1,n_apertures+1)]
        tfanames = [k.replace('IRM','TFA') for k in irm_ap_keys]
        tfaformats = ['D'] * len(tfanames)
        if isinstance(tfadatacols,list):
            # We made the TFA data columns above, appropriately ordered. Skip.
            pass
        else:
            # We got np.array_equal(tfadata['RSTFC'],data['RSTFC']) true.
            tfadatacols = [tfadata[k] for k in tfanames]

        tfacollist = [fits.Column(name=n, format=f, array=a) for n,f,a in
                      zip(tfanames, tfaformats, tfadatacols)]

        tfahdu = fits.BinTableHDU.from_columns(tfacollist)

        new_columns = inhdulist[1].columns + tfahdu.columns
        new_timeseries_hdu = fits.BinTableHDU.from_columns(new_columns)

        # update the flag for whether detrending has been performed
        primaryhdr['DTR_TFA'] = True

        # update the history with the vartools invocation
        primaryhdr['HISTORY'] = str(tfaprimaryhdr['HISTORY']).replace('\n','')

        outhdulist = fits.HDUList([inhdulist[0], new_timeseries_hdu])
        outfile = lcpath
        if os.path.exists(outfile):
            os.remove(outfile)
        outhdulist.writeto(outfile, overwrite=False)

        inhdulist.close()
        tfahdulist.close()

        n_tfa_mags = len(tfadatacols[0])
        print('{}: overwrote {} with {} TFA mags'.format(
            datetime.utcnow().isoformat(), outfile, n_tfa_mags))

        return 1

    elif not tfalcexists and lcexists:
        # if the TFA LC does not exist for some reason, put in all nans to the
        # TFA columns.

        inhdulist = fits.open(lcpath)
        primaryhdr, hdr, data = (
            inhdulist[0].header, inhdulist[1].header, inhdulist[1].data
        )

        if primaryhdr['DTR_TFA']:
            print('{}: WRN! {} found TFA had been performed; skipping'.
                  format(datetime.utcnow().isoformat(),lcpath))
            return 0

        # create the "TF1", "TF2", "EPN" keys, format keys, and data columns.
        n_apertures = len([t[1] for t in hdr.cards if 'IRM' in str(t[1])])
        irm_ap_keys = ['IRM{}'.format(i) for i in range(1,n_apertures+1)]
        tfanames = [k.replace('IRM','TFA') for k in irm_ap_keys]
        tfaformats = ['D'] * len(tfanames)
        tfadatacols = [np.nan * np.ones_like(data[k]) for k in irm_ap_keys]

        tfacollist = [fits.Column(name=n, format=f, array=a) for n,f,a in
                      zip(tfanames, tfaformats, tfadatacols)]

        tfahdu = fits.BinTableHDU.from_columns(tfacollist)

        new_columns = inhdulist[1].columns + tfahdu.columns
        try:
            new_timeseries_hdu = fits.BinTableHDU.from_columns(new_columns)
        except ValueError as e:
            print(e)
            print('ValueError on {}'.format(lcpath))

        # update the flag for whether detrending has been performed
        primaryhdr['DTR_TFA'] = True

        outhdulist = fits.HDUList([inhdulist[0], new_timeseries_hdu])
        outfile = lcpath
        if os.path.exists(outfile):
            os.remove(outfile)
        outhdulist.writeto(outfile, overwrite=False)

        inhdulist.close()

        n_tfa_mags = len(tfadatacols[0])
        print('{}: WRN! wrote {} with all NaN TFA mags'.
              format(datetime.utcnow().isoformat(), outfile, n_tfa_mags))

        return 1

    else:
        raise NotImplementedError(
            'merge tfa lc called on {} and {}. neither exists.'.
            format(lcpath, tfalcpath)
        )
        return 0


def parallel_merge_tfa_lcs(lcdirectory, nworkers=32, maxworkertasks=1000):

    tfalcdir = os.path.join(lcdirectory, 'TFA_LCS')

    tfalcs = glob(os.path.join(tfalcdir,'*_llc.fits'))

    lcpaths = glob(os.path.join(lcdirectory, '*_llc.fits'))

    tasks = [(x, tfalcdir) for x in lcpaths]

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
    results = pool.map(merge_tfa_lc_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return 1


##################
# TIME UTILITIES #
##################
def parallel_apply_barycenter_time_correction(lcdirectory, nworkers=16,
                                              maxworkertasks=1000):

    fitsilcfiles = glob(os.path.join(lcdirectory,'*_llc.fits'))

    print('%sZ: %s lcs to apply barycenter time correction on' %
          (datetime.utcnow().isoformat(), len(fitsilcfiles)))

    tasks = [(x) for x in fitsilcfiles]

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
    results = pool.map(apply_barycenter_time_correction_worker, tasks)

    pool.close()
    pool.join()

    return {result for result in results}


def apply_barycenter_time_correction_worker(task):
    # wrapper for common API
    fitsilcfile = task
    result = apply_barycenter_time_correction(fitsilcfile)
    return result



def apply_barycenter_time_correction(fitsilcfile):
    """
    make the TMID_BJD and BARYCORR columns, and write it.
    """

    inhdulist = fits.open(fitsilcfile)
    primary_hdu = inhdulist[0]

    # utc_tmid calculated in ism.dump_lightcurves_with_grcollect
    tmid_utc = Time(inhdulist[1].data['TMID_UTC'], format='jd',
                    scale='utc')

    # coordinates from gaia
    ra = inhdulist[0].header['RA[deg]']*u.deg
    dec = inhdulist[0].header['Dec[deg]']*u.deg

    # get the midtime in BJD_TDB, and the barycentric correction
    tmid_bjd_tdb, ltt_bary = astropy_utc_time_to_bjd_tdb(
        tmid_utc, ra, dec, observatory='tess', get_barycorr=True
    )

    # write to FITS lightcurve table
    tmidcol = fits.Column(name='TMID_BJD', format='D', array=tmid_bjd_tdb)
    barycorrcol = fits.Column(name='BJDCORR', format='D', array=ltt_bary)

    timecollist = [tmidcol, barycorrcol]
    timehdu = fits.BinTableHDU.from_columns(timecollist)

    new_columns = inhdulist[1].columns + timehdu.columns
    new_timeseries_hdu = fits.BinTableHDU.from_columns(new_columns)

    # update primary HDU time meta-data
    primary_hdu.header['TIMESYS'] = 'TDB'
    primary_hdu.header['TIMEUNIT'] = 'd'
    primary_hdu.header['BTC_RA'] = ra.value
    primary_hdu.header['BTC_DEC'] = dec.value

    for k in ['BTC_RA','BTC_DEC']:
        primary_hdu.header.comments[k] = _map_key_to_comment(k.lower())

    outhdulist = fits.HDUList([primary_hdu, new_timeseries_hdu])
    if os.path.exists(fitsilcfile):
        try:
            os.remove(fitsilcfile)
        except OSError as e:
            # rare occasion, get OSError: [Errno 18] Invalid cross-device link.
            # might be an issue at the nfs level.
            print('{}Z: ERR! got {} in TMID_BJD write to {}. mv -> graveyard.'.
                  format(datetime.utcnow().isoformat(), repr(e), fitsilcfile)
            )
            graveyarddir = "/nfs/phtess1/ar1/TESS/FFI/GRAVEYARD"
            shutil.move(fitsilcfile,
                        os.path.join(graveyarddir,
                                     os.path.basename(fitsilcfile)))
            return 42

    outhdulist.writeto(fitsilcfile, overwrite=False)

    inhdulist.close()

    print('{}Z: wrote TMID_BJD and BARYCORR to {}'.
          format(datetime.utcnow().isoformat(), fitsilcfile)
    )

    return 1


def astropy_utc_time_to_bjd_tdb(tmid_utc, ra, dec, observatory='tess',
                                get_barycorr=False):
    """
    Args:

        tmid_utc (astropy.time.core.Time): measurement midtime in UTC
        reference, as an astropy Time object. It can be vector, e.g.,

            <Time object: scale='utc' format='jd' value=[2458363.41824751
            2458364.41824751]>

        ra, dec: with astropy units included.

        get_barycorr: if True, returns tmid_bjd_tdb, ltt_bary tuple.

    Returns:

        tmid_bjd_tdb (np.ndarray, or float) if not get_barycorr.

        else

        tmid_bjd_tdb, ltt_bary: tuple of midtimes in BJD_TDB, and the
        barycentric time corrections that were applied.
    """

    assert type(tmid_utc) == astropy.time.core.Time
    assert type(ra) == astropy.units.quantity.Quantity

    if observatory=='tess':
        # For TESS, assume the observatory is at the Earth's center. This
        # introduces a maximal timing error of
        #
        # (orbital major axis length)/(speed of light) ~= 80R_earth/c = 1.7 seconds.
        #
        # This is below the timing precision expected for pretty much anything
        # from our lightcurves.

        location = coord.EarthLocation.from_geocentric(0*u.m,0*u.m,0*u.m)
    else:
        raise NotImplementedError

    tmid_utc.location = location

    obj = coord.SkyCoord(ra, dec, frame='icrs')

    ltt_bary = tmid_utc.light_travel_time(obj)

    tmid_bjd_tdb = tmid_utc.tdb + ltt_bary

    if not get_barycorr:
        return tmid_bjd_tdb.jd
    else:
        return tmid_bjd_tdb.jd, ltt_bary.value


def run_pca(tfalclist_path, trendlisttfa_paths, datestfa_path, lcdirectory,
            statsdir,
            nworkers=16, do_bls_ls_killharm=True, npixexclude=10,
            blsqmin=0.002, blsqmax=0.1, blsminper=0.2, blsmaxper=30.0,
            blsnfreq=20000, blsnbins=1000, lsminp=0.1, lsmaxp=30.0,
            lssubsample=0.1, killharmnharm=10, tfafromirm=False,
            outlcdir_tfa=None):

    pass


def get_mag(fitspath, ap='IRM2'):
    with fits.open(fitspath) as hdulist:
        mag = hdulist[1].data[ap]
    return mag


def test_pca():

    sector = 6
    cam = 1
    ccd = 2
    projid = 1501

    # sector = 7
    # cam = 3
    # ccd = 1
    # projid = 1524

    lcdir = ('/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s{}/ISP_{}-{}-{}/'.
             format(str(sector).zfill(4), cam, ccd, projid))
    statsdir = os.path.join(lcdir, 'stats_files')
    outdir = os.path.join(statsdir, 'pca_verification')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # path, x, y. space-separated. for all ~30k light curves to do TFA on.
    tfalclist_path = os.path.join(statsdir,'lc_list_tfa.txt')
    # path, x ,y. space-separated. for 200 TFA template stars.
    trendlisttfa = os.path.join(statsdir,'trendlist_tfa_ap2.txt')
    # key, time
    datestfa_path = os.path.join(statsdir,'dates_tfa.txt')

    df_template_stars = pd.read_csv(trendlisttfa, sep=' ', header=None,
                                    names=['path','x','y'])

    df_dates = pd.read_csv(datestfa_path, sep=' ', header=None,
                           names=['rstfc','btjd'])

    lcpaths = glob(os.path.join(lcdir, '*_llc.fits'))

    # TODO: FIXME :  maybe select better template stars. and more of them...

    # prepare data as a (200 x N_times) matrix. We have N=200 light curves, with
    # K=N_times measurements in each. Think of these as N vectors in a
    # K-dimensional space. I.e. the flux at each time point is a "measured
    # feature". So we have N samples (light curves), and K features (points per
    # light curve).
    mags = nparr(list(map(get_mag, nparr(df_template_stars['path']))))
    mean_mags = np.nanmean(mags, axis=1)
    X = mags - mean_mags[:, None]

    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.linear_model import (LinearRegression, BayesianRidge,
                                      ARDRegression)
    from sklearn.model_selection import cross_val_score
    pca = PCA()
    pca.fit(X)
    fa = FactorAnalysis()
    fa.fit(X)

    eigenvals = pca.explained_variance_ratio_
    eigenvals_cs = eigenvals.cumsum()
    eigenvals_cs /= eigenvals_cs[-1]

    # (200 x N_times) eigenvector matrix. these are basis vectors for the original
    # data.
    #eigenvecs = pca.components_
    eigenvecs = fa.components_  #FIXME

    # plot a sequence of reconstructions, for a set of random light curves
    N_to_make = 15

    for i in range(N_to_make):

        np.random.seed(i)
        mag = get_mag(np.random.choice(lcpaths, size=1)[0])
        if np.all(pd.isnull(mag)):
            while np.all(pd.isnull(mag)):
                mag = get_mag(np.random.choice(lcpaths, size=1)[0])
        mean_mag = np.nanmean(mag)
        mag = mag - mean_mag

        component_list = [1, 2, 4, 8, 12, 16, 20]

        plt.close('all')
        f, axs = plt.subplots(nrows=len(component_list), ncols=2, sharex=True,
                              figsize=(8,9))

        for n_components, ax, ax_r in zip(component_list, axs[:,0], axs[:,1]):
            #
            # modelled off Figure 7.6 of the astroML book. (CITE)
            #

            #
            # eigenvecs shape: 200 x N_times
            #
            # model: 
            # y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.
            #
            # X is matrix of (n_samples, n_features).
            #

            # either linear regression or bayesian ridge regression seems fine
            reg = LinearRegression(fit_intercept=True)
            #reg = BayesianRidge(fit_intercept=True)
            #reg = ARDRegression(fit_intercept=True)

            y = mag
            _X = eigenvecs[:n_components, :]

            reg.fit(_X.T, y)

            model_mag = reg.intercept_ + (reg.coef_ @ _X)

            time = nparr(df_dates['btjd'])

            ax.scatter(time, mag + mean_mag, c='k', alpha=0.9,
                       zorder=2, s=1, rasterized=True, linewidths=0)
            ax.plot(time, model_mag + mean_mag, c='C0', zorder=1,
                    rasterized=True, lw=0.5, alpha=0.7 )

            txt = '{} components'.format(n_components)
            # txt += ', '
            # txt += r'$\sigma_{{\mathrm{{tot}}}}^2$='
            # txt += '{:.2f}'.format(eigenvals_cs[n_components - 1])
            ax.text(0.02, 0.02, txt, ha='left', va='bottom', fontsize='medium',
                    transform=ax.transAxes)

            ax_r.scatter(time, mag-model_mag, c='k', alpha=0.9,
                         zorder=2, s=1, rasterized=True, linewidths=0)
            ax_r.plot(time, model_mag-model_mag, c='C0', zorder=1, rasterized=True,
                      lw=0.5, alpha=0.7)


        for a in axs[:,0]:
            a.set_ylabel('raw mag')
        for a in axs[:,1]:
            a.set_ylabel('resid')
        for a in axs.flatten():
            a.get_yaxis().set_tick_params(which='both', direction='in')
            a.get_xaxis().set_tick_params(which='both', direction='in')

        f.text(0.5,-0.01, 'BJDTDB [days]', ha='center')

        f.tight_layout(h_pad=0, w_pad=0.5)
        savpath = os.path.join(outdir, 'test_reconstruction_{}.png'.format(i))
        f.savefig(savpath, dpi=350, bbox_inches='tight')
        print('made {}'.format(savpath))


    #
    # make plot to find optimal number of components.
    # see
    # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    #
    n_components = np.arange(0,50,1)

    def compute_scores(X):
        pca = PCA()
        fa = FactorAnalysis()
        pca_scores, fa_scores = [], []
        for n in n_components:
            pca.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
            fa.n_components = n
            fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))

        return pca_scores, fa_scores

    pca_scores, fa_scores = compute_scores(X)

    n_components_pca_cv = n_components[np.argmax(pca_scores)]
    n_components_fa_cv = n_components[np.argmax(fa_scores)]

    # # in commented code below, MLE component estimation only supported if
    # # n_samples >= n_features.
    # pca = PCA(svd_solver='full', n_components='mle')
    # pca.fit(X)
    # n_components_pca_mle = pca.n_components_
    # print('n_components_pca_mle: {}'.format(n_components_pca_mle))

    print('n_components_pca_cv: {}'.format(n_components_pca_cv))
    print('n_components_fa_cv: {}'.format(n_components_fa_cv))

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,3))

    ax.plot(n_components, pca_scores, label='PCA CV score')
    ax.plot(n_components, fa_scores, label='FA CV score')

    ax.legend(loc='best')
    ax.set_xlabel('N components')
    ax.set_ylabel('Cross-validation score')

    f.tight_layout(pad=0.2)
    savpath = os.path.join(outdir, 'test_optimal_n_components.png')
    f.savefig(savpath, dpi=350, bbox_inches='tight')
    print('made {}'.format(savpath))
