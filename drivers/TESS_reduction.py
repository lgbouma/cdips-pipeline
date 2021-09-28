# -*- coding: utf-8 -*-
"""
$ python TESS_reduction.py --help

contents:

main
        record_reduction_parameters
        parallel_trim_get_single_extension
        parallel_bkgd_subtract
        parallel_plot_median_filter_quad
        parallel_mask_saturated_stars
        (parallel_mask_dquality_flag_frames)
        make_ccd_temperature_timeseries_pickle
        parallel_append_ccd_temperature_to_hdr
    parallel_move_badframes
    get_files_needed_before_image_subtraction
        parallel_extract_sources
        parallel_astrometrydotnet
        make_fov_catalog
        reform_gaia_fov_catalog
        merge_object_catalog_vs_cdips
        parallel_fitsdir_photometry
    run_imagesubtraction
        parallel_frames_to_database
        framelist_make_xtrnsfits
        generate_photref_candidates_from_xtrns
        generate_combined_photref
        parallel_xtrnsfits_convsub
        parallel_convsubfits_staticphot
        dump_lightcurves_with_grcollect
    run_detrending
        parallel_convert_grcollect_to_fits_lc
        parallel_apply_barycenter_time_correction
        parallel_run_epd_imagesub_fits
        parallel_lc_statistics
        choose_tfa_template
        make_ascii_files_for_vartools
        initial_tfa
        round_two_tfa_selection
        run_tfa
        parallel_merge_tfa_lcs
        plot_stats_file
    assess_run
        percentiles_RMSorMAD_stats_and_plots
        parallel_compute_acf_statistics
        plot_random_lightcurves_and_ACFs
        acf_percentiles_stats_and_plots
        is_image_noise_gaussian
        plot_random_lightcurve_subsample
        are_known_planets_in_field
        measure_known_planet_SNR
        make_cluster_cutout_jpgs

minor functions:
    _plot_normalized_subtractedimg_histogram
    examine_astrometric_shifts
    is_imagesubtraction_complete
    is_presubtraction_complete
    initial_wcs_worked_well_enough
    make_fake_xtrnsfits
    _make_movies
    _get_random_acf_pkls
    _get_random_tfa_lcs
"""
from __future__ import division, print_function

import os, time, warnings
import matplotlib as mpl
mpl.use('AGG')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import aperturephot as ap, shared_variables as sv, \
       autoimagesub as ais, imagesubphot as ism, tessutils as tu, \
       lcstatistics as lcs, imageutils as iu, \
       lcutils as lcu, wcsqualityassurance as wcsqa
from glob import glob
from astropy.io import fits
from astropy import units as units, constants as constants
from astropy import wcs
from datetime import datetime
import argparse, re, pickle
from parse import parse, search

from numpy import array as nparr
from astropy.coordinates import SkyCoord

from shutil import which

import errno

####################
# HELPER FUNCTIONS #
####################

def _get_random_tfa_lcs(lcdirectory, n_desired_lcs=100):

    tfafiles = np.array(glob(os.path.join(lcdirectory,'*_llc.fits')))
    n_possible_lcs = len(tfafiles)

    # if you have fewer than `n_desired_lcs`, just get as many as possible.
    if n_possible_lcs == 0:
        raise AssertionError('need tfa LCs to exist to get any of them')
    if n_possible_lcs > n_desired_lcs:
        inds = np.random.randint(0, n_possible_lcs, n_desired_lcs)
        sel_tfafiles = tfafiles[inds]
    else:
        sel_tfafiles = tfafiles

    return sel_tfafiles


def _get_random_acf_pkls(pkldir, n_desired=10):

    np.random.seed(42)

    pklfiles = np.array(glob(os.path.join(pkldir,'*.pickle')))
    n_possible = len(pklfiles)

    # if you have fewer than `n_desired_lcs`, just get as many as possible.
    if n_possible == 0:
        raise AssertionError('need pkls to exist to get any of them')
    if n_possible > n_desired:
        inds = np.random.randint(0, n_possible, n_desired)
        sel_files = pklfiles[inds]
    else:
        sel_files = pklfiles

    return sel_files


def _make_movies(fitsdir, moviedir, field, camera, ccd, projectid,
                 DO_DEPRECATED=False):

    typestr = 'full' if 'FULL' in fitsdir else 'tune'

    # subtracted frame movies
    jpgglob = os.path.join(fitsdir,
                           'JPEG-SUB*CONV-*tess*cal_img_bkgdsub-xtrns.jpg')
    outmp4path = os.path.join(
        moviedir,
        '{:s}_{:s}_cam{:d}_ccd{:d}_projid{:d}_SUBTRACTEDCONV.mp4'.
        format(field, typestr, int(camera), int(ccd), int(projectid)))
    if not os.path.exists(outmp4path):
        if os.path.exists('/home/lbouma/bin/ffmpeg'):
            ffmpegpath='/home/lbouma/bin/ffmpeg'
        else:
            ffmpegpath=which('ffmpeg')
        if ffmpegpath != '':
            iu.make_mp4_from_jpegs(jpgglob, outmp4path,
                                   ffmpegpath=ffmpegpath)
        else:
            print('ffmpeg not found - cannot make movies')
    else:
        print('found {}'.format(outmp4path))

    # NGC-labelled stars; astrometry.net zscale
    jpgglob = os.path.join(fitsdir, 'tess*_cal_img_bkgdsub-ngc.png')
    outmp4path = os.path.join(
        moviedir,
        '{:s}_{:s}_cam{:d}_ccd{:d}_projid{:d}_NGC.mp4'.
        format(field, typestr, int(camera), int(ccd), int(projectid)))
    if not os.path.exists(outmp4path):
        if os.path.exists('/home/lbouma/bin/ffmpeg'):
            ffmpegpath='/home/lbouma/bin/ffmpeg'
        else:
            ffmpegpath=which('ffmpeg')
        if ffmpegpath != '':
            iu.make_mp4_from_jpegs(jpgglob, outmp4path,
                                   ffmpegpath=ffmpegpath)
        else:
            print('ffmpeg not found - cannot make movies')
    else:
        print('found {}'.format(outmp4path))

    # regular old translated (xtrns) frames, pre-subtraction, but masked
    jpgglob = os.path.join(fitsdir,
                           'JPEG-XTRNS-tess*_cal_img_bkgdsub-xtrns.jpg')
    outmp4path = os.path.join(
        moviedir,
        '{:s}_{:s}_cam{:d}_ccd{:d}_projid{:d}_XTRNS.mp4'.
        format(field, typestr, int(camera), int(ccd), int(projectid)))
    if not os.path.exists(outmp4path) and len(glob(jpgglob))>10:
        if os.path.exists('/home/lbouma/bin/ffmpeg'):
            ffmpegpath='/home/lbouma/bin/ffmpeg'
        else:
            ffmpegpath=which('ffmpeg')
        if ffmpegpath != '':
            iu.make_mp4_from_jpegs(jpgglob, outmp4path,
                                   ffmpegpath=ffmpegpath)
        else:
            print('ffmpeg not found - cannot make movies')
    else:
        print('found (or skipped) {}'.format(outmp4path))

    if DO_DEPRECATED:
        # cluster cut movies: (subtracted & grayscale), (subtracted & bwr), (CAL &
        # grayscale). first, get unique cluster names.  the pattern we're matching
        # is:
        # CUT-NGC_2516_rsub-9ab2774b-tess2018232225941-s0001-4-3-0120_cal_img_bkgdsub-xtrns_SUB_grayscale.jpg
        clusterjpgs = glob(os.path.join(
            fitsdir, 'CUT-*', 'CUT-*_[r|n]sub-*-tess2*_cal_img_bkgdsub*_SUBCONV.jpg'))
        if len(clusterjpgs)>1:

            clusternames = nparr(
                [search('CUT-{}_rsub-{}-{}',c)[0] for c in clusterjpgs]
            )
            uclusternames = np.sort(np.unique(clusternames))
            uclusternames = [u.split('/')[0] for u in uclusternames]

            # iterate over clusters
            for uclustername in uclusternames:

                # iterate over movie formats
                for jpgstr, outstr in zip(
                    ['CUT-{:s}_[r|n]sub-*-tess2*SUBCONV.jpg'.
                     format(uclustername),
                     'CUT-{:s}_[r|n]sub-*-tess2*BKGDSUB.jpg'.
                     format(uclustername),
                     'CUT-{:s}_tess2*CAL.jpg'.
                     format(uclustername)
                    ],
                    ['{:s}_{:s}_cam{:d}_ccd{:d}_projid{:d}_{:s}_SUBCONV.mp4'.
                     format(field, typestr, int(camera), int(ccd), int(projectid),
                            uclustername),
                    '{:s}_{:s}_cam{:d}_ccd{:d}_projid{:d}_{:s}_BKGDSUB.mp4'.
                     format(field, typestr, int(camera), int(ccd), int(projectid),
                            uclustername),
                    '{:s}_{:s}_cam{:d}_ccd{:d}_projid{:d}_{:s}_CAL.mp4'.
                     format(field, typestr, int(camera), int(ccd), int(projectid),
                            uclustername)
                    ]
                ):

                    jpgglob = os.path.join(fitsdir, 'CUT-*', jpgstr)
                    outmp4path = os.path.join(moviedir, outstr)
                    if not os.path.exists(outmp4path) and len(glob(jpgglob))>10:
                        if os.path.exists('/home/lbouma/bin/ffmpeg'):
                            ffmpegpath='/home/lbouma/bin/ffmpeg'
                        else:
                            ffmpegpath=which('ffmpeg')
                        if ffmpegpath != '':
                            iu.make_mp4_from_jpegs(jpgglob, outmp4path,
                                                       ffmpegpath=ffmpegpath)
                        else:
                            print('ffmpeg not found - cannot make movies')
                    else:
                        print('found (or skipped) {}'.format(outmp4path))

        else:
            print('WRN! did not make CUT movies, because did not find jpg matches')


def make_fake_xtrnsfits(fitsdir, fitsglob, fieldinfo):

    wcsfiles = glob(os.path.join(
        fitsdir, fitsglob.replace('.fits','.wcs')))

    fitstoshift = [w.replace('.wcs','.fits') for w in wcsfiles]
    outpaths = [f.replace('.fits','-xtrns.fits') for f in fitstoshift]

    print('making SYMLINKS to FAKE "xtrnsfits" files...')

    for src, dst in zip(fitstoshift, outpaths):
        os.symlink(src, dst)
        print('{} --> {}'.format(src, dst))


    # make identity *.itrans files

    # find this frame's associated active astromref
    framearef = ais.dbget_astromref(fieldinfo['projectid'],
                                    fieldinfo['field'],
                                    fieldinfo['ccd'],
                                    camera=fieldinfo['camera'])
    areffistar = framearef['framepath'].replace('.fits','.fistar')

    # calculate the shift for the astrometric reference fistar only, since it
    # should be the identity
    areffname = os.path.basename(areffistar)

    parsearef = search('{}-astromref-{}_cal_img_bkgdsub.fistar', areffistar)
    arefid = parsearef[1]
    areforiginalfistar = os.path.join(fitsdir,
                                      arefid+'_cal_img_bkgdsub.fistar')

    outdir = fitsdir
    shifted_fistar, shifted_itrans = ism.astromref_shift_worker(
        (areforiginalfistar, areffistar, outdir)
    )

    # then make symlinks for all the other itrans files
    print('making SYMLINKS to FAKE (identity) "itrans" files...')

    outpaths = [f.replace('.fits','.itrans') for f in fitstoshift
                if arefid not in f]
    src = shifted_itrans
    for dst in outpaths:
        os.symlink(src, dst)
        print('{} --> {}'.format(src, dst))


def initial_wcs_worked_well_enough(outdir, fitsglob):
    # check whether initial anet converged on over half of frames.

    N_fitsfiles = len(glob(outdir+fitsglob))
    N_wcsfiles = len(glob(outdir+fitsglob.replace('.fits','.wcs')))

    if N_wcsfiles < N_fitsfiles/2:
        return False
    else:
        return True


def is_presubtraction_complete(outdir, fitsglob, lcdir, RED_dir,
                               percentage_required=95, extractsources=False):
    """
    require at least e.g., 95% of the initial astrometry, photometry, etc to
    exist to return True. in that case, or if any stats_files products are
    found, move on to image subtraction.  else, returns False.
    """

    N_fitsfiles = len(glob(os.path.join(RED_dir,fitsglob)))
    N_fistarfiles = len(glob(outdir+fitsglob.replace('.fits','.fistar')))
    N_wcsfiles = len(glob(outdir+fitsglob.replace('.fits','.wcs')))
    N_projcatalog = len(glob(outdir+fitsglob.replace('.fits','.projcatalog')))
    N_sourcelistfiles = len(glob(outdir+fitsglob.replace('.fits','.sourcelist')))
    N_fiphotfiles = len(glob(outdir+fitsglob.replace('.fits','.fiphot')))

    if extractsources:
        N_files = [N_fitsfiles, N_fistarfiles, N_wcsfiles, N_fiphotfiles,
                   N_sourcelistfiles, N_projcatalog]
    else:
        N_files = [N_fitsfiles, N_fistarfiles, N_wcsfiles, N_fiphotfiles,
                   N_projcatalog]

    statsdir = lcdir+'stats_files'
    N_statsfiles_products = len(glob(statsdir+"*"))
    if N_statsfiles_products > 1:
        print('found stats_files product. skipping to detrending.')
        return True

    elif (np.any( np.abs(np.diff(N_files)/N_fitsfiles)>(1-percentage_required/100) )
        or
        np.any( np.array(N_files)==0 )
    ):
        print('did not find {:d}% completed source extraction, astrometry, '.
              format(percentage_required)+
              'and initial photometry')
        return False

    else:
        print('found {:d}% completed source extraction, astrometry, '.
              format(percentage_required)+
              'and initial photometry. skipping presubtraction steps.')
        return True


def is_imagesubtraction_complete(fitsdir, fitsglob, lcdir):

    N_subfitslist = len(glob(
        fitsdir+'[r|n]sub-????????-'+fitsglob.replace('.fits','-xtrns.fits')))
    N_iphotlist = len(glob(
        fitsdir+'[r|n]sub-????????-'+fitsglob.replace('.fits','.iphot')))
    N_itranslist = len(glob(
        fitsdir+fitsglob.replace('.fits','.itrans')))
    N_kernellist = len(glob(
        fitsdir+'[r|n]sub-????????-'+
        fitsglob.replace('.fits','-xtrns.fits-kernel')))
    N_subconvjpglist = len(glob(
        fitsdir+'JPEG-SUBTRACTEDCONV-[r|n]sub-*-tess*.jpg'))
    N_lcs = len(glob(
        lcdir+'*.grcollectilc'))

    N_files = [N_subfitslist, N_iphotlist, N_itranslist, N_kernellist,
               N_subconvjpglist]

    full_N_files = [N_subfitslist, N_iphotlist, N_itranslist, N_kernellist,
                    N_subconvjpglist, N_lcs]

    statsdir = lcdir+'stats_files/'
    N_statsfiles_products = len(glob(statsdir+"*"))

    if N_statsfiles_products >= 1:
        print('found stats_files product. skipping to detrending.')
        return True

    elif np.any( np.diff(N_files) ) or np.any( np.array(full_N_files)==0 ):
        msg = (
            'did not find completed image-subtracted photometry products '
            'including photometry, images, and lightcurves. found N_files = '
            '{}, full_N_files = {}'.format(repr(N_files), repr(full_N_files))
        )
        print(msg)
        return False

    else:
        return True


def examine_astrometric_shifts(fitsdir, astromref, statsdir,
                               wcsglob='*.wcs',
                               fitsglob='-xtrns.fits'):
    """
    compute, then plot, astrometric shifts. look at both the astrometry.net
    results, as well as the POC's astrometry.
    """

    pklpath = os.path.join(statsdir, 'examine_astrometric_shifts.pickle')

    if not os.path.exists(pklpath):
        # get wcs files from this run
        wcsfiles = glob(os.path.join(fitsdir,wcsglob))
        assert len(wcsfiles) > 1
        xtrnsfits = glob(os.path.join(fitsdir,'*-xtrns.fits'))

        if len(xtrnsfits) != len(wcsfiles):
            if len(wcsfiles) - len(xtrnsfits) < 10:
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
                    '\nN_xtrns: {:d}'.format(len(xtrnsfits))
                )

        # using fits headers, get corresponding timestamps from this run
        hdrs = [iu.read_fits_header(f, ext=0) for f in xtrnsfits]
        tstarts = nparr([h['TSTART'] for h in hdrs])
        tstops = nparr([h['TSTOP'] for h in hdrs])
        tmids = tstarts + (tstops - tstarts)/2.

        # get astromref wcs file
        parsearef = search('{}-astromref-{}_cal_img_bkgdsub.fits',
                           os.path.basename(astromref))
        arefid = parsearef[1]
        arefwcsfile = os.path.join(fitsdir, arefid+'_cal_img_bkgdsub.wcs')
        print('matching against astromref {}'.format(arefwcsfile))

        if not os.path.exists(arefwcsfile):
            raise AssertionError('need astromref wcs to exist')

        # my WCS := WCS computed using astrometry.net
        # compute distances between ra="CRVAL1" dec="CRVAL2" and the astrometric
        # references.
        my_arefwcs = wcs.WCS(fits.open(arefwcsfile)[0].header)
        my_aref_crval = my_arefwcs.wcs.crval
        my_aref_crpix = my_arefwcs.wcs.crpix
        my_aref_crval_c = SkyCoord(my_aref_crval.reshape((1,2))*units.deg,
                                   frame='icrs')

        my_crvals = []
        for ix, f in enumerate(wcsfiles):
            print('{:d}/{:d} for wcs hdr read'.format(ix,len(wcsfiles)))
            this_wcs = wcs.WCS(iu.read_fits_header(f, ext=0))

            # given wcs, compute the sky coordinate at the astromref CRPIX.
            # this is needed b/c astrometry.net lets the CRPIX float.
            # for each frame, we're computing the WCS solution, and asking
            # "what position on-sky does the xy-coordinate 'my_aref_crpix'
            # correspond to?" for my_astromref, that's the CRVAL; for the
            # others, it's the value computed from the full WCS
            # solution as follows:
            my_crvals.append(this_wcs.wcs_pix2world(
                my_aref_crpix.reshape((1,2)), 1 )
            )

        my_crvals = nparr(my_crvals)
        my_crval_c = SkyCoord(my_crvals.squeeze()*units.deg, frame='icrs')

        # SPOC/POC WCS
        poc_arefwcs = wcs.WCS(fits.open(
            arefwcsfile.replace('.wcs','-xtrns.fits'))[0].header)
        poc_aref_crval = poc_arefwcs.wcs.crval
        poc_aref_crval_c = SkyCoord(poc_aref_crval.reshape((1,2))*units.deg,
                                    frame='icrs')

        poc_wcss = []
        from astropy.utils.exceptions import AstropyWarning
        for ix, f in enumerate(xtrnsfits):
            print('{:d}/{:d} for poc wcs hdr read'.format(ix,len(xtrnsfits)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                poc_wcss.append(wcs.WCS(iu.read_fits_header(f, ext=0)))
        poc_crvals = nparr([ w.wcs.crval for w in poc_wcss ])
        poc_crval_c = SkyCoord(poc_crvals*units.deg, frame='icrs')

        d = {}
        d['my_aref_crpix'] = my_aref_crpix
        d['my_aref_crval_c'] = my_aref_crval_c
        d['my_crval_c'] = my_crval_c
        d['poc_aref_crval_c'] = poc_aref_crval_c
        d['poc_crval_c'] = poc_crval_c
        d['tmids'] = tmids

        with open(pklpath, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
        print('made {}'.format(pklpath))

    else:
        d = pickle.load(open(pklpath, 'rb'))

    # now plot them
    my_aref_crval_c = d['my_aref_crval_c']
    my_crval_c = d['my_crval_c']

    poc_aref_crval_c = d['poc_aref_crval_c']
    poc_crval_c = d['poc_crval_c']

    my_diffs = my_aref_crval_c.separation(my_crval_c).to(units.arcsec)
    poc_diffs = poc_aref_crval_c.separation(poc_crval_c).to(units.arcsec)

    # first, scatter plot difference vs time
    scatterpath = os.path.join(statsdir, 'scatter_astrometric_shifts.png')
    histpath = os.path.join(statsdir, 'histogram_astrometric_shifts.png')

    fig,ax = plt.subplots(figsize=(4,3))
    ax.scatter(d['tmids'], my_diffs.value, label='TREX', rasterized=True)
    ax.scatter(d['tmids'], poc_diffs.value, label='POC', rasterized=True)
    ax.legend(loc='best')
    ax.set_xlabel('BTJD = BJD - 2457000', fontsize='x-small')
    ax.set_ylabel('distance from astromref CRVAL [arcsec]', fontsize='x-small')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout()
    fig.savefig(scatterpath, bbox_inches='tight', dpi=300)
    print('saved {}'.format(scatterpath))
    plt.close('all')

    # then, histogram plot of differences
    fig,ax = plt.subplots(figsize=(4,3))
    ax.hist(my_diffs.value, label='TREX', bins=20)
    ax.hist(poc_diffs.value, label='POC', bins=20)
    ax.legend(loc='best')
    ax.set_ylabel('count', fontsize='x-small')
    ax.set_xlabel('distance from astromref CRVAL [arcsec]', fontsize='x-small')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.tight_layout()
    fig.savefig(histpath, bbox_inches='tight', dpi=300)
    print('saved {}'.format(histpath))
    plt.close('all')


def plot_random_lightcurves_and_ACFs(statsdir, pickledir, n_desired=10,
                                     skipepd=False):
    """
    make sequential (6 row?) RAW, (optionally) EPD, TFA plots with ACFs., for
    n_desired total files.
    """

    sel_acffiles = _get_random_acf_pkls(pickledir, n_desired=n_desired)

    for acffile in sel_acffiles:

        d = pickle.load(open(acffile, 'rb'))

        for ap in range(1,len(d)+1):

            key = 'AP{}'.format(ap)

            savdir = os.path.join(statsdir,'random_lightcurves_and_ACFs')
            if not os.path.exists(savdir):
                os.mkdir(savdir)
            savpath = os.path.join(
                savdir,
                os.path.basename(acffile).rstrip('.pickle')+
                '_AP{:d}'.format(ap)+'.png'
            )

            if os.path.exists(savpath):
                continue

            if skipepd:
                # hacky placeholder to prevent too much code rewrite in
                # TFA-direct-from-IRM case
                foo = np.zeros_like(d[key]['lag_time_irm'])

                lcs.plot_lightcurve_and_ACF(
                    d[key]['lag_time_irm'], d[key]['acf_irm'],
                    foo, foo,
                    d[key]['lag_time_tfa'], d[key]['acf_tfa'],
                    d[key]['itimes_irm'], d[key]['ifluxs_irm'],
                    foo, foo,
                    d[key]['itimes_tfa'], d[key]['ifluxs_tfa'],
                    ap, savpath=savpath, skipepd=skipepd
                )

            else:
                lcs.plot_lightcurve_and_ACF(
                    d[key]['lag_time_irm'], d[key]['acf_irm'],
                    d[key]['lag_time_epd'], d[key]['acf_epd'],
                    d[key]['lag_time_tfa'], d[key]['acf_tfa'],
                    d[key]['itimes_irm'], d[key]['ifluxs_irm'],
                    d[key]['itimes_epd'], d[key]['ifluxs_epd'],
                    d[key]['itimes_tfa'], d[key]['ifluxs_tfa'],
                    ap, savpath=savpath, skipepd=skipepd
                )

def plot_random_lightcurve_subsample(lcdirectory, n_desired_lcs=20,
                                     timename='TMID_BJD', isfitslc=True,
                                     skipepd=False,
                                     ext='.fits'):
    """
    make sequential RAW, EPD, TFA plots for `n_desired_lcs`
    """

    sel_tfafiles = _get_random_tfa_lcs(lcdirectory,
                                       n_desired_lcs=n_desired_lcs)

    for tfafile in sel_tfafiles:

        if not isfitslc:
            lcdata = read_tfa_lc(tfafile)
        else:
            hdulist = fits.open(tfafile)
            lcdata = hdulist[1].data

        for ap in range(1,4):
            rawap = 'IRM{:d}'.format(ap)
            epdap = 'EP{:d}'.format(ap)
            tfaap = 'TFA{:d}'.format(ap)

            savdir = os.path.join(os.path.dirname(tfafile),
                                  'stats_files',
                                  'random_lightcurve_subsample')
            if not os.path.exists(savdir):
                os.mkdir(savdir)
            savpath = os.path.join(
                savdir,
                os.path.basename(tfafile).rstrip(ext)+
                '_AP{:d}'.format(ap)+'.png'
            )

            if os.path.exists(savpath):
                continue

            if skipepd:
                lcs.plot_raw_epd_tfa(lcdata[timename],
                                     lcdata[rawap],
                                     np.zeros_like(lcdata[rawap]),
                                     lcdata[tfaap],
                                     ap,
                                     savpath=savpath,
                                     skipepd=skipepd)
            else:
                lcs.plot_raw_epd_tfa(lcdata[timename],
                                     lcdata[rawap],
                                     lcdata[epdap],
                                     lcdata[tfaap],
                                     ap,
                                     savpath=savpath,
                                     skipepd=skipepd)

def _plot_normalized_subtractedimg_histogram(
    subimg_normalized, subimgfile, zerooutnans=True):
    """subimg_normalized: subtracted, normalized image."""

    savdir = os.path.dirname(subimgfile)
    savname = (
        os.path.basename(subimgfile).replace(
            '.fits','-normalized_histogram.png')
    )
    savpath = os.path.join(savdir, savname)

    if zerooutnans:
        subimg_normalized[np.isnan(subimg_normalized)] = 0
        print('ERR! you maybe should not have nans in your image...')
    else:
        raise NotImplementedError

    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3))

    bins = np.arange(-100,100,0.1)
    bins, edges = np.histogram(subimg_normalized, bins=bins)
    left, right = edges[:-1], edges[1:]

    # data
    X = np.array([left,right]).T.flatten()
    Y = np.array([bins,bins]).T.flatten()

    ax.plot(X, Y, label='data')

    # theory: a gaussian
    # 1/sqrt(2pi*sigma^2) * exp( -(x-mu)^2 / (2 sigma^2)  )
    mu, sigma = 0, 1
    x_theory = np.linspace(X.min(), X.max(), num=int(5e3))
    y_theory = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(
        - (x_theory - mu)**2 / (2*sigma**2)
    )

    n_pixels = len(subimg_normalized.flatten())
    ax.plot(x_theory, y_theory*n_pixels, label='theory')

    ax.set_xlim([-4*sigma, 4*sigma])

    ax.legend(loc='best')
    ax.set_xlabel('normalized difference pixel value')
    ax.set_ylabel('number of pixels')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))

    f.tight_layout()
    f.savefig(savpath, dpi=200, bbox_inches='tight')
    print('%sZ: wrote %s' % (datetime.utcnow().isoformat(), savpath))


def is_image_noise_gaussian(
    fitsdir, projectid, field, camera, ccd,
    photrefdir='/nfs/phtess2/ar0/TESS/FFI/BASE/reference-frames/',
    n_histograms_to_make=40):
    """
    The noise in the differenced image should be gaussian.  Oelkers & Stassun
    (2018) suggest the following approach to check whether it is.

    For each subtracted frame, normalize it by the combination of the noise
    from the science frame  and the photref frame:

        expected noise = sqrt( science frame  + photref frame ).

    The normalized pixels should be a gaussian centered at zero, with a std
    devn of 1.
    """

    subglob = '[r|n]sub-*-tess*-xtrns.fits'
    subfiles = np.sort(glob(fitsdir+subglob))

    imgglob = 'tess*-xtrns.fits'
    sciimgfiles = np.sort(glob(fitsdir+imgglob))

    # e.g., proj43-sector-10-cam1-ccd2-combinedphotref-onenight.fits
    photrefglob = ('proj{:d}-{:s}-cam{:s}-ccd{:s}-*.fits'.format(
                   projectid,field,str(camera),str(ccd)))
    photreffile = np.sort(glob(photrefdir+photrefglob))

    if len(photreffile) != 1:
        raise AssertionError('expected a single photometric reference')

    photreffile = photreffile[0]
    photrefimg, _ = iu.read_fits(photreffile, ext=0)

    for subimgfile, sciimgfile in zip(subfiles[:n_histograms_to_make],
                                       sciimgfiles[:n_histograms_to_make]):

        savdir = os.path.dirname(subimgfile)
        savname = (
            os.path.basename(subimgfile).replace(
                '.fits','-normalized_histogram.png')
        )
        savpath = os.path.join(savdir, savname)
        if os.path.exists(savpath):
            continue

        subimg, subhdr = iu.read_fits(subimgfile, ext=0)
        sciimg, scihdr = iu.read_fits(sciimgfile, ext=0)

        expected_noise = np.sqrt(photrefimg + sciimg)

        subimg_normalized = subimg / expected_noise

        _plot_normalized_subtractedimg_histogram(
            subimg_normalized, subimgfile)


def record_reduction_parameters(fitsdir, fitsglob, projectid, field, camnum,
                                ccdnum, outdir, lcdirectory, nworkers,
                                aperturelist, kernelspec,
                                convert_to_fitsh_compatible, anetfluxthreshold,
                                anettweak, initccdextent, anetradius,
                                zeropoint, epdsmooth, epdsigclip,
                                photdisjointradius, tuneparameters, is_ete6,
                                cluster_faint_Rp_mag, field_faint_Rp_mag,
                                fiphotfluxthreshold, photreffluxthreshold,
                                extractsources, binlightcurves, get_masks,
                                tfa_template_sigclip, tfa_epdlc_sigclip,
                                translateimages, reversesubtract, skipepd,
                                useimagenotfistar, fixedtfatemplate,
                                flagvalues, do_cdips_merge):
    """
    each "reduction version" is identified by a project ID. the parameters
    corresponding to each project ID are written in a pickle file, so that we
    can easily translate from project IDs to the parameters that are associated
    with them.
    """

    outd = {
        "fitsdir":fitsdir,
        "fitsglob":fitsglob,
        "projectid":projectid,
        "field":field,
        "camnum":camnum,
        "ccdnum":ccdnum,
        "outdir":outdir,
        "lcdirectory":lcdirectory,
        "nworkers":nworkers,
        "aperturelist":aperturelist,
        "kernelspec":kernelspec,
        "convert_to_fitsh_compatible":convert_to_fitsh_compatible,
        "anetfluxthreshold":anetfluxthreshold,
        "anettweak":anettweak,
        "initccdextent":initccdextent,
        "anetradius":anetradius,
        "zeropoint":zeropoint,
        "epdsmooth":epdsmooth,
        "epdsigclip":epdsigclip,
        "photdisjointradius":photdisjointradius,
        "tuneparameters":tuneparameters,
        "is_ete6":is_ete6,
        "cluster_faint_Rp_mag":cluster_faint_Rp_mag,
        "field_faint_Rp_mag":field_faint_Rp_mag,
        "fiphotfluxthreshold":fiphotfluxthreshold,
        "photreffluxthreshold":photreffluxthreshold,
        "extractsources":extractsources,
        "binlightcurves":binlightcurves,
        "get_masks":get_masks,
        "tfa_template_sigclip":tfa_template_sigclip,
        "tfa_epdlc_sigclip":tfa_epdlc_sigclip,
        "translateimages":translateimages,
        "reversesubtract":reversesubtract,
        "skipepd":skipepd,
        "useimagenotfistar":useimagenotfistar,
        "fixedtfatemplate":fixedtfatemplate,
        "flagvalues":flagvalues,
        "do_cdips_merge":do_cdips_merge
    }

    outpicklename = "projid_{:s}.pickle".format(repr(projectid))
    outpickledir = '/nfs/phtess2/ar0/TESS/FFI/PROJ/REDUCTION_PARAM_PICKLES/'
    outpath = os.path.join(outpickledir, outpicklename)

    with open(outpath, 'wb') as f:
        pickle.dump(outd, f, pickle.HIGHEST_PROTOCOL)
    print('wrote {}'.format(outpath))



##################################################
# WRAPPER FUNCTIONS FOR MAJOR STEPS IN REDUCTION #
##################################################

def get_files_needed_before_image_subtraction(
        sectornum, camera, ccd,
        fitsdir, fitsglob, outdir, initccdextent, ccdgain, zeropoint, exptime,
        ra_nom, dec_nom,
        catra, catdec, catboxsize,
        catalog, catalog_file, reformed_cat_file, field_faint_Rp_mag,
        fnamestr='*-1-1-0016_cal_img_bkgdsub.fits', anetfluxthreshold=20000,
        fistarglob='*.fistar',
        width=13, anettweak=6, anetradius=30, xpix=2048, ypix=2048, cols=(2,3),
        brightrmag=5.0, faintrmag=13.0,
        fiphotfluxthreshold=1000,
        aperturelist='1.45:7.0:6.0,1.95:7.0:6.0,2.45:7.0:6.0',
        nworkers=20,
        useastrometrydotnet=True,
        useimagenotfistar=True,
        extractsources=True,
        do_cdips_merge=True,
        astrometrydownsample=2,
        defaultwcsisspoc=True,
        raise_wcs_error=False,
        isforce=False,
    ):
    """
    get .fistar, .fiphot, and .wcs files needed before image subtraction

    1. run parallel_extract_sources on all frames with threshold ~ 10000 to get
    bright stars that are optionally used for astrometry. (SPOC WCS is
    default for TESS).
    2. run make_fov_catalog to get a FOV source catalog for the field.
    3. run reform_gaia_fov_catalog to cut this down to the columns needed for
    magfit only.
    4. run parallel_anet to get precise WCS headers for all frames.
    5. run parallel_fitsdir_photometry for photometry on all frames (via
    fistar)
    """

    ap.parallel_extract_sources(fitsdir, outdir, ccdextent=initccdextent,
                                ccdgain=ccdgain,
                                fluxthreshold=anetfluxthreshold,
                                zeropoint=zeropoint, exptime=exptime,
                                tailstr='.fits',
                                fnamestr=fnamestr, overwrite=isforce)

    # This function gets all possible sources in the field of view of the
    # frame, given its central pointing coordinates and plate-scale from 2MASS.
    _ = ap.make_fov_catalog(ra=catra, dec=catdec, size=catboxsize,
                            brightrmag=brightrmag, faintrmag=faintrmag,
                            fits=None, outfile=None, outdir=outdir,
                            catalog=catalog, catalogpath=None,
                            columns=None, observatory='tess',
                            gaiaidrequest='GAIA')

    # This converts the full output catalog from 2massread, etc. to the format
    # required for magfit. Also useful for general reforming of the columns.
    if catalog=='GAIADR2':
        ap.reform_gaia_fov_catalog(catalog_file, reformed_cat_file)
    else:
        # you could use ap.reform_fov_catalog, but you should be using Gaia.
        raise ValueError

    # Reduce number of LCs. Merge against CDIPS stars.
    if do_cdips_merge:
        tu.merge_object_catalog_vs_cdips(reformed_cat_file, reformed_cat_file,
                                         G_Rp_cut=field_faint_Rp_mag)


    # Work with WCS from SPOC headers by default. Require median astrometric
    # residual below 0.2 pixel, 90th percentile below 0.4 pixel. Std devn below
    # 1.0 pixel.
    if useastrometrydotnet:

        if defaultwcsisspoc:

            fitslist = glob(os.path.join(fitsdir,fnamestr))
            for f in fitslist:
                wcsqa.write_wcs_from_spoc(f, observatory='tess',
                                          assert_ctype_intact=True)

            # in writing the WCSs, some files may have been moved to badframes.
            fitslist = glob(os.path.join(fitsdir,fnamestr))
            wcslist = [f.replace('.fits','.wcs') for f in fitslist]
            fistarlist = [f.replace('.fits','.fistar') for f in fitslist]

            skip = 100 # check every 100th object

            framecoord = SkyCoord(
                ra=ra_nom, dec=dec_nom, unit=(units.degree, units.degree),
                frame='icrs'
            )

            frame_galacticlatitude = framecoord.galactic.b.value

            if abs(frame_galacticlatitude) < 10:
                # allow worse wcs outliers if frame center within 10 deg of
                # galactic plane
                qualitycondition = {'median_px':0.2,
                                    '90th_px':0.7,
                                    'std_px':2.3 }
            else:
                qualitycondition = {'median_px':0.2,
                                    '90th_px':0.5,
                                    'std_px': 1.7}

            for w,f,s in zip(wcslist[::skip],fitslist[::skip],fistarlist[::skip]):

                if wcsqa.does_wcs_pass_quality_check(
                    w, f, reformed_cat_file, isspocwcs=True, N_bright=1000,
                    N_faint=9000, fistarpath=s, matchedoutpath=None,
                    qualitycondition=qualitycondition
                ):
                    print('{} passed WCS quality check'.format(w))

                else:

                    # weaker check for heavily background contaminated frames.
                    if wcsqa.does_wcs_pass_bkgd_corrected_quality_check(
                        w, f, reformed_cat_file, isspocwcs=True, N_bright=1000,
                        N_faint=9000, fistarpath=s, matchedoutpath=None,
                        qualitycondition={'median_px':0.2, '90th_px':0.8,
                                          'std_px': 3}
                    ):

                        pass

                    else:
                        errmsg = (
                            'ERR! WCS fails quality check '+
                            'wcs, fits, fistar are {}, {}, {}'.format(w,f,s)
                        )
                        print(errmsg)
                        if raise_wcs_error:
                            # NOTE : might need to derive the WCS yourself, using a
                            # parallel_astrometrydotnet call as below...
                            raise AssertionError(errmsg)

        else:

            raise DeprecationWarning('TESS reduction uses SPOC WCS as default')
            raise NotImplementedError

            if not useimagenotfistar:
                ap.fistardir_to_xy(outdir, fistarglob=fistarglob)

            ap.parallel_astrometrydotnet(
                fitsdir, outdir, ra_nom, dec_nom,
                fistarfitsxyglob='tess*_bkgdsub.fistar-fits-xy',
                tweakorder=anettweak, radius=anetradius, xpix=xpix, ypix=ypix,
                nworkers=nworkers, scalelow=10, scalehigh=30,
                scaleunits='arcsecperpix', nobjs=200, xcolname='ximage',
                ycolname='yimage', useimagenotfistar=useimagenotfistar,
                downsample=astrometrydownsample,
                pixelerror=pixelerror,
                uniformize=uniformize
            )

    else:
        raise AssertionError('we don\'t use anet')
        ap.parallel_anet(outdir, outdir, ra_nom, dec_nom,
                         fistarglob=fistarglob,
                         infofromframe=False, width=width, tweak=anettweak,
                         radius=anetradius,
                         xpix=xpix, ypix=ypix,
                         cols=cols # columns with x,y in fistar file.
        )


    if extractsources==True:
        fiphot_xycols = '7,8'
    else:
        fiphot_xycols = '13,14'
    ap.parallel_fitsdir_photometry(fitsdir, outdir, reformed_cat_file,
                                   fluxthreshold=fiphotfluxthreshold,
                                   ccdextent={'x':[0.,2048.],'y':[0.,2048.]},
                                   pixborders=0.0,
                                   aperturelist=aperturelist,
                                   removesourcetemp=True,
                                   removesourcelist=False, binaryoutput=False,
                                   nworkers=nworkers, maxtasksperworker=1000,
                                   saveresults=True, rejectbadframes=True,
                                   minsrcbgv=200.0, maxmadbgv=150.0,
                                   maxframebgv=2000.0, minnstars=500,
                                   fitsglob=fitsglob, ccdgain=ccdgain,
                                   ccdexptime=exptime, zeropoint=zeropoint,
                                   extractsources=extractsources,
                                   fovcat_xycols=(12,13),
                                   projcat_xycols=(24,25),
                                   fiphot_xycols=fiphot_xycols,
                                   observatory='tess',
                                   overwrite=isforce
                                  )


def run_imagesubtraction(fitsdir, fitsglob, fieldinfo, photparams, fits_list,
                         photreftype, dbtype, reformed_cat_file, xtrnsglob,
                         iphotpattern, lcdirectory, outdir, projectid,
                         ra_nom, dec_nom,
                         kernelspec='b/4;i/4;d=4/4', refdir=sv.REFBASEDIR,
                         nworkers=1,
                         aperturelist='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
                         photdisjointradius=2, colorscheme='bwr',
                         photreffluxthreshold=30000, extractsources=True,
                         translateimages=True, reversesubtract=False,
                         useimagenotfistar=True,
                         astrometrydownsample=2, pixelerror=0.3, uniformize=10,
                         isforce=False):

    ccdgain = photparams['ccdgain']
    exptime = photparams['ccdexptime']
    zeropoint = photparams['zeropoint']

    # Step ISP0.
    _ = ais.parallel_frames_to_database(fitsdir, 'calibratedframes',
                                        custom_projid=projectid,
                                        observatory='tess', fitsglob=fitsglob,
                                        overwrite=isforce,
                                        badframetag='badframes',
                                        nonwcsframes_are_ok=False,
                                        nworkers=nworkers, maxworkertasks=1000)

    # Step ISP1.
    arefinfo = ais.dbgen_get_astromref(fieldinfo, makeactive=True,
                                       observatory='tess', overwrite=isforce,
                                       refdir=sv.REFBASEDIR, database=None)
    if arefinfo == None:
        raise AssertionError('you need an astrometric reference!!')

    # Step ISP2. Takes ~1 sec per 20-30 frames.
    _ = ism.get_smoothed_xysdk_coeffs(fitsdir, fistarglob='*.fistar',
                                      nworkers=nworkers, maxworkertasks=1000,
                                      overwrite=isforce)

    # Step ISP3. ~600 in 5 minutes --> 2 frames per second, running over 20 workers.
    # If called with warpcheck=True and warpthreshold=2000, many things will be
    # moved to badframes that are not in fact badframes. The warpthreshold is a bad
    # empirical thing that should be avoided.
    if translateimages:
        _ = ais.framelist_make_xtrnsfits(fits_list, fitsdir, fitsglob,
                                         outdir=outdir,
                                         refinfo='foobar', warpcheck=False,
                                         warpthreshold=15000.0, warpmargins=100,
                                         nworkers=nworkers, observatory='tess',
                                         maxworkertasks=1000, fieldinfo=fieldinfo,
                                         overwrite=isforce)
    else:
        make_fake_xtrnsfits(fitsdir, fitsglob, fieldinfo)

    # # Optional Step ISP3.5: parallelized move of astrometry ref shifted frames to database
    # out = ais.parallel_frames_to_database(fitsdir, 'arefshifted_frames',
    #                                       fitsglob='1-???????_?-xtrns.fits',
    #                                       network='HP', overwrite=False,
    #                                       badframetag='badframes',
    #                                       nonwcsframes_are_ok=False, nworkers=nworkers,
    #                                       maxworkertasks=1000)
    # 

    # Step ISP4: the next thing to do is to select a bunch of frames that can serve
    # as photometric reference frames (photrefs).

    xtrnsfiles = glob(fitsdir+xtrnsglob)
    photrefinfo = ais.generate_photref_candidates_from_xtrns(
        xtrnsfiles, minframes=50, observatory='tess',
        maxbackgroundstdevpctile=100., maxbackgroundmedianpctile=70.,
        minngoodobjectpctile=70., forcecollectinfo=isforce, nworkers=nworkers,
        maxworkertasks=1000)

    # Optional Step ISP5: amend the list, if needed.
    # photrefinfo = ais.amend_candidate_photrefs(photrefinfo)

    # Step ISP6: make a photometric reference frame
    _ = ais.generate_combined_photref(
        photrefinfo, photreftype, dbtype,
        ra_nom, dec_nom,
        photref_reformedfovcat=reformed_cat_file, makeactive=True, field=None,
        ccd=None, projectid=None, combinemethod='median',
        kernelspec=kernelspec, ccdgain=ccdgain, zeropoint=zeropoint,
        ccdexptime=exptime, extractsources=extractsources,
        apertures=aperturelist, framewidth=None, searchradius=8.0,
        nworkers=nworkers, maxworkertasks=1000, observatory='tess',
        fieldinfo=fieldinfo, photreffluxthreshold=photreffluxthreshold,
        overwrite=isforce,
        useimagenotfistar=useimagenotfistar,
        astrometrydownsample=astrometrydownsample,
        pixelerror=pixelerror,
        uniformize=uniformize,
        reformed_cat_file=reformed_cat_file)

    # overplot forced aperture locations on subset of photref frame. this is a
    # sanity check that they are in correct locations.
    from tessutils import plot_apertures_on_frame
    photrefframe = _['combinedphotref']['frame']
    photrefprojcat = photrefframe.replace('.fits','.projcatalog')
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[1024-150,1024+150],ylim=[1024-150,1024+150])
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[1024-70,1024+70],ylim=[1024-70,1024+70])
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[0,300],ylim=[0,300])
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[300+80,300+160],ylim=[300+80,300+160])
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[1300+80,1300+160],ylim=[1300+80,1300+160])
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[2048-300,2048],ylim=[0,300])
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[0,300],ylim=[2048-300,2048])
    plot_apertures_on_frame(photrefframe, photrefprojcat,
                            xlim=[2048-300,2048],ylim=[2048-300,2048])

    # Step ISP7: convolve and subtract all FITS files in the xtrnsfits list from the
    # photometric reference.  With 30 workers, at best process ~few frames per
    # second.

    if isforce or len(glob(os.path.join(fitsdir,'*.iphot')))<10:
        _ = ais.parallel_xtrnsfits_convsub(
            xtrnsfiles, photreftype, fitsdir=fitsdir, fitsglob=fitsglob,
            outdir=outdir, observatory='tess', fieldinfo=fieldinfo,
            overwrite=isforce,
            reversesubtract=reversesubtract, kernelspec=kernelspec,
            nworkers=nworkers, maxworkertasks=1000, colorscheme=colorscheme)
    else:
        print('found .iphot files. skipping convolution+subtraction.')

    # Step ISP8: do photometry on your subtracted frames to produce .iphot files.
    # With 30 workers, at best process ~few frames per second.

    if isforce or len(glob(os.path.join(fitsdir,'*.iphot')))<10:
        subfitslist = glob(os.path.join(fitsdir,'[r|n]sub-????????-'+
                                        fitsglob.replace('.fits','-xtrns.fits')))

        # FIXME FIXME ensure that the projected positions used for forced
        # aperture photometry are correctedly shifted (and correctly span
        # 0:2048, instead of like 0:2048-SCIROWS

        out = ais.parallel_convsubfits_staticphot(
            subfitslist, fitsdir=fitsdir, fitsglob=fitsglob,
            photreftype=photreftype, kernelspec=kernelspec,
            lcapertures=aperturelist, photdisjointradius=photdisjointradius,
            outdir=outdir, fieldinfo=fieldinfo, overwrite=isforce,
            observatory='tess',
            nworkers=nworkers, maxworkertasks=1000, photparams=photparams)
        if out==42:
            raise AssertionError('fatal error in convsubfits_staticphot')
    else:
        print('found .iphot files. skipping their production.')

    # Step ISP9 + 10 : dump lightcurves.
    if isforce or len(glob(os.path.join(lcdirectory,'*.grcollectilc'))) < 10:
        ism.dump_lightcurves_with_grcollect(
            iphotpattern, lcdirectory, '4g', lcextension='grcollectilc',
            objectidcol=3, observatory='tess')
    else:
        print('found grcollectilc files. skipping lc dump.')

    # # Alternative Step 9 + 10: add the .iphot files to postgres database. Collect
    # # LCs from postgres, and then make difference image lightcurve (*.ilc) files in
    # # lcdirectory. AKA "light curve dump", or "the transposition problem".
    # # Surprisingly computationally expensive.  An alternative approach is
    # # fitsh's `grcollect`, which skips the database architecture entirely.
    # 
    # print('beginning insert_phots_into_database')
    # ais.insert_phots_into_database(sv.REDPATH, frameglob='[r|n]sub-*-xtrns.fits',
    #                                photdir=None, photglob='[r|n]sub-*-%s.iphot',
    #                                maxframes=None, overwrite=False, database=None)
    # 
    # hatidlist = ais.get_hatidlist_from_cmrawphot(projectid, field, ccd, photreftype)
    # 
    # print('beginning lightcurve dump')
    # ais.parallel_dbphot_lightcurves_hatidlist(hatidlist, lcdirectory)


def _get_ok_lightcurve_files(lcfiles):
    # given a list of lightcurve files with some that are corrupted, or for
    # some reason have zero length, or other issues. get the ones that are ok.
    # delete the not OK ones.

    # clean LCs.
    iszerosize = np.array([os.stat(f).st_size == 0 for f in lcfiles])

    if np.any(iszerosize):

        for zerosizelc in np.array(lcfiles)[iszerosize]:
            os.remove(zerosizelc)
            print('{} WRN! removed {} because zero size'.format(
                datetime.utcnow().isoformat(), zerosizelc))

        lcfiles = np.array(lcfiles)[~iszerosize]

    isopenable = []
    for ix, lcfile in enumerate(lcfiles):
        print('{}/{}'.format(ix, len(lcfiles)))
        try:
            hdulist = fits.open(lcfile)
            _ = hdulist.info()
            hdulist.close()
            isopenable.append(True)
        except:
            isopenable.append(False)
    isopenable = np.array(isopenable)

    if np.any(~isopenable):

        for notopenablelc in np.array(lcfiles)[~isopenable]:
            os.remove(notopenablelc)
            print('{} WRN! removed {} because could not open'.format(
                datetime.utcnow().isoformat(), notopenablelc))

        lcfiles = np.array(lcfiles)[isopenable]

    return lcfiles


def run_detrending(epdstatfile, tfastatfile, vartoolstfastatfile, lcdirectory,
                   epdlcglob, reformed_cat_file, statsdir, field, fitsdir,
                   fitsglob, camera, ccd, projectid,
                   epdsmooth=11, epdsigclip=10, nworkers=10,
                   binlightcurves=False, tfa_template_sigclip=5.0,
                   tfa_epdlc_sigclip=5.0, skipepd=True, fixedtfatemplate=None,
                   isforce=False):
    """
    Step ISP11: do EPD on all the LCs, and collect stats on the results.
    for ISP LCs, use lcmagcols=([27,28,29],[30,],[30,],[30,])

    Step ISP12: do TFA on all the LCs. First, choose TFA template stars using the
    .epdlc stats. Then run TFA, to get .tfalc.TF{1,2,3} files. Turn them into
    single .tfalc files. Then collect statistics.
    """

    catfile = reformed_cat_file.replace('.reformed_catalog', '.catalog')
    tfaboolstatusfile = os.path.join(statsdir,'are_tfa_plots_done.txt')

    if isforce or len(glob(os.path.join(lcdirectory,'*_llc.fits')))<10:

        engdir = '/nfs/phtess2/ar0/TESS/FFI/ENGINEERING/'
        # e.g., s0002-3-3-0121_key_temperature_count.csv
        temperatureglob = ('{}-{}-{}-????_key_temperature_count.csv'.
                    format(field, camera, ccd))
        temperaturedfpath = glob(os.path.join(engdir, temperatureglob))
        if len(temperaturedfpath) != 1:
            raise AssertionError('expected a single temperature csv file')
        temperaturedfpath = temperaturedfpath[0]

        lcu.parallel_convert_grcollect_to_fits_lc(
            lcdirectory, fitsdir, projectid, catfile=catfile,
            ilcglob='*.grcollectilc', nworkers=nworkers, observatory='tess',
            temperaturedfpath=temperaturedfpath, overwrite=isforce
        )

        lcu.parallel_apply_barycenter_time_correction(
            lcdirectory, nworkers=nworkers
        )
    else:
        print('found >10 fits LCs from grcollect. skip grcollect convert.')

    if (isforce or not os.path.exists(epdstatfile)) and not skipepd:

        raise AssertionError('in TESS reduction, we\'re skipping EPD!')

        fitsilcfiles = glob(os.path.join(lcdirectory,'*_llc.fits'))
        outfiles = fitsilcfiles
        hdulist = fits.open(fitsilcfiles[0])
        if hdulist[0].header['DTR_EPD']:
            print('first LC had DTR_EPD flag true -> skipping EPD.')
        else:
            _ = lcu.parallel_run_epd_imagesub_fits(fitsilcfiles, outfiles,
                                                   smooth=epdsmooth,
                                                   sigmaclip=epdsigclip,
                                                   nworkers=nworkers,
                                                   maxworkertasks=1000,
                                                   minndet=100,
                                                   observatory='tess')

            # n.b. this actually works on fits lcs as well.
            ap.parallel_lc_statistics(lcdirectory, epdlcglob,
                                      reformed_cat_file, tfalcrequired=False,
                                      fitslcnottxt=True,
                                      fovcatcols=(0,6), # objectid, magcol to use
                                      fovcatmaglabel='r', outfile=epdstatfile,
                                      nworkers=nworkers,
                                      workerntasks=500, rmcols=None,
                                      epcols=None, tfcols=None,
                                      rfcols=None, correctioncoeffs=None,
                                      sigclip=5.0, fovcathasgaiaids=True,
                                      istessandmaskedges=True)
    elif skipepd:
        print('skipped EPD because got skipepd flag')
    else:
        print('already made EPD LC stats file')

    if (skipepd
        and (isforce or not os.path.exists(tfastatfile))
        and (isforce or not os.path.exists(epdstatfile))
       ):
        # do the hack of getting "EPD" statistics that are actually IRM
        # statistics, duplicated into all the EPD fields.
        ap.parallel_lc_statistics(lcdirectory, epdlcglob, reformed_cat_file,
                                  tfalcrequired=False, epdlcrequired=False,
                                  fitslcnottxt=True, fovcatcols=(0,6),
                                  fovcatmaglabel='r', outfile=epdstatfile,
                                  nworkers=nworkers, workerntasks=500,
                                  rmcols=None, epcols=None, tfcols=None,
                                  rfcols=None, correctioncoeffs=None,
                                  sigclip=5.0, fovcathasgaiaids=True,
                                  istessandmaskedges=True)

    epdmadplot = glob(os.path.join(statsdir, '*median-EP1-vs-mad-*png'))
    if (not epdmadplot and
        (isforce or not os.path.exists(tfaboolstatusfile)) and
        not skipepd
       ):
        ap.plot_stats_file(epdstatfile, statsdir, field, binned=False,
                           logy=True, logx=False, correctmagsafter=None,
                           rangex=(5.9,16), observatory='tess',
                           fovcathasgaiaids=True, yaxisval='RMS')
    else:
        print('skipped making EPD LC plots')

    # choose the TFA template stars
    if isforce or not os.path.exists(os.path.join(
        statsdir,'tfa-stage1-input-aperture-1.txt')
    ):
        if 'TUNE' in statsdir:
            target_nstars, max_nstars = 40, 42
        elif 'FULL' in statsdir:
            # initial "round-1", take many stars. will cut down after.
            target_nstars, max_nstars = 400, 402
        else:
            raise NotImplementedError

        _ = ap.choose_tfa_template(epdstatfile, reformed_cat_file, lcdirectory,
                                   ignoretfamin=False, fovcat_idcol=0,
                                   fovcat_xicol=3, fovcat_etacol=4,
                                   fovcat_magcol=6,
                                   target_nstars=target_nstars,
                                   max_nstars=max_nstars,
                                   brightest_mag=8.5, faintest_mag=13.0,
                                   max_rms=0.01, max_sigma_above_rmscurve=2.0,
                                   outprefix=statsdir, tfastage1=False,
                                   fovcathasgaiaids=True, epdlcext='_llc.fits')


    if isforce or not os.path.exists(vartoolstfastatfile):
        templatefiles = glob(os.path.join(
            statsdir, 'tfa-stage1-input-aperture-?.txt'))
        lcfiles = glob(os.path.join(lcdirectory,'*_llc.fits'))

        lcfiles = _get_ok_lightcurve_files(lcfiles)

        # create ascii files needed for vartools:
        # lc_list_tfa, trendlist_tfa and dates_tfa
        (tfalclist_path, trendlisttfa_paths, datestfa_path) = (
        lcu.make_ascii_files_for_vartools(lcfiles, templatefiles, statsdir,
                                          fitsdir, fitsglob,
                                          fixedtfatemplate=fixedtfatemplate)
        )

        #
        # Note that sometimes, you should do BLS + LS + killharm too.  for now
        # though, we are making lightcurves; these extra steps can come later.
        # If you skipped EPD, then you do TFA from IRM. otherwise, you do TFA
        # from EPD.
        #
        # Ah. There is also a second stage of TFA template star selection. It
        # is to first run TFA on the template stars themselves. Then apply a
        # "eta-normal" (see astrobase) variability cut on the TFA'd template
        # stars. If they are still variable, they can be omitted.
        #
        # The logic here is: sometimes you have lightcurves that have
        # astrophysical-looking, but actually systematic variability. E.g., for
        # TESS, the 1-day scattered light signal can in some cases look like an
        # EB. You can't apply the eta-normal cut on the reduced mag LCs,
        # because this would throw out stars with a systematic signal that you
        # want to remove.
        #
        npixexclude = 20
        tfafromirm = skipepd

        lcu.initial_tfa(trendlisttfa_paths, datestfa_path, lcdirectory,
                        statsdir, nworkers=nworkers,
                        npixexclude=npixexclude, tfafromirm=tfafromirm)

        #
        # given a directory of candidate template stars with initial TFA
        # applied to them, construct a list of all their eta-normal indices (or
        # other variability metrics, like lomb-scargle FAP).
        # sort it and cut it. make raw/dtr/bkgd plots to verify the sort is
        # sensible. then cut the lists down to ~200 template stars. (or at
        # minimum, n_min_tfa_templates)
        # 
        for ix, apnum in enumerate([1,2,3]):
            templatedir = os.path.join(lcdirectory,
                                       'INITIAL_TFA_LCS',
                                       'ap{}'.format(apnum))
            cand_template_lcpaths = glob(
                os.path.join(templatedir,'*_llc.fits'))
            assert 'ap{}'.format(apnum) in trendlisttfa_paths[ix]
            lcu.round_two_tfa_selection(cand_template_lcpaths,
                                        apnum,
                                        lcdirectory,
                                        trendlisttfa_paths[ix],
                                        n_min_tfa_templates=120,
                                        n_desired_tfa_templates=200)

        lcu.run_tfa(tfalclist_path, trendlisttfa_paths, datestfa_path,
                    lcdirectory, statsdir, nworkers=nworkers,
                    do_bls_ls_killharm=False, npixexclude=npixexclude,
                    tfafromirm=tfafromirm)

        lcu.parallel_merge_tfa_lcs(lcdirectory, nworkers=nworkers)

    if isforce or not os.path.exists(tfastatfile):

        # Catalog projection performed here does some useful things joel's
        # stats file does not for Gaia DR2, catalog mag col corresponds to G_Rp
        # (Gaia "R" band).
        epdlcrequired = not skipepd
        ap.parallel_lc_statistics(lcdirectory, epdlcglob, reformed_cat_file,
                                  tfalcrequired=True,
                                  epdlcrequired=epdlcrequired,
                                  fitslcnottxt=True,
                                  fovcatcols=(0,6), # objectid, magcol from
                                  fovcatmaglabel='r', outfile=tfastatfile,
                                  nworkers=nworkers, workerntasks=500,
                                  rmcols=None, epcols=None, tfcols=None,
                                  rfcols=None, correctioncoeffs=None,
                                  sigclip=5.0, fovcathasgaiaids=True)
    else:
        print('already made TFA LC stats file')

    if os.path.exists(tfaboolstatusfile):
        # NOTE: this invalidates the check for whether the plots exist. however
        # writing them is fairly quick. this makes it happen once.
        os.remove(tfaboolstatusfile)
    if isforce or not os.path.exists(tfaboolstatusfile):
        ap.plot_stats_file(tfastatfile, statsdir, field, binned=False,
                           logy=True, logx=False, correctmagsafter=None,
                           rangex=(5.9,16), observatory='tess',
                           fovcathasgaiaids=True, yaxisval='RMS')
        with open(tfaboolstatusfile+'','w') as f:
            f.write('1\n')
    else:
        print('found done TFA plots (unbinned) through {:s}. continuing.'.
              format(tfaboolstatusfile))

    if binlightcurves:

        binsizes = [3600,21600]

        binnedlcfiles = glob(lcdirectory+'*.binned-*sec-lc.pkl')
        if isforce or len(binnedlcfiles) == 0:
            ap.parallel_bin_lightcurves(
                lcdirectory, '*epdlc', binsizes=binsizes,
                lcexts=('epdlc', 'tfalc'),
                lcmagcols=([27,28,29],[30,31,32]),
                jdcol=0, nworkers=nworkers, workerntasks=1000)
        else:
            print('found >=1 binned lightcurve. continuing.')

        onehr_binstatfile = (
            os.path.join(statsdir,'onehr-binned-lightcurve-statistics.txt'))
        onehrglob = '*.binned-3600sec-lc.pkl'
        sixhr_binstatfile = (
            os.path.join(statsdir,'sixhr-binned-lightcurve-statistics.txt'))
        sixhrglob = '*.binned-21600sec-lc.pkl'

        # make statfiles and MAD vs. mag plots for all binned LCs.
        for binstatfile, binglob, cadence in zip(
            [onehr_binstatfile, sixhr_binstatfile], [onehrglob, sixhrglob],
            binsizes):

            if isforce or not os.path.exists(binstatfile):
                ap.parallel_binnedlc_statistics(
                    lcdirectory, binglob, reformed_cat_file, fovcatcols=(0,6),
                    fovcatmaglabel='r', corrmagsource=None, corrmag_idcol=0,
                    corrmag_magcols=[122,123,124], outfile=binstatfile,
                    nworkers=nworkers, workerntasks=500, sigclip=5)
            else:
                print('found {:s}, continue'.format(binstatfile))

            outprefix = field+'-'+str(cadence)
            ap.plot_stats_file(binstatfile, statsdir, outprefix,
                               binned=cadence, logy=True, logx=False,
                               correctmagsafter=None, rangex=(5.9,16),
                               observatory='tess', yaxisval='RMS')
    else:
        print('will not bin lightcurves or make assoiated statplots')


def run_detrending_on_raw_photometry():
    """
    run detrending on raw photometry, to compare vs. image subtracted

    Steps [1-5] of raw photometry (source extraction, astrometry, creation of a
    source catalog, reforming the source catalog, and raw photometry) have been
    performed.  To detrend, we first "magnitude fit" (see Sec 5.5 of Zhang,
    Bakos et al 2016).  This procedure empirically fits magnitude differences
    (vs. a reference) for each star in each frame, using on-camera position,
    catalog magnitude, catalog color, and subpixel position. Then we run EPD
    and TFA. Procedurally:

    6. run get_magfit_frames to select a single magfit photometry reference and
    set up per-CCD work directories, symlinks, etc. for the next steps.

    7. run make_magfit_config to generate magfit config files for
    MagnitudeFitting.py

    8. run make_fiphot_list to make lists of fiphot files for each CCD.

    9. run MagnitudeFitting.py in single reference mode.

    10. run do_masterphotref.py to get the master mag fit reference.

    11. run MagnitudeFitting.py in master reference mode.

    12. run parallel_collect_lightcurves to collect all lightcurves into .rlc
    files.

    13. run serial_run_epd or parallel_run_epd to do EPD on all LCs.

    14. run parallel_lc_statistics to collect stats on .epdlc files.

    15. run choose_tfa_template to choose TFA template stars using the .epdlc
    stats.

    16. run parallel_run_tfa for TFA to get .tfalc files

    17. run parallel_lc_statistics to collect stats on .tfalc files.

    20. run plot_stats_file to make MAD vs. mag plots for all unbinned and
    binned LCs.
    """

    raise NotImplementedError


def assess_run(statsdir, lcdirectory, starttime, outprefix, fitsdir, projectid,
               field, camera, ccd, tfastatfile, ra_nom, dec_nom,
               projcatalogpath, astromrefpath, sectornum, tfa_templates_path,
               binned=False, make_percentiles_plot=True,
               percentiles_xlim=[4,17], percentiles_ylim=[1e-5,1e-1],
               nworkers=16, moviedir='/nfs/phtess2/ar0/TESS/FFI/MOVIES/',
               skipepd=False):
    """
    write files with summary statistics of run. also, make movies.

    args:
        statsdir (str): e.g.,
        '/nfs/phtess2/ar0/TESS/FFI/LC/TUNE/sector-10/ISP_1-2/stats_files/'

        lcdirectory (str): one level up from statsdir.

        starttime (datetime obj)

    kwargs:
        is statfile binned?

        make_percentiles_plot (bool)
    """

    lcglob = '*_llc.fits'
    tu.plot_lc_positions(lcdirectory, lcglob, statsdir,
                         outname='lc_position_scatter_plot.png',
                         N_desired=20000)

    percentilesfiles = glob(os.path.join(statsdir,'percentiles_*png'))
    if not percentilesfiles:
        lcs.percentiles_RMSorMAD_stats_and_plots(
            statsdir, outprefix, binned=binned,
            make_percentiles_plot=make_percentiles_plot,
            percentiles=[25,50,75],
            percentiles_xlim=percentiles_xlim,
            percentiles_ylim=percentiles_ylim, yaxisval='RMS')
    else:
        print('found percentiles plots')

    # do ACF statistics for say 100 lightcurves
    acf_lcs = _get_random_tfa_lcs(lcdirectory, n_desired_lcs=100)
    outdir = os.path.join(statsdir,'acf_stats')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    lcs.parallel_compute_acf_statistics(
        acf_lcs, outdir, nworkers=nworkers,
        eval_times_hr=[1,2,6,12,24,48,60,96,120,144,192],
        dtrtypes=['IRM','TFA']
    )

    outdir = os.path.join(statsdir,'tfa_templates')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    lcs.plot_tfa_templates(tfa_templates_path, outdir)

    # plot some lightcurves and their ACFs
    pickledir = os.path.join(statsdir, 'acf_stats')
    plot_random_lightcurves_and_ACFs(statsdir, pickledir, n_desired=10,
                                     skipepd=skipepd)

    lcs.acf_percentiles_stats_and_plots(statsdir, outprefix, make_plot=True,
                                        percentiles=[25,50,75],
                                        skipepd=skipepd)

    # check if image noise is gaussian
    if False:
        is_image_noise_gaussian(fitsdir, projectid, field, camera, ccd)

    # just make some lightcurve plots to look at them
    plot_random_lightcurve_subsample(lcdirectory, n_desired_lcs=10,
                                     skipepd=skipepd)

    n_rawlc = len(glob(os.path.join(lcdirectory,'*.grcollectilc')))
    n_epdlc = len(glob(os.path.join(lcdirectory,'*.epdlc')))
    n_tfalc = len(glob(os.path.join(lcdirectory,'*.tfalc')))

    # measure S/N of known HJ transits
    kponchippath = os.path.join(statsdir,'knownplanet_onchip.csv')
    if not os.path.exists(kponchippath):
        if tu.are_known_planets_in_field(ra_nom, dec_nom, kponchippath):
            tu.measure_known_planet_SNR(kponchippath, projcatalogpath, lcdirectory,
                                        statsdir, sectornum, nworkers=nworkers,
                                        skipepd=skipepd)
        else:
            print('did not find any known HJs on this field')
    else:
        print('found {}; skipping planet SNR measurement'.format(kponchippath))

    DO_DEPRECATED = False
    if DO_DEPRECATED:
        # plot and examine size of astrometric shifts
        if not os.path.exists(os.path.join(
            statsdir, 'examine_astrometric_shifts.pickle')
        ):
            examine_astrometric_shifts(fitsdir, astromrefpath, statsdir)

        # make cutouts jpgs of clusters
        tu.make_cluster_cutout_jpgs(sectornum, fitsdir, ra_nom, dec_nom, field,
                                    camera, ccd, statsdir,
                                    clusterdistancecutoff=2000,
                                    nworkers=nworkers)

    _make_movies(fitsdir, moviedir, field, camera, ccd, projectid)

    # how long did the pipeline take? how many lightcurves did it produce?
    endtime = datetime.utcnow()
    howlong = (endtime - starttime).total_seconds()*units.s

    SUMMARY={
        'starttime': starttime.isoformat(),
        'endttime': endtime.isoformat(),
        'howlong_day':howlong.to(units.day).value,
        'howlong_hr':howlong.to(units.hr).value,
        'howlong_min':howlong.to(units.minute).value,
        'howlong_sec':howlong.to(units.second).value,
        'n_rawlc':n_rawlc,
        'n_epdlc':n_epdlc,
        'n_tfalc':n_tfalc
    }
    summarydf = pd.DataFrame(SUMMARY, index=[0])
    summarypath = os.path.join(statsdir,'run_summary.csv')
    summarydf.to_csv(summarypath, index=False)
    print('wrote {}'.format(summarypath))


def main(fitsdir, fitsglob, projectid, field, camnum, ccdnum,
         outdir=sv.REDPATH,
         lcdirectory=None, nworkers=1,
         aperturelist='1.45:7.0:6.0,1.95:7.0:6.0,2.45:7.0:6.0',
         kernelspec='b/4;i/4;d=4/4', convert_to_fitsh_compatible=True,
         anetfluxthreshold=20000, anettweak=6, initccdextent='0:2048,0:2048',
         anetradius=30,
         zeropoint=11.82,
         epdsmooth=21, epdsigclip=10, photdisjointradius=2,
         tuneparameters='true', is_ete6=False,
         cluster_faint_Rp_mag=13, field_faint_Rp_mag=11,
         fiphotfluxthreshold=1000, photreffluxthreshold=1000,
         extractsources=True, binlightcurves=False, get_masks=1,
         tfa_template_sigclip=5.0, tfa_epdlc_sigclip=5.0, translateimages=True,
         reversesubtract=False, skipepd=True, useimagenotfistar=True,
         fixedtfatemplate=None, flagvalues=[-1,4,32,36,2048,2080,2052,2084],
         do_cdips_merge=True, isforce=False
         ):
    """
    args:

        projectid (int): ...

        field (str): e.g., "s0001", for sector 1. This regex is checked.

    kwargs:

        initccdextent (string): <x1>:<x2>,<y1>:<y2> for initial source
        extraction and astrometry.

        anetradius (float): how far apart the index files used for anet can be.
        make this a good deal bigger than the FoV of the image you're using,
        especially if it's wide-field (> 10deg x 10deg).

        anetfluxthreshold (float/int): this value sets the minimum flux for
        bright stars used in the astrometric solution. if it's too low, then
        you get too many stars, and astrometry.net gets confused.

        zeropoint (float): 11.82, from "get_zero_point.py" in ete6 repo.
        since we're doing relative photometry, and the same stars are always in
        the same cameras, the exact value isn't very important.

        extractsources (bool): if True, uses fistar for source extraction
        (with the fiphotfluxthreshold). if False, uses the background catalog
        (e.g., 2MASS), projected onto the frame with WCS, and the
        cluster_faint_Rp_mag cutoff to make the list of sources.
    """

    record_reduction_parameters(fitsdir, fitsglob, projectid, field, camnum,
                                ccdnum, outdir, lcdirectory, nworkers,
                                aperturelist, kernelspec,
                                convert_to_fitsh_compatible, anetfluxthreshold,
                                anettweak, initccdextent, anetradius,
                                zeropoint, epdsmooth, epdsigclip,
                                photdisjointradius, tuneparameters, is_ete6,
                                cluster_faint_Rp_mag, field_faint_Rp_mag,
                                fiphotfluxthreshold, photreffluxthreshold,
                                extractsources, binlightcurves, get_masks,
                                tfa_template_sigclip, tfa_epdlc_sigclip,
                                translateimages, reversesubtract, skipepd,
                                useimagenotfistar, fixedtfatemplate,
                                flagvalues, do_cdips_merge)

    starttime = datetime.utcnow()

    ###########################################################################
    # get list of ete6 reduced images. (different format from images fitsh can
    # work with). trim images, and save to a single-extension fits file.
    if not re.match("s[0-9][0-9][0-9][0-9]", field):
        raise AssertionError('field must be in format "s0001" (case-aware)')
    sectornum = int(field[1:])

    if is_ete6:
        CAL_dir = '/nfs/phtess2/ar0/TESS/SIMFFI/ARCHIVAL/'
        mast_calibrated_ffi_list = np.sort(
            glob(CAL_dir+fitsglob.replace('_cal_img.fits','-s_ffic.fits'))
        )
    else:
        CAL_dir = '/nfs/phtess2/ar0/TESS/FFI/CAL/sector-{:d}'.format(sectornum)
        mast_calibrated_ffi_list = np.sort(
            glob(os.path.join(
                CAL_dir,fitsglob.replace('_cal_img_bkgdsub.fits','-s_ffic.fits')))
        )

    if tuneparameters=='true':
        # select 150 sequential images for pipeline tuning
        mast_calibrated_ffi_list = mast_calibrated_ffi_list[300:450]
    else:
        pass

    RED_dir = (
        '/nfs/phtess2/ar0/TESS/FFI/RED/sector-{:d}/cam{}_ccd{}'.
        format(sectornum, camnum, ccdnum)
    )
    if not os.path.exists(os.path.dirname(RED_dir)):
        os.mkdir(os.path.dirname(RED_dir))
    if not os.path.exists(RED_dir):
        os.mkdir(RED_dir)
    fits_list = np.sort(glob(os.path.join(RED_dir, fitsglob)))
    exists = np.array(list(os.path.exists(f) for f in fits_list)).astype(bool)
    mostexist = len(exists)!=0
    if len(exists)>0:
        mostexist &= len(exists[exists])/len(exists)>0.9

    statsdir = os.path.dirname(lcdirectory) + '/stats_files/'
    camera = int(camnum)
    ccd = int(ccdnum)
    tfastatfile = os.path.join(statsdir, 'camera' + str(camera) +
                               '_ccd' + str(ccd) + '.tfastats')
    alreadydone = os.path.exists(tfastatfile)

    if (convert_to_fitsh_compatible and get_masks
        and (isforce or (not mostexist and not alreadydone))
    ):

        # First, trim each frame, and turn it into a single extension FITS
        # image. Then pass a median filter over it, over box size 64 x 64.
        # Plot the results, make a movie of them. Then mask saturated stars,
        # and frames with the DQUALITY=32 flag.
        tu.parallel_trim_get_single_extension(mast_calibrated_ffi_list,
                                              RED_dir, projectid,
                                              nworkers=nworkers, isforce=isforce)

        fits_list = np.sort(glob(os.path.join(
            RED_dir,
            fitsglob.replace('_cal_img_bkgdsub.fits','_cal_img.fits')))
        )

        #tu.parallel_bkgd_subtract(fits_list, method='none',
        #                          isfull=True, k=None, k_sigma=None,
        #                          outdir=RED_dir, nworkers=nworkers)
        tu.parallel_bkgd_subtract(fits_list, method='boxblurmedian',
                                  isfull=True, k=48, k_sigma=48,
                                  outdir=RED_dir, nworkers=nworkers,
                                  isforce=isforce)

        tu.parallel_plot_median_filter_quad(RED_dir, nworkers=nworkers,
                                            isforce=isforce)

        moviedir='/nfs/phtess2/ar0/TESS/FFI/MOVIES/'
        outmp4path = os.path.join(
            moviedir, 'sector{}_cam{}_ccd{}_quad_bkgd.mp4'.
            format(sectornum, camnum, ccdnum)
        )
        quadglob = os.path.join(RED_dir, 'tess*_cal_img_bkgd.png')
        if isforce or not os.path.exists(outmp4path):
            if os.path.exists('/home/lbouma/bin/ffmpeg'):
                ffmpegpath='/home/lbouma/bin/ffmpeg'
            else:
                ffmpegpath=which('ffmpeg')
            if ffmpegpath != '':
                iu.make_mp4_from_jpegs(quadglob, outmp4path,
                                       ffmpegpath=ffmpegpath)
            else:
                print('ffmpeg not found - cannot make movies')

        # The TESS handbook quotes saturation beginning at "around 2e5
        # electrons". gain ~= 5 electrons per ADU, so this corresponds to
        # saturation at ~40000 ADU. From inspection of the frames, the
        # mustaches start to become clearly visible around 1e5 ADU. We don't
        # want to set the saturation level too low, and mask many stars,
        # particularly in crowded regions.
        fits_list = np.sort(glob(os.path.join(RED_dir, fitsglob)))

        tu.parallel_mask_saturated_stars(fits_list, saturationlevel=80000,
                                         nworkers=nworkers)

        # append CCD temperature information to headers
        engdatadir = '/nfs/phtess2/ar0/TESS/FFI/ENGINEERING/'
        temperaturepklpath = os.path.join(
            engdatadir,
            'sector{:s}_ccd_temperature_timeseries.pickle'.
            format(str(sectornum).zfill(4))
        )
        if not os.path.exists(temperaturepklpath):
            tu.make_ccd_temperature_timeseries_pickle(sectornum)

        tu.parallel_append_ccd_temperature_to_hdr(
            fits_list, temperaturepklpath
        )

    else:
        print('skipping initial trimming & background filtering')
        pass

    # symlink images so that they "live" in each reduction directory.
    if isforce or not alreadydone:
        fits_list = np.sort(glob(os.path.join(RED_dir,fitsglob)))
        for fitspath in fits_list:
            dstname = os.path.basename(fitspath)
            dstdir = fitsdir
            dstpath = os.path.join(dstdir, dstname)
            if isforce or not os.path.exists(dstpath):
                if isforce:
                    try:
                        os.symlink(fitspath, dstpath)
                    except OSError as e:
                        if e.errno == errno.EEXIST:
                            os.remove(dstpath)
                            os.symlink(fitspath, dstpath)
                        else:
                            raise e
                else:
                    os.symlink(fitspath, dstpath)
                print('symlink {}->{}'.format(fitspath, dstpath))
            else:
                print('SKIP symlink {}->{}'.format(fitspath, dstpath))
        fits_list = np.sort(glob(os.path.join(fitsdir,fitsglob)))

        # remove frames tagged as momentum dumps, tagged as coarse point, and
        # tagged as both. further, mask frames that are at known BAD TIMES.
        # (these would get bit value 128 in LCs processed by SPOC; they get
        # value -1 in pipe-trex).  move images with bad quality flags, and at
        # bad times, to the badframes directory.
        if not os.path.exists(os.path.join(fitsdir,'badframes')):
            os.mkdir(os.path.join(fitsdir,'badframes'))
        tu.parallel_move_badframes(fits_list, flagvalues=flagvalues,
                                   nworkers=nworkers)

    ###########################################################################

    # TESS specific variables.
    # [degrees]. 12 degrees produces LCs that looked around 12 degrees from
    # the center of field. 24 is simply to play it safe (more likely (12^2 +
    # 12^2)^(1/2) = 17 is the needed number...)
    catboxsize = 24

    # get gain, ccdextent, zeropoint, exposure time from header.
    ntry_rand_fits=0
    while True:
        rand_fits = fits_list[np.random.randint(0, high=len(fits_list))]
        if os.path.exists(rand_fits):
            break
        if ntry_rand_fits >= 10000:
            raise AssertionError('Cannot find a random fits to extract header info after moving the badframes')

    hdu_list = fits.open(rand_fits)
    hdr = hdu_list[0].header

    ccdgain = hdr['GAINA'] # electrons/count, from CCD output A. (there are four!)
    exptime = int(np.round(hdr['TELAPSE']*24*60*60)) # in seconds, 1800
    ra_nom = hdr['CRVAL1']  # RA at CRPIX1, CRPIX2. Roughly "camera boresight".
    dec_nom = hdr['CRVAL2'] # DEC at CRPIX1, CRPIX2

    catalog, catra, catdec, catbox = 'GAIADR2', ra_nom, dec_nom, catboxsize
    catalog_file = (
        '%s-RA%s-DEC%s-SIZE%s.catalog' % (catalog, catra, catdec, catbox)
    )
    reformed_cat_file = catalog_file.replace('.catalog', '.reformed_catalog')

    catalog_file = os.path.join(outdir, catalog_file)
    reformed_cat_file = os.path.join(outdir, reformed_cat_file)

    ###########################################################################

    if isforce or not is_presubtraction_complete(outdir, fitsglob, lcdirectory, outdir,
                                      extractsources=extractsources):
        get_files_needed_before_image_subtraction(
            sectornum, camera, ccd,
            fitsdir, fitsglob, outdir, initccdextent, ccdgain, zeropoint, exptime,
            ra_nom, dec_nom, catra, catdec, catboxsize, catalog, catalog_file,
            reformed_cat_file, field_faint_Rp_mag, fnamestr=fitsglob,
            anetfluxthreshold=anetfluxthreshold, anetradius=anetradius,
            fistarglob='*.fistar', width=13, anettweak=anettweak, xpix=2048,
            ypix=2048, cols=(2,3), brightrmag=6.0, faintrmag=cluster_faint_Rp_mag,
            fiphotfluxthreshold=fiphotfluxthreshold, aperturelist=aperturelist,
            nworkers=nworkers, extractsources=extractsources,
            useimagenotfistar=useimagenotfistar, do_cdips_merge=do_cdips_merge,
            isforce=isforce)

    else:
        print('found fistar, fiphot, and wcs files. proceeding to image '
              'subtraction.')

    if not initial_wcs_worked_well_enough(outdir, fitsglob):
        raise AssertionError('if initial wcs failed on majority of frames, '
                             'you need to fix that.')

    #############################################################
    # run image subtraction convolution, then do the photometry #
    #############################################################

    camera = int(camnum)
    ccd = int(ccdnum)

    fieldinfo = {}
    fieldinfo['camera'] = camera
    fieldinfo['ccd'] = ccd
    fieldinfo['projectid'] = projectid
    fieldinfo['field'] = field

    photparams = {
        'ccdgain': ccdgain, 'ccdexptime': exptime, 'zeropoint': zeropoint
    }

    photreftype, dbtype = 'onenight', 'postgres'

    epdlcglob, tfalcglob = '*_llc.fits', '*_llc.fits'
    statsdir = os.path.dirname(lcdirectory) + '/stats_files/'
    for dirname in [lcdirectory, statsdir]:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    epdstatfile = statsdir + 'camera' + str(camera) + '_ccd' + str(ccd) + '.epdstats'
    tfastatfile = statsdir + 'camera' + str(camera) + '_ccd' + str(ccd) + '.tfastats'
    vartoolstfastatfile = os.path.join(statsdir, 'vartools_tfa_stats.txt')

    xtrnsglob = fitsglob.replace('.fits','-xtrns.fits')
    if reversesubtract:
        iphotpattern = fitsdir+'rsub-????????-'+fitsglob.replace('.fits','.iphot')
    else:
        iphotpattern = fitsdir+'nsub-????????-'+fitsglob.replace('.fits','.iphot')

    if isforce or not is_imagesubtraction_complete(fitsdir, fitsglob, lcdirectory):

        run_imagesubtraction(fitsdir, fitsglob, fieldinfo, photparams,
                             fits_list, photreftype, dbtype, reformed_cat_file,
                             xtrnsglob, iphotpattern, lcdirectory, outdir,
                             projectid, ra_nom, dec_nom,
                             kernelspec=kernelspec, refdir=sv.REFBASEDIR,
                             nworkers=nworkers, aperturelist=aperturelist,
                             photdisjointradius=photdisjointradius,
                             colorscheme='bwr',
                             photreffluxthreshold=photreffluxthreshold,
                             extractsources=extractsources,
                             translateimages=translateimages,
                             reversesubtract=reversesubtract,
                             useimagenotfistar=useimagenotfistar,
                             isforce=isforce)
    else:
        print('found that image subtraction is complete.')

    if isforce or not alreadydone:
        run_detrending(epdstatfile, tfastatfile, vartoolstfastatfile, lcdirectory,
                       epdlcglob, reformed_cat_file, statsdir, field, fitsdir,
                       fitsglob, camera, ccd, projectid,
                       epdsmooth=epdsmooth, epdsigclip=epdsigclip,
                       nworkers=nworkers, binlightcurves=binlightcurves,
                       tfa_template_sigclip=tfa_template_sigclip,
                       tfa_epdlc_sigclip=tfa_epdlc_sigclip, skipepd=skipepd,
                       fixedtfatemplate=fixedtfatemplate, isforce=isforce)

    statsdir = os.path.dirname(epdstatfile)+'/'
    outprefix = str(field)+'-'+str(projectid)
    refdir = sv.REFBASEDIR
    projcatalogpath = os.path.join(
        refdir,
        'proj{:d}-{:s}-cam{:s}-ccd{:s}-combinedphotref-{:s}.projcatalog'.
        format(projectid, str(field), str(camera), str(ccd), photreftype)
    )
    astromrefglob = os.path.join(
        refdir,
        'proj{:d}-camera{:s}-ccd{:s}-astromref-tess*-{:s}-{:s}-{:s}-*.fits'.
        format(projectid, str(camera), str(ccd), str(field), str(camera),
               str(ccd))
    )
    astromrefpath = glob(astromrefglob)
    if len(astromrefpath) < 1:
        raise AssertionError('FATAL ERR! astromrefglob wrong for run assessment')
    astromrefpath = astromrefpath[0]
    tfa_templates_path = os.path.join(statsdir,'trendlist_tfa_ap2.txt')
    if not os.path.exists(tfa_templates_path):
        raise AssertionError('where is tfa templates???')
    assess_run(statsdir, lcdirectory, starttime, outprefix, fitsdir, projectid,
               field, camera, ccd, tfastatfile, ra_nom, dec_nom,
               projcatalogpath, astromrefpath, sectornum, tfa_templates_path,
               binned=False, make_percentiles_plot=True, percentiles_xlim=None,
               nworkers=nworkers, skipepd=skipepd)


def check_args(args):
    if not args.fitsdir:
        print('fitsdir must not be null')
    if not args.fitsglob:
        print('fitsglob must not be null')
    if not args.field:
        print('field must not be null')
    if not ( (args.tuneparameters=='true') or (args.tuneparameters=='false') ):
        raise AssertionError('boolean tuneparameters must be set as string.')
    if not (args.camnum in [1,2,3,4]) and (args.ccdnum in [1,2,3,4]):
        print('camnum and ccdnum must be given')
    if not args.skipepd:
        raise AssertionError('skipepd must be true')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=('Given images, make lightcurves.')
    )

    parser.add_argument('--fitsdir', type=str, default=None,
        help=('e.g., /foo/FFI/RED'))

    parser.add_argument('--fitsglob', type=str, default=None,
        help=('e.g., *.fits.gz. Chosen frames are fitsdir+fitsglob.'))

    parser.add_argument('--projectid', type=int, default=1,
        help=('unique integer identifying this reduction run'))

    parser.add_argument('--field', type=str, default=None,
        help=('field identifier string, e.g., "sector1"'))

    parser.add_argument('--camnum', type=int, default=0,
        help=('camera id number'))
    parser.add_argument('--ccdnum', type=int, default=0,
        help=('ccd id number'))

    parser.add_argument('--outdir', type=str, default=None,
        help=('e.g., /foo/FFI/RED'))

    parser.add_argument('--lcdirectory', type=str, default=None,
        help=('e.g., /foo/RED/LC'))

    parser.add_argument(
        '--aperturelist', type=str,
        default="1.45:7.0:6.0,1.95:7.0:6.0,2.45:7.0:6.0",
        help=('formatted as "inner radius: outer radius: bkgd annulus width"')
    )

    parser.add_argument(
        '--kernelspec', type=str, default='b/4;i/4;d=4/4',
        help=('https://fitsh.net/wiki/man/ficonv explains the formatting.'
              'i/<spatial order> identity kernel (a.k.a. flux term) with the'
              'specified order of polynomial spatial variations.'
              'b/<spatial order> constant offset kernel (a.k.a. background'
              'term) with the specified order of polynomial spatial'
              'variations'
              'd=<size>/<spatial order> discrete kernel with the half-size of'
              '<size> and the specified order of polynomial spatial variations'
              'g=<size>,<sigma>,<order>/<spatial order> Gaussian kernel with'
              'the half-size of <size>, standard deviation of <sigma> and'
              'Hermite basis order of <order>, with the specified order of'
              'polynomial spatial variations'
             )
    )

    parser.add_argument(
        '--photdisjointradius', type=float, default=2,
        help=('https://fitsh.net/wiki/man/fiphot gives details. '
              'During the bacground determination on the aperture annuli, '
              'omit the pixels which are closer to the other centroids than '
              'the specified radius.'
             )
    )

    parser.add_argument('--anetfluxthreshold', type=int, default=20000,
        help=(
            'flux threshold used to identify bright stars in anet '
            'initial wcs solution attempt of frames.'
        )
    )
    parser.add_argument('--anettweak', type=int, default=6,
        help=(
            'SIP (Simple Imaging Polynomial) order. Higher = more flexibility '
            'for frame distortion. Max of 10.'
        )
    )
    parser.add_argument('--anetradius', type=float, default=30.,
        help=(
            'Radius (in deg) over which to search for index files in '
            'astrometry.net or net. For TESS, you want to go big -- e.g., 30.'
        )
    )
    parser.add_argument(
        '--initccdextent', type=str,
        default="0:2048,0:2048",
        help=(' section <x1>:<x2>,<y1>:<y2> for initial source extraction and '
              'astrometry. (not for initial photometry).')
    )

    parser.add_argument('--epdsmooth', type=int, default=21,
        help=(
            'number of cadences used when passing a median filter over the'
            'magnitude time series before EPD (see '
            'imagesubphot.epd_magseries_imagesub). For HATPI default is 21, or'
            ' a ~10 minute median filter. For TESS FFIs, try 11 = 5.5hrs.'
        )
    )
    parser.add_argument('--epdsigclip', type=float, default=10.0,
        help=(
            'sigma clipping aplpied to epd lightcurves. assumed symmetric.'
        )
    )

    parser.add_argument('--nworkers', type=int, default=1,
        help=('number of workers to thread over'))

    parser.add_argument(
        '--convert_to_fitsh_compatible', dest='cfc', action='store_true',
        help=('converts calibrated frames to single-fits-header fitsh readable'
              'objects. if already done, will find them and skip.')
    )
    parser.add_argument(
        '--no-convert_to_fitsh_compatible', dest='cfc', action='store_false',
        help=('don\'t do fitsh-compatibility conversion.')
    )
    parser.set_defaults(cfc=True)

    parser.add_argument(
        '--binlightcurves', dest='binlcs', action='store_true',
        help=('will bin lightcurves to 1hr and 6hr cadence. only purpose: '
              'making plots to understand the red noise')
    )
    parser.add_argument(
        '--no-binlightcurves', dest='binlcs', action='store_false',
        help=('don\'t bin lightcurves (it takes a while).')
    )
    parser.set_defaults(binlcs=True)

    parser.add_argument(
        '--tuneparameters', type=str,
        default="true",
        help=('TUNING: iterate through different img subtraction parameters '
              'if true. Gain speed by selecting a small set of frames. '
              'Otherwise, run in FULL reduction mode.')
    )

    parser.add_argument('--cluster_faint_Rp_mag', type=float, default=13.0,
        help=('faint GAIA catalog Rp magnitude used to make fovcat, '
              'which sets the cluster stars that get photometered'))
    parser.add_argument('--field_faint_Rp_mag', type=float, default=11.0,
        help=('faint GAIA catalog Rp magnitude used to make fovcat, '
              'which sets the field stars that get photometered'))

    parser.add_argument('--fiphotfluxthreshold', type=int, default=1000,
        help=('faint flux threshold used to do initial raw photometry on '
              'frame, in pre-img subtraction.'
             ))
    parser.add_argument('--photreffluxthreshold', type=int, default=1000,
        help=('faint flux threshold used to extract sources on photometric '
              'reference frame, in image subtraction. (these sources will '
              'then be subtracted against).'
             ))
    parser.add_argument('--extractsources', type=int, default=1,
        help=(
            'extractsources (int): if 1, uses fistar for source extraction '
            '(with the fiphotfluxthreshold). if 0, use the background catalog '
            '(e.g., 2MASS), projected onto the frame with WCS, and the '
            'cluster_faint_Rp_mag cutoff to make the list of sources. ')
         )

    parser.add_argument('--tfa_template_sigclip', type=float, default=5.,
        help=('sigclip tfa templates by this much'))
    parser.add_argument('--tfa_epdlc_sigclip', type=float, default=10000.,
        help=('sigclip EPD lightcurves by this much before applying TFA. '
              'using this option is probably always a bad idea.'))

    parser.add_argument(
        '--translateimages', dest='trnsimgs', action='store_true',
        help=('translate images to the astrometric reference.')
    )
    parser.add_argument(
        '--no-translateimages', dest='trnsimgs', action='store_false',
        help=('do not translate images to astrometric reference.')
    )
    parser.set_defaults(trnsimgs=True)

    parser.add_argument(
        '--reversesubtract', dest='reversesubtract', action='store_true',
        help=('diff img = reference - model, for model = (img*kernel) + bkgd.')
    )
    parser.add_argument(
        '--no-reversesubtract', dest='reversesubtract', action='store_false',
        help=('diff img = image - model, for model = (ref*kernel) + bkgd.')
    )
    parser.set_defaults(reversesubtract=False)

    parser.add_argument(
        '--skipepd', dest='skipepd', action='store_true',
        help=('skip EPD processing, run TFA on IRM mags')
    )
    parser.add_argument(
        '--no-skipepd', dest='skipepd', action='store_false',
        help=('do not skip EPD processing. run TFA on EPD mags.')
    )
    parser.set_defaults(skipepd=True)

    parser.add_argument(
        '--fixedtfatemplate', type=str,
        default=None,
        help=('(for optimization): find the TFA template stars lists used in '
              'VARTOOLS TFA call in this directory. idea is that you\'ve done a '
              'previous run, and now want to fix the template stars to tweak '
              'other things.')
    )

    parser.add_argument(
        '--force', dest='isforce', action='store_true',
        help=('force overwrite and redo previously completed steps.')
    )
    parser.set_defaults(isforce=False)

    args = parser.parse_args()

    check_args(args)

    if args.extractsources not in [0,1]:
        raise AssertionError('extractsources must be 0 or 1')
    if args.extractsources == 1:
        extractsources = True
    else:
        extractsources = False

    np.random.seed(42)

    main(args.fitsdir, args.fitsglob, args.projectid, args.field,
         args.camnum, args.ccdnum,
         outdir=args.outdir, lcdirectory=args.lcdirectory,
         nworkers=args.nworkers, aperturelist=args.aperturelist,
         convert_to_fitsh_compatible=args.cfc, kernelspec=args.kernelspec,
         epdsmooth=args.epdsmooth, epdsigclip=args.epdsigclip,
         photdisjointradius=args.photdisjointradius,
         tuneparameters=args.tuneparameters,
         anetfluxthreshold=args.anetfluxthreshold,
         anettweak=args.anettweak, initccdextent=args.initccdextent,
         anetradius=args.anetradius,
         cluster_faint_Rp_mag=args.cluster_faint_Rp_mag,
         field_faint_Rp_mag=args.field_faint_Rp_mag,
         fiphotfluxthreshold=args.fiphotfluxthreshold,
         photreffluxthreshold=args.photreffluxthreshold,
         extractsources=extractsources,
         binlightcurves=args.binlcs,
         tfa_template_sigclip=args.tfa_template_sigclip,
         tfa_epdlc_sigclip=args.tfa_epdlc_sigclip,
         translateimages=args.trnsimgs,
         reversesubtract=args.reversesubtract,
         skipepd=args.skipepd,
         fixedtfatemplate=args.fixedtfatemplate,
         isforce=args.isforce
    )
