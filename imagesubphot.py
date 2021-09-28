#!/usr/bin/env python

"""imagesubphot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - March 2015

This contains functions to do image subtraction photometry.

GENERAL ORDER OF THINGS

0. you need reduced frames with accompanying .fiphot (photometry) files, and
   .fistar (source detection) files, as well as frame-projected source detection
   lists for each frame (.sourcelist). see aperturephot.py's docstring (steps 1
   through 5) for how to get to this state. also see framecalib.py for how to
   get from raw frames to calibrated reduced frames.

   (see /nfs/phs3/ar1/S/HP0/PHOT_WB/ccd5-work on phs3 for an example of a
    directory that has most of the needed stuff in it)

1. the first order of business is to select an astrometric reference frame
   (astromref) using select_astromref_frame. this should be a frame that has the
   sharpest and roundest stars. see select_astromref_frame below to see other
   useful selectors that are applied.

   (if using autoimagesub.py with sqlite3, use generate_astromref for this
   step.  if instead using PostgreSQL, parallel_frames_to_database with the
   "calibrated frames" arg in the appropriate place, then
   dbgen_get_astromref does this. The latter is the faster
   option.)

2. next, use get_smoothed_xysdk_coeffs to generate files that contain the
   smoothed S, D, K coefficients for each source detection list. this is needed
   later when we do photometry on the subtracted frames.

3. use get_astromref_shifts to calculate the X and Y coordinate shifts between
   the selected astromref and all other frames we're working on. these will be
   used to shift these other frames to the coordinate system of the astromref,
   which is required to subtract these frames cleanly.

   (if using autoimagesub.py with PostgreSQL database, use
   framelist_make_xtrnsfits. Then put the shifted aref-transformed frames into
   the database with parallel_frames_to_database, frametype set to be
   'arefshifted_frames')

4. use transform_frames_to_astromref to do the actual shifting of all frames to
   the coordinate system of the astromref. this produces new FITS files with the
   '-xtrns.fits' postfix in their filenames. these are the files we'll use from
   now on.

   (if you ran framelist_make_xtrnsfits, this step was already done)

5. use generate_astromref_registration_info to generate a file that grids up the
   astromref source detections into a 30 x 30 grid based on their x and y
   coordinates. not sure what this does exactly, but the file it generates is
   required by the actual convolution steps later.

   (skip if running off autoimagesub)

6. the next thing to do is to select a bunch of frames that can serve as
   photometric reference frames (photrefs). use select_photref_frames for
   this. see the docstring there for the list of selectors used. we'll then
   stack these photrefs into a combined photref later.

   (use ais.generate_photref_candidates_from_xtrns if running off autoimagesub)

7. now that we have photrefs, we have to convolve their PSFs. we convolve their
   PSFs to one of the photrefs in this list. this frame should have round stars,
   low background, but somewhere around the softest PSF FWHM in the list of
   candidate photrefs (we don't want the sharpest of the images -> so choosing
   the astromref as the photref is a bad idea).

   in this way, we match both the PSF of the image and the coordinates; these
   are needed to combine the frames correctly. use covolve_photref_frames for
   this task. this function produces FITS with PHOTREF- prefixes to indicate
   that these are the convolved photref frames.

8. use combine_frames to combine all PHOTREF-*-xtrns.fits frames. this creates a
   single high quality photometric reference frame that we'll subtract from all
   other frames to produce difference images.

   (use combination of amend_candidate_photrefs and generate_combined_photref
   if running via autoimagesub. This also handles step 9 below.)

9. get raw photometry on this combined photref by using
   photometry_on_combined_photref. this produces the base photometry values that
   we'll be diffing from those found in the difference images to get difference
   magnitudes. we need to redo source extraction and photometry on this frame
   (possibly with anet to find actual sources). best not to use the existing
   sourcelist for the normal frame counterpart to the combined photometric
   reference frame.

10. use convolve_and_subtract_frames to convolve all other frames to the
    combined photref, and generate difference images. the produced images will
    have a subtracted- prefix in their filenames.

11. finally, use photometry_on_subtracted_frames to do photometry on the
    subtracted frames to produce difference magnitudes for each image. these
    calculated mags are put into .iphot files.

12. use parallel_collect_imagesub_lightcurves to collect the .iphot files into .ilc
    lightcurve files containing image subtraction photometric timeseries for
    each star.

the columns in these .ilc files are:

    rjd    Reduced Julian Date (RJD = JD - 2400000.0)
    rstfc  Unique frame key ({STID}-{FRAMENUMBER}_{CCDNUM})
    hat    HAT ID of the object
    xcc    original X coordinate on CCD before shifting to astromref
    ycc    original y coordinate on CCD before shifting to astromref
    xic    shifted X coordinate on CCD after shifting to astromref
    yic    shifted Y coordinate on CCD after shifting to astromref
    bgv    Background value
    bge    Background measurement error
    fsv    Measured S value
    fdv    Measured D value
    fkv    Measured K value
    irm1   Instrumental magnitude in aperture 1
    ire1   Instrumental magnitude error for aperture 1
    irq1   Instrumental magnitude quality flag for aperture 1 (0 or G OK, X bad)
    irm2   Instrumental magnitude in aperture 2
    ire2   Instrumental magnitude error for aperture 2
    irq2   Instrumental magnitude quality flag for aperture 2 (0 or G OK, X bad)
    irm3   Instrumental magnitude in aperture 3
    ire3   Instrumental magnitude error for aperture 3
    irq3   Instrumental magnitude quality flag for aperture 3 (0 or G OK, X bad)

13. run serial_run_epd_imagesub or parallel_run_epd_imagesub to do EPD on all
LCs.

the next few steps are common between imagesubphot.py and aperturephot.py, so
you can use the functions there for them. make sure to use columns 12, 15, 18
for the reduced magnitudes (rmcols) instead of the default, since these are
different from the aperture photometry lightcurves.

14. run parallel_lc_statistics to collect stats on .epdlc files.

15. run choose_tfa_template to choose TFA template stars using the .epdlc stats.

16. run parallel_run_tfa for TFA to get .tfalc files (and .tfalc.TF{1,2,3}
    files).

17. run parallel_lc_statistics to collect stats on .tfalc files.

18. run parallel_bin_lightcurves to bin LCs to desired time-bins.

19. run parallel_binnedlc_statistics to collect stats for the binned LCs.

20. run plot_stats_file to make RMS vs. mag plots for all unbinned and binned
    LCs.

21. run plot_magrms_comparison to compare the mag-RMS relation for various
    CCDs. note: you can also use this to compare the aperture photometry LCs
    produced by aperturephot.py to image subtraction photometry LCs produced by
    this module.

22. run plot_ismphot_comparison to compare against ISM photometry statistics for
    the same field (requires common stars). this step is pretty much obsolete
    since we're now using the same functions to produce ISM and AP lightcurve
    stats.
"""

#############
## IMPORTS ##
#############

import os, glob, time, shutil
import os.path
import multiprocessing as mp
import pandas as pd

try:
    import subprocess32 as subprocess
    from subprocess32 import check_output
    import cPickle as pickle
except:
    import subprocess
    from subprocess import check_output
    import pickle

from datetime import datetime
import shlex, re, json, random
from hashlib import md5
from copy import deepcopy
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from functools import partial

# used for fast random access to lines in text files
from linecache import getline

import sqlite3

import numpy as np
from numpy import isfinite as npisfinite
import numpy.random as nprand

from scipy.spatial import cKDTree as kdtree
from scipy.signal import medfilt
from scipy.linalg import lstsq
from scipy.stats import sigmaclip as stats_sigmaclip
from scipy.optimize import curve_fit

from astropy import units as units

import scipy.stats

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from astropy.io import fits

import imageutils
from imageutils import get_header_keyword, fits_to_full_jpeg, \
        clipped_linscale_img

# get fiphot binary reader
try:
    from HATpipepy.Common.BinPhot import read_fiphot
    HAVEBINPHOT = True
except:
    print("can't import binary fiphot reading functions from "
          "HATpipe, binary fiphot files will be unreadable!")
    HAVEBINPHOT = False

# get useful things from aperturephot
from aperturephot import extract_frame_sources, anet_solve_frame, \
    do_photometry, astrometrydotnet_solve_frame, fistarfile_to_xy

import shared_variables as sv

from astrobase import lcmath
from scipy.interpolate import splprep, splev
from numpy import array as nparr

#################
## DEFINITIONS ##
#################

# set this to show extra info
DEBUG = True

# CCD minimum and maximum X,Y pixel coordinates
# used to strip things outside FOV from output of make_frame_sourcelist
CCDEXTENT = {'x':[0.0,2048.0],
             'y':[0.0,2048.0]}

# zeropoint mags for the HATPI lenses given exp time of 30 seconds
# from Chelsea's src directory on phs3: run_phot_astrom.py (2014-12-15)
# FIXME: check where these came from and fix if out of date, especially if
# cameras moved around
ZEROPOINTS = {3:17.11,
              5:17.11,
              6:17.11,
              7:17.11,
              8:16.63}

# used to get the station ID, frame number, and CCD number from a FITS filename
FRAMEREGEX = re.compile(r'(\d{1})\-(\d{6}\w{0,1})_(\d{1})')


#######################
## COMMAND TEMPLATES ##
#######################

XYSDKCMD = ('grtrans {fistarfile} --col-xy 2,3 --col-fit 6,7,8 '
            '--col-weight 10 --order 4 '
            '--iterations 3 --rejection-level 3 '
            '--comment --output-transformation {xysdkfile}')

FRAMESHIFTCALCCMD = (
    'grmatch --match-points '
    '-r {astromref} --col-ref 2,3 --col-ref-ordering +9 '
    '-i {fistartoshift} --col-inp 2,3 --col-inp-ordering +9 '
    '--weight reference,column=9 '
    '--triangulation maxinp=5000,maxref=5000,conformable,auto,unitarity=0.01 '
    '--order 4 --max-distance 1 --comment '
    '--output-transformation {outtransfile} '
    '--output /dev/null'
    )

FRAMETRANSFORMCMD = ('fitrans {frametoshift} -k '
                     '--input-transformation {itransfile} '
                     '--reverse -o {outtransframe}')

PHOTREFCONVOLVECMD = ('ficonv -i {targetframe} '
                      '-r {frametoconvolve} '
                      '-it {convregfile} '
                      '-k "{kernelspec}" '
                      '-oc {outputfile}')

FRAMECOMBINECMD = ('ficombine {framelist} -m {combinemethod} -o {outfile}')

CONVOLVESUBFRAMESCMD = ('ficonv -r {frametoconvolve} '
                        ' -i {targetframe} '
                        '-it {convregfile} '
                        '-k "{kernelspec}" '
                        '-ok {outputkernel} '
                        '-os {outputfile}')

COMBINEDREFPHOTCMD = (
    "fiphot --input {photref} "
    "--input-list {srclist} "
    "--col-id {srclist_idcol} "
    "--col-xy {srclist_xycol} "
    "--gain {ccdgain} "
    "--mag-flux {zeropoint},{exptime} "
    "--apertures '{aperturestring}' "
    "--sky-fit 'mode,sigma=3,iterations=2' --disjoint-radius 2 "
    "--serial {photrefbase} "
    "--format 'IXY-----,sMm' --nan-string 'NaN' "
    "--aperture-mask-ignore 'saturated' "
    "--comment '--comment' --single-background 3 "
    "-op {outfile} -k"
)


# FIXME: why do we not get the background values by default? EPD needs these
# ANSWER: we're ignoring the background for the subtracted frame photometry, and
#         EPD on imagesub photometry doesn't use the background values
#         extract them anyway, since it might be useful

SUBFRAMEPHOTCMD = (
    "fiphot --input-subtracted {subtractedframe} "
    "--input-raw-photometry {photrefrawphot} "
    "--sky-fit mode,iterations=2,sigma=3 "
    "--format IXY-----,BbFfMms "
    "--mag-flux {zeropoint},{exptime} "
    "--gain {ccdgain} "
    "--disjoint-radius {disjointradius} "
    "{magfitstr}"
    "--input-kernel {subtractedkernel} "
    "--spline "
    "--nan-string 'NaN' --single-background 3 "
    "--comment --output - | "
    "grtrans --col-xy 2,3 "
    "--input-transformation {subtracteditrans} "
    "--col-out 4,5 "
    "--output - | "
    "grtrans --col-xy 4,5 "
    "--input-transformation {subtractedxysdk} "
    "--col-out 6,7,8 "
    "--output - | sort -k 1 > {outiphot}"
)

GRCOLLECTCMD = ('grcollect {fiphotfileglob} --col-base {objectidcol} '
                '--prefix {lcdir} --extension {lcextension} '
                '--max-memory {maxmemory}'
               )


####################
## SQLITE SCHEMAS ##
####################

PHOTS_TABLE = 'create table phots (phot text, rjd double precision, frame text)'
HATIDS_TABLE = 'create table hatids (hatid text, phot text, photline integer)'
META_TABLE = 'create table metainfo (photdir text, framedir text)'
PRAGMA_CMDS = 'pragma journal_mode = WAL'

PHOTS_INDEX_CMD = 'create index phots_index on phots (phot)'
HATIDS_INDEX_CMD = 'create index hatid_index on hatids (hatid)'
HATIDS_PHOT_INDEX_CMD = 'create index hatid_phot_index on hatids (phot)'

PHOTS_INSERT_CMD = 'insert into phots values (?,?,?)'
HATIDS_INSERT_CMD = 'insert into hatids values (?,?,?)'
META_INSERT_CMD = 'insert into metainfo values (?,?)'

PHOT_SELECT_CMD = ('select a.rjd, a.phot, b.photline from '
                   'phots a join hatids b on (a.phot = b.phot) '
                   'where b.hatid = ? order by a.rjd')
META_SELECT_CMD = ('select * from metainfo')
DISTINCT_HATIDS_CMD = ('select distinct hatid from hatids')


##################################
## ASTROMETRIC REFERENCE FRAMES ##
##################################

def select_astromref_frame(fitsdir,
                           fitsglob,
                           srclistdir=None,
                           srclistext='.fistar',
                           photdir=None,
                           photext='.fiphot',
                           jpeg=True,
                           overrideref=None):
    """
    This picks an astrometric reference frame.

    We're looking for (in order):

    - highest median S (smallest FWHM)
    - median D value closest to zero (roundest stars)
    - lowest median background
    - largest number of sources with good extractions
    """

    # if fitsdir is already a list, then use it directly
    if isinstance(fitsdir, list):
        fitslist = fitsdir
    # otherwise, get the list of fits to use
    else:
        fitslist = glob.glob(os.path.join(fitsdir, fitsglob))

    print('%sZ: %s FITS files found, '
          'finding photometry and source lists...' %
          (datetime.utcnow().isoformat(),
           len(fitslist)))

    goodframes = []
    goodphots = []
    goodsrclists = []

    # associate the frames with their fiphot files
    for fits in fitslist:

        # find the fistar file
        if not srclistdir and not isinstance(fitsdir, list):
            srclistdir = fitsdir
        elif not srclistdir and isinstance(fitsdir, list):
            srclistdir = os.path.dirname(fits)

        srclistpath = os.path.join(
            srclistdir,
            re.sub(sv.FITS_TAIL, srclistext, os.path.basename(fits))
            )

        # find the fiphot file
        if not photdir and not isinstance(fitsdir, list):
            photdir = fitsdir
        elif not photdir and isinstance(fitsdir, list):
            photdir = os.path.dirname(fits)

        photpath = os.path.join(
            photdir,
            re.sub(sv.FITS_TAIL, photext, os.path.basename(fits))
            )

        # check if they both exist and add to list of good fits
        if os.path.exists(photpath) and os.path.exists(srclistpath):
            goodframes.append(fits)
            goodphots.append(photpath)
            goodsrclists.append(srclistpath)


    # we only work on goodframes now
    print('%sZ: selecting an astrometric reference frame '
          'using %s frames, %s fiphots, %s fistars...' %
          (datetime.utcnow().isoformat(),
           len(goodframes), len(goodphots), len(goodsrclists)))


    # manual override of astrometric reference frame selection. good if you
    # have reason to skip the long serial process below.
    if overrideref:

        selectedreference = overrideref
        selectsrc = re.sub(sv.FITS_TAIL, '.fistar', selectedreference)
        selectphot = re.sub(sv.FITS_TAIL, '.fiphot', selectedreference)

        print('%sZ: manually chosen astrometric reference frame is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        if jpeg:
            framejpg = fits_to_full_jpeg(
                selectedreference,
                out_fname=os.path.join(
                    os.path.dirname(selectedreference),
                    ('JPEG-ASTROMREF-%s.jpg' %
                     re.sub(sv.FITS_TAIL, '',
                            os.path.basename(selectedreference)))
                    )
                )
        else:
            framejpg = None

        photdata_f = read_fiphot(selectedphot)
        if photdata_f:
            photdata = {
                'mag':np.array(photdata_f['per aperture'][2]['mag']),
                'err':np.array(photdata_f['per aperture'][2]['mag err']),
                'flag':np.array(
                    photdata_f['per aperture'][2]['status flag']
                    )
                }
            del photdata_f
        else:
            raise AssertionError('manually chosen astr ref has bad photometry')

        # number of good detections
        goodind = np.where(photdata['flag'] == 0)
        ndet = len(photdata['mag'][goodind])

        srcdata = np.genfromtxt(selectedsrc, usecols=(3,5,6), dtype='f8,f8,f8',
                                names=['background', 'svalue', 'dvalue'])


        return {
            'astromref':selectedreference,
            'framejpg':framejpg,
            'sval':np.nanmedian(srcdata['svalue']),
            'dval':np.nanmedian(srcdata['dvalue']),
            'bgv':np.nanmedian(srcdata['background']),
            'ndet':ndet,
            'comment':'astromref chosen manually'
        }

    median_sval = []
    median_dval = []
    median_background = []
    good_detections = []

    # go through all the frames and find their properties
    for frame, phot, srclist in zip(goodframes, goodphots, goodsrclists):

        if DEBUG:
            print('working on frame %s' % frame)

        # decide if the phot file is binary or not. read the first 600
        # bytes and look for the '--binary-output' text
        with open(phot,'rb') as photf:
            header = photf.read(1000)

        if '--binary-output' in header and HAVEBINPHOT:

            photdata_f = read_fiphot(phot)
            if photdata_f:
                photdata = {
                    'mag':np.array(photdata_f['per aperture'][2]['mag']),
                    'err':np.array(photdata_f['per aperture'][2]['mag err']),
                    'flag':np.array(
                        photdata_f['per aperture'][2]['status flag']
                        )
                    }
                del photdata_f
            else:
                print('no photdata in %s, skipping...' % phot)
                continue

        elif '--binary-output' in header and not HAVEBINPHOT:

            print('%sZ: %s is a binary phot file, '
                  'but no binary phot reader is present, skipping...' %
                  (datetime.utcnow().isoformat(), phot))
            continue

        else:

            # read in the phot file
            photdata = np.genfromtxt(
                phot,
                usecols=(12,13,14),
                dtype='f8,f8,U5',
                names=['mag','err','flag']
                )

        # now, get the data from the associated fistar file
        srcdata = np.genfromtxt(srclist,
                                usecols=(3,5,6),
                                dtype='f8,f8,f8',
                                names=['background',
                                       'svalue',
                                       'dvalue'])

        # find good frames
        if '--binary-output' in header:
            goodind = np.where(photdata['flag'] == 0)
        else:
            goodind = np.where(photdata['flag'] == 'G')
        # number of good detections
        good_detections.append(len(photdata['mag'][goodind]))

        # median background, d, and s
        median_background.append(np.nanmedian(srcdata['background']))
        median_dval.append(np.nanmedian(srcdata['dvalue']))
        median_sval.append(np.nanmedian(srcdata['svalue']))

    #
    # now find the best astrometric reference frame
    #

    # to np.arrays first
    median_sval = np.array(median_sval)
    median_dval = np.array(median_dval)
    median_background = np.array(median_background)
    good_detections = np.array(good_detections)

    # get the best S --> largest S at the top
    median_sval_ind = np.argsort(median_sval)[::-1]

    # here, we want the values closest to zero to be at the top
    median_dval_ind = np.argsort(np.fabs(median_dval))

    # want the smallest background
    median_background_ind = np.argsort(median_background)

    # and the most number of detections
    good_detections_ind = np.argsort(good_detections)[::-1]

    # get the top 200 of each index
    median_sval_ind = median_sval_ind[:500]
    median_dval_ind = median_dval_ind[:500]
    median_background_ind = median_background_ind[:500]
    good_detections_ind = good_detections_ind[:500]

    # now intersect all of these arrays to find the best candidates for the
    # astrometric reference frame

    sd_ind =  np.intersect1d(median_sval_ind,
                             median_dval_ind,
                             assume_unique=True)


    best_frame_ind = np.intersect1d(
        sd_ind,
        np.intersect1d(median_background_ind,
                       good_detections_ind,
                       assume_unique=True),
        assume_unique=True
        )

    sdndet_ind = np.intersect1d(sd_ind,
                                good_detections_ind,
                                assume_unique=True)

    # pick a good astrometric reference frame
    goodframes = np.array(goodframes)


    # if all selectors produced a result, use that one
    if len(best_frame_ind) > 0:

        selectedreference = goodframes[best_frame_ind[0]]

        print('%sZ: selected best astrometric reference frame is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        if jpeg:
            framejpg = fits_to_full_jpeg(
                selectedreference,
                out_fname=os.path.join(
                    os.path.dirname(selectedreference),
                    ('JPEG-ASTROMREF-%s.jpg' %
                     os.path.basename(re.sub(sv.FITS_TAIL, '',
                                             selectedreference)))
                    )
                )
        else:
            framejpg = None

        return {
            'astromref':selectedreference,
            'framejpg':framejpg,
            'sval':median_sval[best_frame_ind[0]],
            'dval':median_dval[best_frame_ind[0]],
            'bgv':median_background[best_frame_ind[0]],
            'ndet':good_detections[best_frame_ind[0]],
            'comment':'astromref chosen using sval, dval, bgval, ndet'
        }


    # otherwise, fall back to to the frames with the best values of S, D and
    # a large number of detections
    elif len(sdndet_ind) > 0:

        selectedreference = goodframes[sdndet_ind[0]]

        print('WRN! %sZ: selected best astrometric reference frame '
              '(using S, D, and ndet only) is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        if jpeg:
            framejpg = fits_to_full_jpeg(
                selectedreference,
                out_fname=os.path.join(
                    os.path.dirname(selectedreference),
                    ('JPEG-ASTROMREF-%s.jpg' %
                     os.path.basename(re.sub(sv.FITS_TAIL, '',
                                             selectedreference)))
                    )
                )
        else:
            framejpg = None

        return {
            'astromref':selectedreference,
            'framejpg':framejpg,
            'sval':median_sval[sdndet_ind[0]],
            'dval':median_dval[sdndet_ind[0]],
            'bgv':median_background[sdndet_ind[0]],
            'ndet':good_detections[sdndet_ind[0]],
            'comment':'astromref chosen using sval, dval, ndet'
        }


    # otherwise, fall back to to the frames with the best values of S and D
    elif len(sd_ind) > 0:

        selectedreference = goodframes[sd_ind[0]]

        print('WRN! %sZ: selected best astrometric reference frame '
              '(using S and D only) is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        if jpeg:
            framejpg = fits_to_full_jpeg(
                selectedreference,
                out_fname=os.path.join(
                    os.path.dirname(selectedreference),
                    ('JPEG-ASTROMREF-%s.jpg' % os.path.basename(
                        re.sub(sv.FITS_TAIL, '', selectedreference)))
                    )
                )

        else:
            framejpg = None

        return {
            'astromref':selectedreference,
            'framejpg':framejpg,
            'sval':median_sval[sd_ind[0]],
            'dval':median_dval[sd_ind[0]],
            'bgv':median_background[sd_ind[0]],
            'ndet':good_detections[sd_ind[0]],
            'comment':'astromref chosen using sval, dval'
        }


    # if that fails, fail back to the best S value frame
    elif len(median_sval_ind) > 0:

        selectedreference = goodframes[median_sval_ind[0]]

        print('WRN! %sZ: selected best astrometric reference frame '
              '(using S only) is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        if jpeg:
            framejpg = fits_to_full_jpeg(
                selectedreference,
                out_fname=os.path.join(
                    os.path.dirname(selectedreference),
                    ('JPEG-ASTROMREF-%s.jpg' %
                     os.path.basename(re.sub(sv.FITS_TAIL, '',
                                             selectedreference)))
                    )
                )
        else:
            framejpg = None

        return {
            'astromref':selectedreference,
            'framejpg':framejpg,
            'sval':median_sval[median_sval_ind[0]],
            'dval':median_dval[median_sval_ind[0]],
            'bgv':median_background[median_sval_ind[0]],
            'ndet':good_detections[median_sval_ind[0]],
            'comment':'astromref chosen using sval'
        }


    else:

        print('ERR! %sZ: could not select a good astrometric reference frame!' %
              (datetime.utcnow().isoformat(), ))

        return



def xysdk_coeffs_worker(task):
    """
    This is a parallel worker to run the xysdk coeff operation.
    """

    fistar, fistarglob = task

    outxysdk = fistar.replace(fistarglob.split('.')[-1], 'xysdk')

    cmdtorun = XYSDKCMD.format(fistarfile=fistar,
                               xysdkfile=outxysdk)

    returncode = os.system(cmdtorun)

    if returncode == 0:
        print('%sZ: XYSDK coeffs OK: %s -> %s' %
              (datetime.utcnow().isoformat(), fistar, outxysdk))
        return fistar, outxysdk
    else:
        print('ERR! %sZ: XYSDK coeffs failed for %s' %
              (datetime.utcnow().isoformat(), fistar))
        if os.path.exists(outxysdk):
            os.remove(outxysdk)
        return fistar, None


def get_smoothed_xysdk_coeffs(fistardir,
                              fistarglob='*.fistar',
                              nworkers=16,
                              maxworkertasks=1000,
                              overwrite=False):
    """
    This generates smoothed xy and sdk coefficents for use with iphot later
    (these go into the output photometry file or something).

    grtrans ${APPHOT}/$base.${EXT_FISTAR} --col-xy 2,3 --col-fit 6,7,8 \
                  --col-weight 10 --order 4 \
                  --iterations 3 --rejection-level 3 \
                  --comment --output-transformation ${IPHOT}/$base.xysdk
    """

    fistarlist = glob.glob(os.path.join(os.path.abspath(fistardir), fistarglob))

    if not overwrite:
        # if all the files exist, don't bother replicating them.
        xysdk_exists = np.array(
            [os.path.exists(f.replace('.fistar','.xysdk')) for f in fistarlist]
        )
        if np.all(xysdk_exists):
            print('%sZ: found all xysdk files already existed. continue.' %
                  (datetime.utcnow().isoformat()))
            return None

    print('%sZ: %s files to process in %s' %
          (datetime.utcnow().isoformat(), len(fistarlist), fistardir))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, fistarglob) for x in fistarlist]

    # fire up the pool of workers
    results = pool.map(xysdk_coeffs_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



def astromref_shift_worker(task):
    """
    This is a parallel worker for getting the shifts between the astromref frame
    and the frame in the task definition.

    task[0] = target frame fistar
    task[1] = astromref frame fistar
    task[2] = outdir
    """

    fistartoshift, astromref, outdir = task

    if outdir:
        outfile = os.path.join(
            os.path.abspath(outdir),
            os.path.basename(fistartoshift).replace('fistar','itrans')
            )

    else:
        outfile = fistartoshift.replace('fistar','itrans')

    cmdtorun = FRAMESHIFTCALCCMD.format(astromref=astromref,
                                        fistartoshift=fistartoshift,
                                        outtransfile=outfile)

    if DEBUG:
        print(cmdtorun)

    returncode = os.system(cmdtorun)

    if returncode == 0:
        print('%sZ: shift transform calc OK: %s -> %s' %
              (datetime.utcnow().isoformat(), fistartoshift, outfile))
        return fistartoshift, outfile
    else:
        print('ERR! %sZ: shift transform calc failed for %s' %
              (datetime.utcnow().isoformat(), fistartoshift))
        if os.path.exists(outfile):
            os.remove(outfile)
        return fistartoshift, None



def get_astromref_shifts(fistardir,
                         astromreffistar,
                         fistarglob='*.fistar',
                         outdir=None,
                         nworkers=16,
                         maxworkertasks=1000):
    """
    This gets shifts between the astrometric reference frame and all other
    frames.
    """

    if isinstance(fistardir, list):
        fistarlist = fistardir
    else:
        fistarlist = glob.glob(os.path.join(os.path.abspath(fistardir),
                                            fistarglob))

    print('%sZ: %s files to process' %
          (datetime.utcnow().isoformat(), len(fistarlist)))

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    tasks = [(x, astromreffistar, outdir) for x in fistarlist]

    # fire up the pool of workers
    results = pool.map(astromref_shift_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



def frame_to_astromref_worker(task):
    """
    This is a parallel worker for the frame shift to astromref frame operation.

    task[0] = FITS frame to shift
    task[1] = directory where transform files are
    task[2] = output directory
    """

    frametoshift, transdir, outdir = task

    # figure out the transfile path
    if transdir:
        itransfile = os.path.join(
            os.path.join(
                os.path.abspath(transpath),
                os.path.basename(frametoshift).replace('.fits','.itrans')
            )
        )

    else:
        itransfile = frametoshift.replace('.fits','.itrans')

    # make sure the itransfile for this frame exists before we proceed
    if not os.path.exists(itransfile):
        print('ERR! %sZ: frame transform to astromref failed for %s, '
              'no itrans file found' %
              (datetime.utcnow().isoformat(), frametoshift))
        return frametoshift, None

    # figure out the output path
    if outdir:
        outtransframe = os.path.join(
            os.path.abspath(outdir),
            os.path.basename(frametoshift).replace('.fits','-xtrns.fits')
            )

    else:
        outtransframe = frametoshift.replace('.fits','-xtrns.fits')

    cmdtorun = FRAMETRANSFORMCMD.format(itransfile=itransfile,
                                        frametoshift=frametoshift,
                                        outtransframe=outtransframe)

    returncode = os.system(cmdtorun)

    if returncode == 0:
        print('%sZ: transform to astromref OK: %s -> %s' %
              (datetime.utcnow().isoformat(), frametoshift, outtransframe))


        framejpg = fits_to_full_jpeg(
            outtransframe,
            out_fname=os.path.join(
                os.path.dirname(outtransframe),
                ('JPEG-XTRNS-%s.jpg' %
                 os.path.basename(re.sub(sv.FITS_TAIL, '', outtransframe)))
                )
            )

        return frametoshift, outtransframe
    else:
        print('ERR! %sZ: transform to astromref failed for %s' %
              (datetime.utcnow().isoformat(), frametoshift))
        if os.path.exists(outtransframe):
            os.remove(outtransframe)
        return frametoshift, None



def transform_frames_to_astromref(fitsdir,
                                  fitsglob='*.fits',
                                  itransdir=None,
                                  outdir=None,
                                  nworkers=16,
                                  maxworkertasks=1000):
    """
    This shifts all frames to the astrometric reference.
    """

    if isinstance(fitsdir, list):
        fitslist = fitsdir
    else:
        fitslist = glob.glob(os.path.join(os.path.abspath(fitsdir), fitsglob))

    print('%sZ: %s files to process' %
          (datetime.utcnow().isoformat(), len(fitslist)))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, itransdir, outdir) for x in fitslist]

    # fire up the pool of workers
    results = pool.map(frame_to_astromref_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}


##################################
## PHOTOMETRIC REFERENCE FRAMES ##
##################################

def select_photref_frames(fitsdir,
                          fitsglob='*-xtrns.fits',
                          photdir=None,
                          photext='.fiphot',
                          srclistdir=None,
                          srclistext='.fistar',
                          minframes=50,
                          maxhourangle=3.0,
                          maxmoonphase=25.0,
                          maxmoonelev=0.0,
                          maxzenithdist=30.0,
                          maxbackgroundstdev=10.0,
                          maxbackgroundmedian=1000.0,
                          forcecollectinfo=False):
    """
    This selects a group of photometric reference frames that will later be
    stacked and medianed to form the single photometric reference frame.

    0. this is run on the transformed frames (the ones with -xtrns.fits)

    1. returns a list that is at least minframes long of frames suitable for
    combining into a median reference frame, using the following list of
    selectors (on the original versions (?; probably)):

    - best median scatter of photometry
    - lowest median error in photometry
    - lowest median background measurement
    - lowest MAD in the background measurements
    - low zenith distance
    - high moon distance and lowest moon elevation
    - hour angle with +/- 3 hours
    - large number of stars detected with good flags

    for all selected frames, we will get the median of the background values
    near the center 512x512 pixels. then, we'll enforce that the median of
    background values of each frames be within some delta of the overall
    median. this is basically a slacker way to get rid of cloudy nights.

    2. we convolve all of these to the the best photometric reference frame's
    PSF (they're already in the same coordinate frame since everything was
    translated to the astromref's coordinate frame).

    3. now that all the frames are in the same coordinate system, and have been
    convolved to the same PSF, we can median-stack them (using scipy or
    ficombine)
    """

    if isinstance(fitslist, list):
        fitslist = fitsdir
    else:
        fitslist = glob.glob(os.path.join(fitsdir, fitsglob))



    print('%sZ: %s FITS files found in %s matching glob %s, '
          'finding photometry and source lists...' %
          (datetime.utcnow().isoformat(),
           len(fitslist), fitsdir, fitsglob))

    goodframes = []
    goodphots = []
    goodsrclists = []

    # associate the frames with their fiphot files
    for fits in fitslist:

        fitsbase = os.path.splitext(os.path.basename(fits))[0]

        # find the fistar file
        if not srclistdir and not isinstance(fitsdir, list):
            srclistdir = fitsdir
        elif not srclistdir and isinstance(fitsdir, list):
            srclistdir = os.path.dirname(fits)

        # find the fiphot file
        if not photdir and not isinstance(fitsdir, list):
            photdir = fitsdir
        elif not photdir and isinstance(fitsdir, list):
            photdir = os.path.dirname(fits)

        # if the xtrns files are passed in, make sure we look at the
        # right fistar and fiphot files
        if '-xtrns' in fitsbase:
            fitsbase = fitsbase.rstrip('-xtrns')

        photpath = os.path.join(
            photdir,
            fitsbase + photext
            )
        srclistpath = os.path.join(
            srclistdir,
            fitsbase + srclistext
            )

        if os.path.exists(photpath) and os.path.exists(srclistpath):
            goodframes.append(fits)
            goodphots.append(photpath)
            goodsrclists.append(srclistpath)

    # we only work on goodframes now
    print('%sZ: %s good frames found in %s, '
          'now selecting photometric reference frames...' %
          (datetime.utcnow().isoformat(), len(goodframes), fitsdir))

    # things we need to worry about
    # best median scatter of photometry
    # lowest median error in photometry
    # lowest median background measurement
    # low zenith distance
    # high moon and sun distance
    # large number of stars detected

    outpostfix = (
        fitsglob.replace('*','').replace('?','').replace('.fits','')
        )
    pklfile = os.path.join(os.path.abspath(fitsdir),
                           'TM-imagesub-photref-%s.pkl' % outpostfix)

    if (not os.path.exists(pklfile) or forcecollectinfo):

        # from the FITS
        (zenithdist, moondist, moonelev,
         moonphase, hourangle) = [], [], [], [], []

        # from the fiphot and fistar files
        (ngoodobjects, medmagerr, magerrmad,
         medsrcbg, stdsrcbg,
         medsvalue, meddvalue) = [], [], [], [], [], [], []

        for frame, phot, srclist in zip(goodframes, goodphots, goodsrclists):

            if DEBUG:
                print('working on frame %s' % frame)

            # 1. get the data from FITS header
            headerdata = imageutils.get_header_keyword_list(
                frame,
                ['Z','MOONDIST','MOONELEV','MOONPH','HA']
                )

            # 2. get the data from the fiphot file

            # decide if the phot file is binary or not. read the first 600
            # bytes and look for the '--binary-output' text
            with open(phot,'rb') as photf:
                header = photf.read(1000)

            if '--binary-output' in header and HAVEBINPHOT:

                photdata_f = read_fiphot(phot)
                photdata = {
                    'mag':np.array(photdata_f['per aperture'][2]['mag']),
                    'err':np.array(photdata_f['per aperture'][2]['mag err']),
                    'flag':np.array(
                        photdata_f['per aperture'][2]['status flag']
                        )
                    }
                del photdata_f

            elif '--binary-output' in header and not HAVEBINPHOT:

                print('WRN! %sZ: %s is a binary phot file, '
                      'but no binary phot reader is present, skipping...' %
                      (datetime.utcnow().isoformat(), phot))
                continue

            else:

                # read in the phot file
                photdata = np.genfromtxt(
                    phot,
                    usecols=(12,13,14),
                    dtype='f8,f8,U5',
                    names=['mag','err','flag']
                    )

            # 3. get the data from the fistar file
            srcdata = np.genfromtxt(srclist,
                                    usecols=(3,5,6),
                                    dtype='f8,f8,f8',
                                    names=['background',
                                           'svalue',
                                           'dvalue'])

            # now we have headerdata, photdata, and srcdata, fill in the lists

            # header data
            if 'Z' in headerdata:
                zenithdist.append(headerdata['Z'])
            else:
                zenithdist.append(np.nan)

            if 'MOONDIST' in headerdata:
                moondist.append(headerdata['MOONDIST'])
            else:
                moondist.append(np.nan)

            if 'MOONELEV' in headerdata:
                moonelev.append(headerdata['MOONELEV'])
            else:
                moonelev.append(np.nan)

            if 'MOONPH' in headerdata:
                moonphase.append(headerdata['MOONPH'])
            else:
                moonphase.append(np.nan)

            if 'HA' in headerdata:
                hourangle.append(headerdata['HA'])
            else:
                hourangle.append(np.nan)

            # fiphot data
            if '--binary-output' in header:
                goodind = np.where(photdata['flag'] == 0)
            else:
                goodind = np.where(photdata['flag'] == 'G')

            median_mag = np.median(photdata['mag'][goodind])

            # these are the quantities we're interested in
            ngood = len(goodind[0])
            median_magerr = np.nanmedian(photdata['err'][goodind])
            medabsdev_mag = np.nanmedian(
                np.abs(photdata['mag'][goodind] - median_mag)
                )

            # put these in the lists
            ngoodobjects.append(ngood)
            medmagerr.append(median_magerr)
            magerrmad.append(medabsdev_mag)

            # fistar data -- background
            medsrcbg.append(np.nanmedian(srcdata['background']))
            stdsrcbg.append(np.nanstd(srcdata['background']))

            # fistar data -- PSF
            medsvalue.append(np.nanmedian(srcdata['svalue']))
            meddvalue.append(np.nanmedian(srcdata['dvalue']))

        #
        # done with collecting data, get the best photometric reference frames
        #

        # convert all lists to np.arrays first
        zenithdist = np.array(zenithdist)
        moondist = np.array(moondist)
        moonelev = np.array(moonelev)
        moonphase = np.array(moonphase)
        hourangle = np.array(hourangle)

        ngoodobjects = np.array(ngoodobjects)
        medmagerr = np.array(medmagerr)
        magerrmad = np.array(magerrmad)

        medsrcbg = np.array(medsrcbg)
        stdsrcbg = np.array(stdsrcbg)

        medsvalue = np.array(medsvalue)
        meddvalue = np.array(meddvalue)

        goodframes = np.array(goodframes)

        infodict = {
            'frames':goodframes,
            'zenithdist':zenithdist,
            'moondist':moondist,
            'moonelev':moonelev,
            'moonphase':moonphase,
            'hourangle':hourangle,
            'ngoodobjs':ngoodobjects,
            'medmagerr':medmagerr,
            'magerrmad':magerrmad,
            'medsrcbkg':medsrcbg,
            'stdsrcbkg':stdsrcbg,
            'medsvalue':medsvalue,
            'meddvalue':meddvalue
        }

        # write this info dict to a file so we can quickly load it later
        outpf = open(pklfile,'wb')
        pickle.dump(infodict, outpf, pickle.HIGHEST_PROTOCOL)
        outpf.close()

    # if the imagesub photref info file exists already, load it up
    else:

        outpostfix = (
            fitsglob.replace('*','').replace('?','').replace('.fits','')
            )
        inpf = open(pklfile,'rb')

        print('%sZ: loading existing photref select info from %s' %
              (datetime.utcnow().isoformat(), pklfile))

        infodict = pickle.load(inpf)
        inpf.close()

    #
    # now do the filtering
    #

    # filter on hour angle
    haind = np.fabs(infodict['hourangle']) < maxhourangle

    # get dark nights
    moonind = ((np.fabs(infodict['moonphase']) < maxmoonphase) |
               (infodict['moonelev'] < maxmoonelev))

    # get low zenith distance nights
    zenithind = infodict['zenithdist'] < maxzenithdist

    # get nights with background stdev < max_bgv_stdev (to possibly remove
    # cloudy nights)
    backgroundstdevind = infodict['stdsrcbkg'] < maxbackgroundstdev

    # get nights with background median < maxbackgroundmedian (to possibly
    # remove cloudy nights)
    backgroundmedind = infodict['medsrcbkg'] < maxbackgroundmedian

    # this is the final operating set of frames that will be sorted for the
    # following tests
    selectind = haind & moonind & zenithind & backgroundstdevind

    selected_frames = infodict['frames'][selectind]
    selected_ngoodobj = infodict['ngoodobjs'][selectind]

    selected_medmagerr = infodict['medmagerr'][selectind]
    selected_magerrmad = infodict['magerrmad'][selectind]

    selected_medsrcbkg = infodict['medsrcbkg'][selectind]
    selected_stdsrcbkg = infodict['stdsrcbkg'][selectind]

    selected_medsvalue = infodict['medsvalue'][selectind]
    selected_meddvalue = infodict['meddvalue'][selectind]

    print('%sZ: selected %s frames with acceptable '
          'HA, Z, moon phase, and elevation for further filtering...' %
          (datetime.utcnow().isoformat(), len(selected_frames)))

    # we select in the following order
    # 1. D closest to 0
    # 2. largest S

    # then we get filter out any images that have background >
    # maxbackgroundmedian and backgroundstdev > maxbackgroundstdev

    # first sort selector
    stage1_sort_ind = (np.argsort(selected_medsvalue))[::-1]

    stage1_frames = selected_frames[stage1_sort_ind[:2*minframes]]
    stage1_median_bgv = selected_medsrcbkg[stage1_sort_ind[:2*minframes]]
    stage1_stdev_bgv = selected_stdsrcbkg[stage1_sort_ind[:2*minframes]]
    stage1_svalue = selected_medsvalue[stage1_sort_ind[:2*minframes]]
    stage1_dvalue = selected_meddvalue[stage1_sort_ind[:2*minframes]]

    # next, sort by roundest stars
    stage2_sort_ind = (np.argsort(np.fabs(stage1_dvalue)))

    stage2_frames = stage1_frames[stage2_sort_ind]
    stage2_median_bgv = stage1_median_bgv[stage2_sort_ind]
    stage2_stdev_bgv = stage1_stdev_bgv[stage2_sort_ind]
    stage2_svalue = stage1_svalue[stage2_sort_ind]
    stage2_dvalue = stage1_dvalue[stage2_sort_ind]

    final_bgvmed_ind = stage2_median_bgv < maxbackgroundmedian
    final_bgvstd_ind = stage2_stdev_bgv < maxbackgroundstdev
    final_selector_ind = final_bgvmed_ind & final_bgvstd_ind

    final_frames = stage2_frames[final_selector_ind][:minframes]
    final_median_bgv = stage2_median_bgv[final_selector_ind][:minframes]
    final_stdev_bgv = stage2_stdev_bgv[final_selector_ind][:minframes]
    final_svalues = stage2_svalue[final_selector_ind][:minframes]
    final_dvalues = stage2_dvalue[final_selector_ind][:minframes]

    print('%sZ: selected %s final frames as photref' %
          (datetime.utcnow().isoformat(), len(final_frames)))

    # the master photref is the frame we'll convolve all of the rest of the
    # photrefs to. it's the softest of these frames
    candidate_master_photref = final_frames[np.nanargmin(final_svalues)]

    # make JPEGs of the selected photref frames
    for final_frame in final_frames:

        framejpg = fits_to_full_jpeg(
            final_frame,
            out_fname=os.path.join(
                os.path.dirname(final_frame),
                ('JPEG-PHOTREF-%s.jpg' %
                 os.path.basename(re.sub(sv.FITS_TAIL, '', final_frame)))
                )
            )

    return candidate_master_photref, final_frames.tolist(), infodict



def generate_masterphotref_registration_info(masterphotref_fistar,
                                             outfile,
                                             xycols=(1,2),
                                             fluxcol=8):
    """
    This generates a registration information file using the master
    photometric reference frame. This file is then used by the convolution step
    somehow to figure out the convolution kernel? In any case, it's needed for:

    - generating convolved reference frames to be ultimately stacked into a
      single photometric reference frame

    - do the convolution of the reference frame to each -xtrns target frame when
      doing the image subtraction
    """

    # get the x and y coordinate columns from the source list (fistar)
    srcinfo = np.genfromtxt(masterphotref_fistar,
                            usecols=tuple(list(xycols) + [fluxcol]),
                            dtype='f8,f8,f8',
                            names=['x','y','flux'])

    refx, refy, refflux = srcinfo['x'], srcinfo['y'], srcinfo['flux']

    # set up the grid (this weirdness is transcribed directly from Chelsea's
    # regslct.py) TODO: figure out WTF this does

    BX = 30.; BY = 30.

    mx = np.zeros(BX*BY)-1
    my = np.zeros(BX*BY)-1
    fluxblock = np.zeros(BX*BY)

    xsize = 2048.
    ysize = 2048.

    bx = (refx*BX/xsize).astype(int)
    by = (refy*BY/ysize).astype(int)
    idx = by*BX+bx

    for i, j in zip(idx, refflux):
        if fluxblock[i] < j:
            mx[i] = refx[i]
            my[i] = refy[i]
            fluxblock[i] = j

    np.savetxt(outfile, np.transpose((mx, my, np.ones(BX*BY)*20)))


def genreg(masterphotref_fistar,
           outfile,
           xycols=(1,2)):
    """
    This generates a registration information file using the master
    photometric reference frame. This file is then used by the convolution step
    somehow to figure out the convolution kernel? In any case, it's needed for:

    - generating convolved reference frames to be ultimately stacked into a
      single photometric reference frame

    - do the convolution of the reference frame to each -xtrns target frame when
      doing the image subtraction

    NOTE: masterphotref_fistar might also be a master *astrometric* reference.
    """

    # get the x and y coordinate columns from the source list (fistar)
    srcxy = np.genfromtxt(masterphotref_fistar,
                          usecols=xycols,
                          dtype='f8,f8',
                          names=['x','y'])

    # set up the grid (this weirdness is transcribed directly from Chelsea's
    # regslct.py) TODO: figure out WTF this does

    BX = 30; BY = 30
    mx = np.zeros(BX*BY)-1
    my = np.zeros(BX*BY)-1
    ma = np.zeros(BX*BY)
    xsize = 2048.
    ysize = 2048.
    bx = (srcxy['x']*BX/xsize).astype(int)
    by = (srcxy['y']*BY/ysize).astype(int)
    mx[by*bx+bx] = srcxy['x']
    my[by*bx+bx] = srcxy['y']

    outf = open(outfile,'wb')

    for i in range(int(BX*BY)):
        outf.write(("%8.0f %8.0f %8.0f\n" % (mx[i],my[i],20)).encode("utf-8"))

    outf.close()



def photref_convolution_worker(task):
    """
    This is a parallel worker to convolve the photref frames to the chosen
    master photref frame. Used by convolve_photref_frames below.

    task[0] -> the frame to convolve
    task[1] -> the frame to use as the convolution target
    task[2] -> the convolution target's registration info file
    task[3] -> the kernel specification for the convolution
    task[4] -> the output directory where to place the results
    """

    frametoconvolve, targetframe, convregfile, kernelspec, outdir = task

    try:

        if not outdir:
            outfile = os.path.join(os.path.dirname(frametoconvolve),
                                   'PHOTREF-%s' %
                                   os.path.basename(frametoconvolve))

        else:
            outfile = os.path.join(outdir,
                                   'PHOTREF-%s' %
                                   os.path.basename(frametoconvolve))

        cmdtorun = PHOTREFCONVOLVECMD.format(
            targetframe=targetframe,
            frametoconvolve=frametoconvolve,
            convregfile=convregfile,
            kernelspec=kernelspec,
            outputfile=outfile
        )

        if DEBUG:
            print(cmdtorun)

        returncode = os.system(cmdtorun)

        if returncode == 0:
            print('%sZ: photref convolution OK: %s -> %s' %
                  (datetime.utcnow().isoformat(), frametoconvolve, outfile))

            framejpg = fits_to_full_jpeg(
                outfile,
                out_fname=os.path.join(
                    os.path.dirname(outfile),
                    ('JPEG-CONVPHOTREF-%s.jpg' %
                     os.path.basename(re.sub(sv.FITS_TAIL, '', outfile)))
                    )
                )

            return frametoconvolve, outfile
        else:
            print('ERR! %sZ: photref convolution failed for %s' %
                  (datetime.utcnow().isoformat(), frametoconvolve,))
            if os.path.exists(outfile):
                os.remove(outfile)
            return frametoconvolve, None

    except Exception as e:

        print('ERR! %sZ: photref convolution failed for %s, exception %s' %
              (datetime.utcnow().isoformat(), frametoconvolve, e))
        return frametoconvolve, None


def convolve_photref_frames(photreflist,
                            photrefframe,
                            convregfile,
                            kernelspec='b/4;i/4;d=4/4',
                            nworkers=16,
                            maxworkertasks=1000,
                            outdir=None):

    """
    This convolves all photref frames in photreflist to the photrefframe. See
    getref() in run_ficonv.py.
    """

    # make a list of tasks

    tasks = [(x, photrefframe, convregfile, kernelspec, outdir)
             for x in photreflist]

    print('%sZ: %s photref files to convolve to %s' %
          (datetime.utcnow().isoformat(), len(photreflist), photrefframe))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(photref_convolution_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



def combine_frames(framelist,
                   outfile,
                   combinemethod='median'):
    """
    This combines all of the frames in framelist (a list of filenames) using
    ficombine and the specified combinemethod. combinemethod is one of the
    following strings (taken from ficombine's help):

     average, mean      The mean value of the pixels.
     median             The median value of the pixels.
     rejmed, rejection  Median value after rejecting outliers.
     rejavg             Mean value after rejecting outliers.
     sum                Sum of the pixel values.
     squaresum          Sum for the squarers of the pixel values.
     scatter, stddev    Pixel scatter (standard deviation).
     or                 Use logical OR combination between masks.
     and                Use logical AND combination between masks.
     ignorenegative     Ignore (i.e. mask) pixels with negative values.
     ignorezero         Ignore (i.e. mask) pixels with a zero value.
     ignorenegative     Ignore (i.e. mask) pixels with positive values.
    """

    combineflist = ' '.join(framelist)

    cmdtorun = FRAMECOMBINECMD.format(
        framelist=combineflist,
        combinemethod=combinemethod,
        outfile=outfile
    )

    if DEBUG:
        print(cmdtorun)

    returncode = os.system(cmdtorun)

    if returncode == 0:
        print('%sZ: framelist combine OK: %s images in framelist -> %s' %
              (datetime.utcnow().isoformat(), len(framelist), outfile))

        framejpg = fits_to_full_jpeg(
            outfile,
            out_fname=os.path.join(
                os.path.dirname(outfile),
                ('JPEG-COMBINED-%s.jpg' %
                 os.path.basename(re.sub(sv.FITS_TAIL, '', outfile)))
                )
            )

        return framelist, outfile
    else:
        print('ERR! %sZ: framelist combine failed!' %
              (datetime.utcnow().isoformat(),))
        if os.path.exists(outfile):
            os.remove(outfile)
        return framelist, None


def catmag_to_flux(catmag, catmag_0, flux_0):
    return flux_0 * 10** ( -0.4*(catmag - catmag_0) )


def photometry_on_combined_photref(
        photref_frame,
        fovcatalog,
        ccdnumber,
        ra_nom,
        dec_nom,
        ccdgain=None,
        zeropoint=None,
        ccdexptime=None,
        extractsources=True,
        apertures='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
        outfile=None,
        framewidth=None,
        searchradius=8.0,
        photreffluxthreshold=50000,
        observatory='hatpi',
        useimagenotfistar=False,
        astrometrydownsample=2,
        pixelerror=0.3,
        uniformize=10,
        reformed_cat_file=None,
        projid=None,
        raise_wcs_error=False
        ):
    """
    This does source extraction, WCS, catalog projection, and then runs fiphot
    in the special iphot mode on the combined photometric reference frame. See
    cmrawphot.sh for the correct commandline to use.

    Chelsea's apertures='2.95:7.0:6.0,3.35:7.0:6.0,3.95:7.0:6.0'
    """

    # get the required header keywords from the FITS file
    if observatory=='hatpi':
        headerlist = ['GAIN', 'GAIN1', 'GAIN2', 'EXPTIME', 'RAC', 'DECC',
                      'FOV']
    elif observatory=='tess':
        headerlist = ['GAINA', 'TELAPSE', 'CRVAL1', 'CRVAL2']

    header = imageutils.get_header_keyword_list(photref_frame, headerlist)

    # get the RA and DEC from the frame header for astrometry
    if 'RAC' in header and 'DECC' in header:
        frame_ra = header['RAC']*360.0/24.0
        frame_dec = header['DECC']
    elif 'CRVAL1' in header and 'CRVAL2' in header:
        frame_ra = header['CRVAL1']
        frame_dec = header['CRVAL2']
    else:
        print('ERR! %sZ: no RAC or DECC defined for %s' %
              (datetime.utcnow().isoformat(),
               photref_frame))
        return None

    # figure out the width of the frame for astrometry
    if 'FOV' in header:
        framewidth = header['FOV']
    elif 'FOV' not in header and observatory=='tess':
        framewidth = 24
    elif framewidth is None:
        print('ERR! %sZ: no frame width defined for %s, '
              'astrometry not possible' %
              (datetime.utcnow().isoformat(),
               photref_frame))
        return None

    # handle the gain and exptime parameters
    if not ccdgain:

        if 'GAIN1' in header and 'GAIN2' in header:
            ccdgain = (header['GAIN1'] + header['GAIN2'])/2.0
        elif 'GAIN' in header:
            ccdgain = header['GAIN']
        elif 'GAINA' in header:
            ccdgain = header['GAINA']
        else:
            ccdgain = None

    if not ccdexptime:
        ccdexptime = header['EXPTIME'] if 'EXPTIME' in header else None

    if not (ccdgain or ccdexptime):
        print('ERR! %sZ: no GAIN or EXPTIME defined for %s' %
              (datetime.utcnow().isoformat(),
               photref_frame))
        return None

    # handle the zeropoints
    # if the zeropoint isn't provided and if this is a HAT frame, the ccd
    # number will get us the zeropoint in the ZEROPOINTS dictionary
    if zeropoint:
        pass
    elif (not zeropoint) and (ccdnumber in ZEROPOINTS) and (observatory=='hatpi'):
        zeropoint = ZEROPOINTS[ccdnumber]
    else:
        print('ERR! %sZ: no zeropoint magnitude defined for %s' %
              (datetime.utcnow().isoformat(),
               photref_frame))
        return None

    # figure out the output path
    if not outfile:
        outfile = os.path.abspath(re.sub(sv.FITS_TAIL, '.cmrawphot',
                                         photref_frame))

    # FIRST: do source extraction so that we can get a .wcs file for the
    # photometric reference.
    print('%sZ: extracting sources for astrometry from %s...' %
          (datetime.utcnow().isoformat(), photref_frame))
    astromfistarf = os.path.abspath(re.sub(sv.FITS_TAIL, '.fistar-astrometry',
                                           photref_frame))
    print(zeropoint, ccdgain, ccdexptime)
    astromfistar = extract_frame_sources(
        photref_frame,
        astromfistarf,
        fluxthreshold=photreffluxthreshold,
        zeropoint=zeropoint,
        ccdgain=ccdgain,
        exptime=ccdexptime
    )

    # SECOND: add WCS
    print('%sZ: running astrometry solution for %s...' %
          (datetime.utcnow().isoformat(), photref_frame))
    wcsf = os.path.abspath(re.sub(sv.FITS_TAIL, '.wcs', photref_frame))
    if observatory=='hatpi':
        wcsfile = anet_solve_frame(astromfistar,
                                   wcsf,
                                   frame_ra,
                                   frame_dec,
                                   width=framewidth,
                                   radius=searchradius)
    elif observatory=='tess':

        #
        # for TESS, default is to use astrometry from SPOC headers, and then to
        # check that it satisfies precision requirements. the WCS for the
        # photref should always be same as that for the astromref.
        #
        astromref_wcs = glob.glob(
            os.path.join(
                os.path.dirname(photref_frame),
                'proj{}-*astromref*.wcs'.format(projid)
            )
        )
        if len(astromref_wcs) != 1:
            raise AssertionError('expected 1 astromref wcs')
        astromref_wcs = astromref_wcs[0]

        shutil.copy(astromref_wcs, photref_frame.replace('.fits','.wcs'))
        print('did hard copy of {} -> {}'.format(
            astromref_wcs, photref_frame.replace('.fits','.wcs')))

        import wcsqualityassurance as wcsqa
        from astropy.coordinates import SkyCoord

        # allow worse wcs outliers if frame center within 10 deg of
        # galactic plane
        framecoord = SkyCoord(
            ra=ra_nom, dec=dec_nom, unit=(units.degree, units.degree),
            frame='icrs'
        )
        frame_galacticlatitude = framecoord.galactic.b.value
        if abs(frame_galacticlatitude) < 20:
            qualitycondition = {'median_px':0.2,
                                '90th_px':0.7,
                                'std_px':2.3 }
        else:
            qualitycondition = {'median_px':0.2,
                                '90th_px':0.5,
                                'std_px': 1.7}

        if wcsqa.does_wcs_pass_quality_check(
            wcsf, photref_frame, reformed_cat_file, isspocwcs=True, N_bright=1000,
            N_faint=9000, fistarpath=astromfistar, matchedoutpath=None,
            qualitycondition=qualitycondition
        ):
            pass

        else:
            errmsg = (
                'PHOTREF WCS fails quality check '+
                'wcs, fits are {}, {}. frame glat is {} deg'.
                format(wcsf, photref_frame, frame_galacticlatitude)
            )
            print(errmsg)
            if raise_wcs_error:
                raise AssertionError(errmsg)

        if False:
            # NOTE: deprecated for TESS
            if not useimagenotfistar:
                fistarfile_to_xy(astromfistar)

            wcsfile = astrometrydotnet_solve_frame(
                astromfistar.replace(os.path.splitext(astromfistar)[-1],'.fistar-fits-xy'),
                wcsf,
                frame_ra,
                frame_dec,
                radius=20,
                scalelow=20,
                scalehigh=22,
                scaleunits='arcsecperpix',
                tweakorder=6,
                useimagenotfistar=useimagenotfistar,
                downsample=astrometrydownsample,
                pixelerror=pixelerror,
                uniformize=uniformize
            )

    # THIRD: run do_photometry to get a .sourcelist file with HATID,X,Y
    if observatory=='hatpi':
        photf = do_photometry(
            os.path.abspath(photref_frame),
            os.path.abspath(fovcatalog),
            outdir=os.path.dirname(os.path.abspath(photref_frame)),
            extractsources=extractsources,
            zeropoint=zeropoint,
            aperturelist=apertures,
            observatory='hatpi'
        )
    elif observatory=='tess':
        # nb. fiphot_xycols is adjusted inside `do_photometry` to match
        # extractsources. extractforsdk makes a fistar file with SDK values for
        # the photref frame, which are kept for later bookkeeping.
        if extractsources==True:
            extractforsdk = False
        elif extractsources==False:
            extractforsdk = True

        photf = do_photometry(
            os.path.abspath(photref_frame),
            os.path.abspath(fovcatalog),
            outdir=os.path.dirname(os.path.abspath(photref_frame)),
            extractsources=extractsources,
            zeropoint=zeropoint,
            ccdgain=ccdgain,
            ccdexptime=ccdexptime,
            binaryoutput=False,
            observatory='tess',
            fluxthreshold=photreffluxthreshold,
            extractforsdk=extractforsdk,
            aperturelist=apertures
        )

    if extractsources:
        photref_sourcelist = os.path.abspath(re.sub(sv.FITS_TAIL,
                                                    '.sourcelist',
                                                    photref_frame))
        srclist_xy = '7,8'
    else:
        photref_sourcelist = os.path.abspath(re.sub(sv.FITS_TAIL,
                                                    '.projcatalog',
                                                    photref_frame))
        srclist_xy = '13,14'

    # FINALLY, run the cmrawphot command using the locations and IDs from the
    # .sourcelist produced by do_photometry
    cmdtorun = COMBINEDREFPHOTCMD.format(
        photref=photref_frame,
        srclist=photref_sourcelist,
        srclist_idcol='1',
        srclist_xycol=srclist_xy,
        ccdgain=ccdgain,
        zeropoint=zeropoint,
        exptime=ccdexptime,
        aperturestring=apertures,
        photrefbase=os.path.splitext(os.path.basename(photref_frame))[0],
        outfile=outfile
    )

    print('fiphot command: %s' % cmdtorun)

    returncode = os.system(cmdtorun)

    if observatory != 'tess':
        # Current default non-TESS behavior: get reference fluxes by measuring
        # them from photometric reference frame. This is a bad default in
        # crowded regions, because the flux of a star in a crowded region
        # measured from the frame will be overestimated.  The resulting
        # _relative depth_ of signals is decreased (since the reference
        # magnitude is overestimated). This lowers detectability. It also
        # weakens detectability since the COMBINEDREFPHOTCMD above hard-codes
        # in a disjoint radius of 2 pixels, so you will get NaNs in crowded
        # regions (which image subtraction can in theory avoid).

        print('WARNING\n'*42)
        print('Please use a better method to measure reference fluxes.')
        print('WARNING\n'*42)

        if returncode == 0:
            print('%sZ: photometry on photref %s OK -> %s' %
                  (datetime.utcnow().isoformat(), photref_frame, outfile))
            return photref_frame, outfile
        else:
            print('ERR! %sZ: photometry on photref %s failed!' %
                  (datetime.utcnow().isoformat(), photref_frame))
            if os.path.exists(outfile):
                os.remove(outfile)
            return photref_frame, None

    elif observatory=='tess' and not extractsources:

        # Above, the reference flux values were measured on the reference frame. We
        # now want to fit these flux values vs catalog magnitudes, and then write
        # the cmrawphot file with reference fluxes that are predicted from the fit.
        colnames = ['sourceid', 'x', 'y', 'n_ap', 'catalog_mag', 'catalog_color',
                    'rad_ap1', 'ann_inner_ap1', 'ann_width_ap1', 'flux_ap1',
                    'flux_err_ap1', 'qflag_ap1',
                    'rad_ap2', 'ann_inner_ap2', 'ann_width_ap2', 'flux_ap2',
                    'flux_err_ap2', 'qflag_ap2',
                    'rad_ap3', 'ann_inner_ap3', 'ann_width_ap3', 'flux_ap3',
                    'flux_err_ap3', 'qflag_ap3']

        # raw photometry dataframe. outfile has lines formatted like:
        #    4623663518281658240        2.904       8.380   3          -
        #    -  1.00  7.00  6.00     718.634     11.8163 0x007f  1.50  7.00
        #    6.00     1978.89      19.583 0x007f  2.25  7.00  6.00     4679.89
        #    30.1105 0x007f

        rpdf = pd.read_csv(outfile, comment='#', delim_whitespace=True,
                           names=colnames)

        # Now need to read in catalog information. This is catalog and observatory
        # specific. Below is only implemented for the TESS pipeline, which has the
        # Gaia DR2 .projcatalog reformated catalog file format, where each line is
        # formatted like:
        #
        #   4623663518281658240 88.6421522981 -78.623013112 -4.0583529625
        #   -6.9954454144 10.6766643524 10.2180233002 10.9946670532 3.0136
        #   1.6152298953 26.648822804 NOT_AVAILABLE     2.9039     8.3803

        # projected catalog sourcelist dataframe
        # cols = ['id,ra,dec,xi,eta,G,Rp,Bp,plx,pmra,pmdec,varflag,x,y']
        from tessutils import read_object_reformed_catalog
        _ = read_object_reformed_catalog(photref_sourcelist, isgaiaid=True,
                                         gaiafull=True)
        sldf = pd.DataFrame(_)
        sldf['sourceid'] = sldf['id'].astype(np.int64)

        mdf = rpdf.merge(sldf, on='sourceid', how='left',
                         suffixes=('_rawphot','_sourcelist'))

        # Stassun+2019, TICv8 paper give conversion from GaiaDR2 photometry to
        # the TESS bandpass. Scatter is 0.006 mag.
        mdf['T'] = (mdf['G']
                    - 0.00522555 * (mdf['Bp'] - mdf['Rp'])**3
                    + 0.0891337 * (mdf['Bp'] - mdf['Rp'])**2
                    - 0.633923 * (mdf['Bp'] - mdf['Rp'])
                    + 0.0324473)

        # For each aperture, fit flux as function of catalog mag.  Only want
        # stars with 0x0000 flags in each aperture.  Also only want stars with
        # finite Gaia G, Rp, and Bp mags
        sel = nparr(mdf['qflag_ap1'] == '0x0000')
        sel &= nparr(mdf['qflag_ap2'] == '0x0000')
        sel &= nparr(mdf['qflag_ap3'] == '0x0000')

        sel &= ~nparr(pd.isnull(mdf['G']))
        sel &= ~nparr(pd.isnull(mdf['Rp']))
        sel &= ~nparr(pd.isnull(mdf['Bp']))

        # Do the selection.
        sdf = mdf[sel]

        n_apertures = 3
        for ap in range(1,n_apertures+1):

            popt, pcov = curve_fit(catmag_to_flux, nparr(sdf['T']),
                                   nparr(sdf['flux_ap{}'.format(ap)]),
                                   p0=(10, 1e4))
            if np.all(popt == (10, 1e4)):
                errmsg = 'curve_fit optimization failed. fix this!'
                raise AssertionError(errmsg)
            if np.any(pd.isnull(pcov)):
                errmsg = 'curve_fit covariance optimization failed. fix this!'
                raise AssertionError(errmsg)

            # Make a plot that shows the line being fit to log10(flux) vs mag.
            plt.close('all')
            f,ax =plt.subplots(figsize=(4,3))

            T_lin = np.arange(min(sdf['T']), max(sdf['T']), 0.1)
            ax.scatter(sdf['T'], sdf['flux_ap{}'.format(ap)], rasterized=True,
                       s=0.15, zorder=2, c='C0')
            ax.plot(T_lin, catmag_to_flux(T_lin, popt[0], popt[1]), zorder=3,
                    c='C1')

            ax.set_yscale('log')
            ax.set_xlabel('T mag')
            ax.set_ylabel('flux in ap{} [ADU]'.format(ap))
            outpath = os.path.join(
                os.path.splitext(photref_frame)[0]+
                '_fluxap{}_vs_Tmag.png'.format(ap)
            )
            f.savefig(outpath, bbox_inches='tight')

            mdf['flux_ap{}_predicted'.format(ap)] = (
                catmag_to_flux(mdf['T'], popt[0], popt[1])
            )

        # Save `mdf`, so that for future reference we know what the measured vs
        # the predicted fluxes were.
        outpath = os.path.join(
            os.path.splitext(photref_frame)[0]+
            '_measured_and_predicted_fluxes.csv'
        )
        mdf.to_csv(outpath, index=False)
        print('saved {}'.format(outpath))

        # Ok. Now copy `outfile` (.cmrawphot) to a .rawphot extension.  Then
        # overwrite .cmrawphot measured fluxes with the predicted reference
        # fluxes. Do it line by line, since fiphot reads the .cmrawphot file
        # with very specific formatting.
        if not outfile.endswith('.cmrawphot'):
            raise AssertionError('expected outfile to be .cmrawphot')

        shutil.copyfile(outfile, outfile.replace('.cmrawphot','.rawphot'))
        print('copied {}->{}'.format(
            outfile, outfile.replace('.cmrawphot','.rawphot')))

        with open(outfile, 'r') as f:
            lines = f.readlines()
        outlines = deepcopy(lines)

        for ix, l in enumerate(lines):

            if l.startswith('#'):
                continue

            sourceid = np.int64(l.split(' ')[0])

            sel = mdf['sourceid'] == sourceid

            # the exact number of characters per string apparently must be
            # matched.  because the
            # `sscanf(cmd[6+i*6+3],"%lg",&rfflux[i].flux)` in
            # fiphot-io.c/read_raw_photometry couldn't work with say...  COMMA
            # SEPARATED values. it needs to be variable length whitespace, with
            # fixed max numbers of characters per column (but variable numbers
            # of characters). LOL! so let's format some strings.
            #
            # each line is like:
            #
            #   4767791282021775744     1931.906       2.682   3          -
            #   -  1.00  7.00  6.00     107.937      5.0005 0x007f  1.50  7.00
            #   6.00     260.402     7.85987 0x007f  2.25  7.00  6.00     390.784
            #   10.2589 0x007f
            #

            # replace measured fluxes with predicted fluxes. do it via EXACT
            # STRING matching, because the %11g C-format with which they're
            # written seems otherwise mind-numbingly hard to get in python
            slices_dict = {
                'ap1':slice(88,100),
                'ap2':slice(137,149),
                'ap3':slice(186,198),
            }

            for ap in range(1,n_apertures+1):

                # " "-padded string with flux in this aperture
                thisslice = slices_dict['ap{}'.format(ap)]
                measured = l[thisslice]

                # this minimum is needed... so that at most 7 characters are
                # written. (ignoring whitespace inserted further on...)
                n_characters = 7

                predicted = '{:s}'.format(
                    str('{:.3f}'.format(
                        float(mdf[sel]['flux_ap{}_predicted'.format(ap)]))
                    )[:n_characters]
                )

                if len(predicted)>7:
                    print('somehow len predicted is above 7')
                    import IPython; IPython.embed()

                # each column entry has max width 7 characters (but might be
                # fewer)... and is designated to END at a particular line,
                # without periods. trailing periods therefore need to be
                # replaced by precursor whitespace.
                if predicted.endswith('.'):
                    predicted = ' {:s}'.format(predicted.replace('.',''))
                elif predicted.endswith('.0'):
                    predicted = predicted[:-2]
                elif predicted.endswith('0'):
                    predicted = predicted[:-1]

                nspaces_needed = len(measured) - len(predicted)
                predicted = nspaces_needed*' ' + predicted

                # the following is more complicated than a .replace()
                # statement, but for 0-columns .replace would be ambiguous
                outline = (
                    outlines[ix][:thisslice.start]
                    + predicted
                    + outlines[ix][thisslice.stop:]
                )
                outlines[ix] = outline

            # add the catalog mag and catalog color.
            # these are read by fitsh with `sscanf(cmd[4],"%lg",&ref_mag)`, so
            # format them as floats. fun trick: python's `.replace` method
            # has an optional argument for maximum number of occurrences.

            # # catalog mag column first
            # outlines[ix] = outlines[ix].replace(
            #     '-',
            #     '{:.3f}'.format(float(mdf[sel]['T'])),
            #     1
            # )

            # # then color column
            # outlines[ix] = outlines[ix].replace(
            #     '-',
            #     '{:.3f}'.format(float(mdf[sel]['Rp']-mdf[sel]['Bp'])),
            #     1
            # )

        with open(outfile, 'w') as f:
            f.writelines(outlines)
        print('overwrote {} with reformatted fluxes, catmags, and colors'.
              format(outfile))

        if returncode == 0:
            print('%sZ: photometry on photref %s OK -> %s' %
                  (datetime.utcnow().isoformat(), photref_frame, outfile))
            return photref_frame, outfile
        else:
            print('ERR! %sZ: photometry on photref %s failed!' %
                  (datetime.utcnow().isoformat(), photref_frame))
            if os.path.exists(outfile):
                os.remove(outfile)
            return photref_frame, None

    else:
        raise NotImplementedError



#################################
## CONVOLUTION AND SUBTRACTION ##
#################################

def get_convsubfits_hash(photreftype, subtracttype, kernelspec):
    """
    This calculates a hash for the convsub FITS files.
    """

    return md5(
        ('%s-%s-%s' % (photreftype, subtracttype, kernelspec)).encode('utf-8')
    ).hexdigest()[:8]



def get_convsubphot_hash(photreftype, subtracttype,
                         kernelspec, lcapertures):
    """
    This calculates a hash for the convsub iphot files.
    """

    return md5(
        ('%s-%s-%s-%s' % (photreftype, subtracttype,
                          kernelspec, lcapertures)).encode('utf-8')
    ).hexdigest()[:8]




def subframe_convolution_worker(task, **kwargs):
    """
    This is a parallel worker to convolve the combined photref frame to each
    input frame, subtract them, and then return the subtracted frame. Used by
    convolve_and_subtract_frames below.

    D_xy = I_xy - M_xy, for

    M_xy = (R \conv K)_xy   + B_xy,

    to minimize

    chi^2 = \sum_xy  (I_xy - M_xy)^2.

    where I_xy is the target image, D_xy is the difference image, M_xy is the
    static background model, R is the reference image, K is the convolution
    kernel, and B_xy is a background term that can be added in (e.g., Alard
    2000 -- the `fitsh` "b/N" term).

    task[0] -> the frame to convolve (AKA: reference, R)
    task[1] -> the frame to use as the convolution target (AKA: image, I)
    task[2] -> the convolution target's registration info file
    task[3] -> the kernel specification for the convolution
    task[4] -> the output directory where to place the results
    task[5] -> whether this is a reverse subtraction
    task[6] -> the photrefprefix to attach to the output convsub FITS filename
    """

    (frametoconvolve, targetframe, convregfile,
     kernelspec, outdir, reversesubtract, photrefprefix) = task

    try:
        colorscheme = kwargs['colorscheme']
    except KeyError:
        colorscheme = None


    # get the hash to prepend to the file
    outfitshash = get_convsubfits_hash(photrefprefix,
                                       ('reverse' if reversesubtract else
                                        'normal'),
                                       kernelspec)

    try:

        # swap these if we're convolving the targetframe to the referenceframe
        if reversesubtract:
            frametoconvolve, targetframe = targetframe, frametoconvolve

        if not outdir:
            if reversesubtract:
                outfile = os.path.join(os.path.dirname(targetframe),
                                       'rsub-%s-%s' %
                                       (outfitshash,
                                        os.path.basename(targetframe)))

                outkernel = os.path.join(os.path.dirname(targetframe),
                                         'rsub-%s-%s-kernel' %
                                         (outfitshash,
                                          os.path.basename(targetframe)))
            else:
                outfile = os.path.join(os.path.dirname(frametoconvolve),
                                       'nsub-%s-%s' %
                                       (outfitshash,
                                        os.path.basename(frametoconvolve)))

                outkernel = os.path.join(os.path.dirname(frametoconvolve),
                                         'nsub-%s-%s-kernel' %
                                         (outfitshash,
                                          os.path.basename(frametoconvolve)))

        else:
            if reversesubtract:
                outfile = os.path.join(outdir,
                                       'rsub-%s-%s' %
                                       (outfitshash,
                                        os.path.basename(targetframe)))
                outkernel = os.path.join(outdir,
                                         'rsub-%s-%s-kernel' %
                                         (outfitshash,
                                          os.path.basename(targetframe)))
            else:
                outfile = os.path.join(outdir,
                                       'nsub-%s-%s' %
                                       (outfitshash,
                                        os.path.basename(frametoconvolve)))
                outkernel = os.path.join(outdir,
                                         'nsub-%s-%s-kernel' %
                                         (outfitshash,
                                          os.path.basename(frametoconvolve)))

        cmdtorun = CONVOLVESUBFRAMESCMD.format(
            targetframe=targetframe,
            frametoconvolve=frametoconvolve,
            convregfile=convregfile,
            kernelspec=kernelspec,
            outputkernel=outkernel,
            outputfile=outfile
        )

        if DEBUG:
            print(cmdtorun)

        returncode = os.system(cmdtorun)

        if returncode == 0:
            print('%sZ: convolution and subtraction OK: %s -> %s' %
                  (datetime.utcnow().isoformat(), frametoconvolve, outfile))

            framejpg = fits_to_full_jpeg(
                outfile,
                out_fname=os.path.join(
                    os.path.dirname(outfile),
                    ('JPEG-SUBTRACTEDCONV-%s.jpg' %
                     os.path.basename(re.sub(sv.FITS_TAIL, '', outfile)))
                    ),
                scale_func=clipped_linscale_img,
                colorscheme=colorscheme
                )

            return frametoconvolve, outfile
        else:
            print('ERR! %sZ: convolution and subtraction failed for %s' %
                  (datetime.utcnow().isoformat(), frametoconvolve,))
            if os.path.exists(outfile):
                os.remove(outfile)
            return frametoconvolve, None

    except Exception as e:

        print('ERR! %sZ: photref convolution failed for %s, exception %s' %
              (datetime.utcnow().isoformat(), frametoconvolve, e))
        return frametoconvolve, None



def convolve_and_subtract_frames(fitsdir,
                                 combinedphotref,
                                 photrefregfile,
                                 reversesubtract=False,
                                 fitsglob='*-xtrns.fits',
                                 kernelspec='b/4;i/4;d=4/4',
                                 photrefprefix=None,
                                 nworkers=16,
                                 maxworkertasks=1000,
                                 outdir=None,
                                 colorscheme=None):
    """
    This convolves the photometric reference to each frame, using the specified
    kernel, then subtracts the frame from the photometric reference to produce
    the subtracted frames.
    """

    if isinstance(fitsdir, list):
        transframelist = fitsdir
    else:
        # find all the files to process
        transframelist = glob.glob(os.path.join(os.path.abspath(fitsdir),
                                                fitsglob))

    # make a list of tasks
    tasks = [(x, combinedphotref, photrefregfile,
              kernelspec, outdir, reversesubtract, photrefprefix)
             for x in transframelist]

    print('%sZ: %s frames to convolve to %s and subtract' %
          (datetime.utcnow().isoformat(), len(transframelist), combinedphotref))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    # NOTE: without kwargs, this would look like:
    #       `results = pool.map(subframe_convolution_worker, tasks)`
    kwargs = {'colorscheme':colorscheme}
    results = pool.map(
        partial(subframe_convolution_worker, **kwargs), tasks
    )

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



def subframe_photometry_worker(task):
    """
    This runs the special version of fiphot in subtracted image mode to
    calculate the ISM magnitudes.

    task[0] -> subtracted frame FITS
    task[1] -> photometric reference frame .cmrawphot file
    task[2] -> disjoint radius
    task[3] -> subtracted frame kernel file
    task[4] -> subtracted frame itrans file
    task[5] -> subtracted frame xysdk file
    task[6] -> output directory
    task[7] -> photrefprefix
    task[8] -> kernelspec (copy this from input to subtraction)
    task[9] -> lcapertures (copy this from input to photref photometry)
    task[10] -> observatory
    task[11] -> photparams. if hatpi, None. elif tess, dict with gain, exptime,
                and zeropoint.
    task[12] -> domagfit (bool). if true, includes the "magfit" option, which
                fits instrument magnitdues vs catalog magnitudes and colors.
                these catalog mags and colors must be specified a priori. (i.e.
                this functionality has not been implemented in a general way to
                pipe-trex).
    """

    # get the info out of the task
    (subframe, photrefrawphot, disjointrad,
     subframekernel, subframeitrans, subframexysdk, outdir, photrefprefix,
     kernelspec, lcapertures, observatory, photparams, domagfit) = task

    try:

        if observatory=='hatpi':

            headerlist = ['GAIN', 'GAIN1', 'GAIN2', 'EXPTIME']
            # get the CCD info out of the subframe
            header = imageutils.get_header_keyword_list(subframe, headerlist)

            if 'GAIN1' in header and 'GAIN2' in header:
                ccdgain = (header['GAIN1'] + header['GAIN2'])/2.0
            elif 'GAIN' in header:
                ccdgain = header['GAIN']
            else:
                ccdgain = None

            ccdexptime = header['EXPTIME'] if 'EXPTIME' in header else None

            # get the zeropoint. if this is a HAT frame, the ccd number will get us
            # the zeropoint in the ZEROPOINTS dictionary
            frameinfo = FRAMEREGEX.findall(os.path.basename(subframe))

            if frameinfo:
                zeropoint = ZEROPOINTS[int(frameinfo[0][-1])]

        elif observatory=='tess':

            ccdgain = photparams['ccdgain']
            ccdexptime = photparams['ccdexptime']
            zeropoint = photparams['zeropoint']

        if not (ccdgain or ccdexptime):
            print('%sZ: no GAIN or EXPTIME defined for %s' %
                  (datetime.utcnow().isoformat(),
                   subframe))
            return subframe, None

        if observatory=='hatpi':
            if not frameinfo:
                print('%sZ: no zeropoint magnitude defined for %s' %
                      (datetime.utcnow().isoformat(),
                       subframe))
                return subframe, None

        # set up the subtraction type in the output filename
        if (os.path.basename(subframe)).startswith('rsub'):
            subtractionbit = 'rsub'
            subtracttype = 'reverse'
        elif (os.path.basename(subframe)).startswith('nsub'):
            subtractionbit = 'nsub'
            subtracttype = 'normal'
        else:
            print('%sZ: unknown subtraction type (not reverse/normal) for %s' %
                  (datetime.utcnow().isoformat(),
                   subframe))
            return subframe, None

        # calculate the hash
        outiphothash = get_convsubphot_hash(photrefprefix,
                                            subtracttype,
                                            kernelspec,
                                            lcapertures)

        if observatory=='hatpi':
            frameiphot = '%s-%s-%s-%s_%s.iphot' % (subtractionbit,
                                                   outiphothash,
                                                   frameinfo[0][0],
                                                   frameinfo[0][1],
                                                   frameinfo[0][2])
        elif observatory=='tess':
            namesub = re.findall(
                'tess20.*?-[0-9][0-9][0-9][0-9]_cal_img_bkgdsub', subframe)
            if not len(namesub) == 1:
                raise AssertionError(
                    'expected only one subframe, got {:s}'.
                    format(repr(namesub)))
            name = namesub[0]
            frameiphot = '%s-%s-%s.iphot' % (subtractionbit, outiphothash, name)

        if outdir:
            outfile = os.path.join(os.path.abspath(outdir),
                                   frameiphot)
        else:
            outfile = os.path.join(os.path.abspath(os.path.dirname(subframe)),
                                   frameiphot)

        if domagfit:
            magfitstr = (
                "--magfit orders=4:2,niter=3,sigma=3 "
                "--col-magnitude 5 --col-color 6 "
            )
        else:
            magfitstr = ""

        cmdtorun = SUBFRAMEPHOTCMD.format(
            subtractedframe=subframe,
            photrefrawphot=photrefrawphot,
            zeropoint=zeropoint,
            exptime=ccdexptime,
            ccdgain=ccdgain,
            disjointradius=disjointrad,
            subtractedkernel=subframekernel,
            subtracteditrans=subframeitrans,
            subtractedxysdk=subframexysdk,
            magfitstr=magfitstr,
            outiphot=outfile
            )

        if DEBUG:
            print(cmdtorun)

        returncode = os.system(cmdtorun)

        if returncode == 0:
            print('%sZ: subtracted frame photometry OK for %s -> %s' %
                  (datetime.utcnow().isoformat(), subframe, outfile))
            return subframe, outfile
        else:
            print('ERR! %sZ: subtracted frame photometry failed for %s' %
                  (datetime.utcnow().isoformat(), subframe,))
            if os.path.exists(outfile):
                os.remove(outfile)
            return subframe, None

    except Exception as e:

        print('ERR! %sZ: subtracted frame photometry failed for %s' %
              (datetime.utcnow().isoformat(), subframe,))
        if os.path.exists(outfile):
            os.remove(outfile)
        return subframe, None



def photometry_on_subtracted_frames(subframedir,
                                    photrefrawphot,
                                    subframeglob='subtracted-*.fits',
                                    subframekerneldir=None,
                                    subframeitransdir=None,
                                    subframexysdkdir=None,
                                    photdisjointradius=2,
                                    photrefprefix=None,
                                    hardprefix='',
                                    nworkers=16,
                                    kernelspec='b/4;i/4;d=4/4',
                                    lcapertures='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
                                    maxworkertasks=1000,
                                    outdir=None,
                                    photparams=None):

    """
    This runs photometry on the subtracted frames and finally produces the ISM
    magnitudes.

    See run_iphot.py and IMG-3-PHOT_st5.sh for what this is supposed to do.
    """

    if isinstance(subframedir, list):
        subframelist = subframedir
    else:
        subframelist = glob.glob(os.path.join(os.path.abspath(subframedir),
                                              subframeglob))

    # we need to find the accompanying kernel, itrans, and xysdk files for each
    # subtracted frame for the tasks list
    tasks = []

    if not subframekerneldir:
        subframekerneldir = subframedir
    if not subframeitransdir:
        subframeitransdir = subframedir
    if not subframexysdkdir:
        subframexysdkdir = subframedir

    if photrefprefix and photrefprefix == 'oneframe':
        photrefbit = 'oneframeref-'
    elif photrefprefix and photrefprefix == 'onehour':
        photrefbit = 'onehourref-'
    elif photrefprefix and photrefprefix == 'onenight':
        photrefbit = 'onenightref-'
    else:
        photrefbit = ''

    # find matching kernel, itrans, and xysdk files for each subtracted frame
    for subframe in subframelist:

        frameinfo = FRAMEREGEX.findall(os.path.basename(subframe))
        kernel = '%s%s%s-%s_%s-xtrns.fits-kernel' % (photrefbit,
                                                   hardprefix,
                                                   frameinfo[0][0],
                                                   frameinfo[0][1],
                                                   frameinfo[0][2])
        kernel = os.path.abspath(os.path.join(subframekerneldir,kernel))

        itrans = '%s-%s_%s.itrans' % (frameinfo[0][0],
                                      frameinfo[0][1],
                                      frameinfo[0][2])
        itrans = os.path.abspath(os.path.join(subframeitransdir,itrans))

        xysdk = '%s-%s_%s.xysdk' % (frameinfo[0][0],
                                    frameinfo[0][1],
                                    frameinfo[0][2])
        xysdk = os.path.abspath(os.path.join(subframexysdkdir,xysdk))

        if (os.path.exists(kernel) and
            os.path.exists(itrans) and
            os.path.exists(xysdk)):

            tasks.append((os.path.abspath(subframe),
                          os.path.abspath(photrefrawphot),
                          photdisjointradius,
                          kernel,
                          itrans,
                          xysdk,
                          outdir,
                          photrefprefix,
                          kernelspec,
                          lcapertures))

    # now start up the parallel photometry
    print('%sZ: %s good frames to run photometry on in %s, starting...' %
          (datetime.utcnow().isoformat(), len(tasks), subframedir))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(subframe_photometry_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



#############################################
## AD HOC LC COLLECTION FOR SINGLE OBJECTS ##
#############################################



def get_lc_for_object(lcobject,
                      framedir,
                      outfile,
                      frameglob='1-*_?.fits',
                      iphotglob='1-*_?.iphot',
                      datekeyword='BJD'):
    """
    This pulls out the photometry for an arbitrary object.

    Gets the line for the object from the iphot files and generates a flux light
    curve. Assumes 5 apertures and that the iphots are the same name pattern as
    the fits files so it can get JD out of their headers.
    """

    # make a list of the iphots
    iphotlist = sorted(glob.glob(os.path.join(framedir, iphotglob)))

    lclines = {}

    regexstr = r'({lcobject}.*)'.format(lcobject=lcobject)
    lcobjregex = re.compile(regexstr)

    for iphotf in iphotlist:

        infd = open(iphotf,'rb')
        iphotcontents = infd.read()
        infd.close()
        objectline = lcobjregex.findall(iphotcontents)

        # find the associated fits frame
        fitspath = iphotf.replace('.iphot','.fits')

        # if we found this object in the LC, then grab its info
        if len(objectline) == 1 and os.path.exists(fitspath):

            objectline = objectline[0].rstrip()

            # find this object's JD
            framedate = imageutils.get_header_keyword(fitspath,
                                                      datekeyword)
            lclines[framedate] = objectline

    # now that we've collected the LC, write it out to disk
    outfd = open(outfile, 'wb')

    for date in sorted(lclines.keys()):

        outline = '{framedate} {framemeasures}\n'
        framedate = date
        framemeasures = lclines[date]
        outfd.write(outline.format(framedate=framedate,
                                   framemeasures=framemeasures))

    outfd.close()

    return outfile



###########################
## LIGHTCURVE COLLECTION ##
###########################

def dump_lightcurves_with_grcollect(photfileglob, lcdir, maxmemory,
                                    objectidcol=3,
                                    lcextension='grcollectilc',
                                    observatory='tess'):
    """
    Given a list of photometry files (text files at various times output by
    fiphot with rows that are objects), make lightcurve files (text files for
    various objects whose rows are times).

    (For TESS, the timestamp imprinted here is JD_UTC midtime.)

    This routine uses the `fitsh` routine `grcollect` to do the dumping. This
    is comparably fast to the postgresql indexing approach without heavy
    optimization (dumps ~1 photometry file per second).

    An important intermediate step, implemented here, is prepending times and
    filenames to *.iphot lightcurve files, to make *.iphottemp files.

    *.iphot photometry files look like:

        HAT-381-0000004     1482.093      521.080   1482.10899   521.079941
        1.86192159    -0.217043415    -0.152707564         0.35         0.41
        583942.10       334.31      5.54226      0.00062            G
        605285.46       340.12      5.50328      0.00061            G
        619455.38       344.29      5.47816      0.00060            G

    Args:

        photfileglob (str): bash glob to pass to grcollect, e.g., the first
        input line below

            grcollect /home/mypath/rsub-3f62ef9f-tess201913112*.iphot \
            --col-base 1 --prefix /nfs/phtess1/ar1/TESS/SIMFFI/LC/ISP_1-1/ \
            --extension grcollectilc --max-memory 4g

        objectidcol (int): column of object id in *.iphottemp files (default:
            3)

        lcdir (str): directory where lightcurves get dumped

        maxmemory (str): e.g., "2g", "100m" for 2gb, or 100mb. Maximum amount
        of memory for `grcollect`. See https://fitsh.net/wiki/man/grcollect for
        terse details.

    Keyword args:

        lcextension (str): e.g., "grcollectilc" for the extension you want your
        dumped lightcurves to have. Also common is "ilc" for "image-subtracted
        lightcurve".


    Returns:

        diddly-squat.
    """

    if not os.path.exists(lcdir):
        os.mkdir(lcdir)

    starttime = time.time()

    photpaths = glob.glob(photfileglob)

    print('%sZ: called dump lightcurves for %d photfiles' %
          (datetime.utcnow().isoformat(), len(photpaths)))

    if len(photpaths)==0:
        print('ERR! %sZ: failed to find %s, continuing' %
              (datetime.utcnow().isoformat(), frametoshift))
        return 0

    # *.iphot files need to have timestamp and filenames prepended on each
    # line, otherwise lightcurves will have no times. make the correctly
    # formatted ".iphottemp" files here.
    for ix, photpath in enumerate(photpaths):

        if observatory=='tess':
            framekey = re.findall(
                'tess20.*?-[0-9][0-9][0-9][0-9]_cal_img_bkgdsub', photpath)
            if not len(framekey) == 1:
                raise AssertionError(
                    'expected only one photframe, got {:s}'.
                    format(repr(framekey)))
            originalframe = os.path.join(os.path.dirname(photpath),
                                         framekey[0]+'.fits')
        elif observatory=='hatpi':
            framekey = re.findall('(.-.{7}_.)\.iphot', photpath)
            assert len(framekey) == 1, 'HATPI specific regex!'
            originalframe = os.path.join(os.path.dirname(photpath),
                                         framekey[0]+'.fits')
        else:
            raise NotImplementedError

        # check these files exist, and populate the dict if they do
        if os.path.exists(originalframe):

            tempphotpath = photpath.replace('.iphot','.iphottemp')

            if os.path.exists(tempphotpath):
                print('%sZ: skipping %d/%d frame, found .iphottemp file %s' %
                      (datetime.utcnow().isoformat(), ix, len(photpaths),
                       tempphotpath))
                continue

            print('%sZ: writing %d/%d frame, %s to .iphottemp file %s' %
                  (datetime.utcnow().isoformat(), ix, len(photpaths), photpath,
                   tempphotpath))

            # this is the ORIGINAL FITS frame, since the subtracted one
            # contains some weird JD header (probably inherited from the
            # photref frame)
            if observatory=='tess':

                from astropy.time import Time

                # observation start and stop time as BJD_UTC calendar dates.
                # calculated by SPOC. (these include the leapseconds). the
                # wrong barycentric correction has been applied to all
                # available timestamps (and the sign convention for how the
                # correction is applied -- is it added, or subtracted? -- has
                # been confirmed via private comm. with Jon Jenkins)
                tstart_bjd_utc_str = get_header_keyword(
                    originalframe, 'DATE-OBS')
                tstop_bjd_utc_str = get_header_keyword(
                    originalframe, 'DATE-END')
                ltt_barycenter_spoc = get_header_keyword(
                    originalframe, 'BARYCORR')

                # record the midtime in JD_UTC with grcollect. barycentric
                # correction comes later. (once ra/dec are accessible).
                tstart_bjd_utc = Time(tstart_bjd_utc_str, format='isot',
                                      scale='utc')
                tstop_bjd_utc = Time(tstop_bjd_utc_str, format='isot',
                                     scale='utc')

                tmid_bjd_utc = (
                    tstart_bjd_utc.jd + (tstop_bjd_utc.jd - tstart_bjd_utc.jd)/2.
                )

                # WANT: bjd_tdb_me = jd_utc_spoc + ltt_barycenter_me + leapseconds (eq 1)
                # HAVE: bjd_utc_spoc = jd_utc_spoc + ltt_barycenter_spoc (eq 2)
                # use eq (2) to get jd_tdb_spoc:
                tmid_jd_utc = tmid_bjd_utc - ltt_barycenter_spoc

                frametime = tmid_jd_utc

            elif observatory=='hatpi':
                frametime = get_header_keyword(originalframe, 'JD')
            else:
                raise NotImplementedError

            # now open the phot file, read it, and write to the tempphot file.
            output = StringIO()
            with open(photpath, 'rb') as photfile:

                # write time precise to 1e-7 days = 8.6 milliseconds.
                for line in photfile:
                    output.write('%.7f\t%s\t%s' % (
                        frametime, framekey[0], line.decode('utf-8'))
                    )

            with open(tempphotpath, 'w') as tempphotfile:
                output.seek(0)
                shutil.copyfileobj(output, tempphotfile)

            output.close()

    grcollectglob = photfileglob.replace('.iphot', '.iphottemp')
    cmdtorun = GRCOLLECTCMD.format(fiphotfileglob=grcollectglob,
                                   lcdir=lcdir,
                                   objectidcol=objectidcol,
                                   lcextension=lcextension,
                                   maxmemory=maxmemory)
    if DEBUG:
        print(cmdtorun)

    # execute the grcollect shell command. NB we execute through a shell
    # interpreter to get correct wildcards applied.
    grcollectproc = subprocess.Popen(cmdtorun, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    _, stderr = grcollectproc.communicate()

    if grcollectproc.returncode == 0:
        print('%sZ: grcollect dump succeeded' % datetime.utcnow().isoformat())
    else:
        print('ERR! %sZ: grcollect failed' % datetime.utcnow().isoformat())
        print('error was %s' % stderr )

    print('%sZ: done, time taken: %.2f minutes' %
          (datetime.utcnow().isoformat(), (time.time() - starttime)/60.0))



def make_photometry_indexdb(framedir,
                            outfile,
                            frameglob='rsub-*-xtrns.fits',
                            photdir=None,
                            photglob='rsub-*-%s.iphot',
                            maxframes=None,
                            overwrite=False):

    """
    This is like make_photometry_index below, but uses an sqlite3 database
    instead of an in-memory disk.
    """

    # make sure we don't overwrite anything unless we're supposed to
    if os.path.exists(outfile) and not overwrite:

        print('WRN! %sZ: a photometry index DB by this name already exists!' %
              (datetime.utcnow().isoformat(),))
        return outfile

    if overwrite and os.path.exists(outfile):
        os.remove(outfile)

    db = sqlite3.connect(outfile)
    cur = db.cursor()

    # make the database tables
    # cur.execute(PRAGMA_CMDS)   # not sure if we want WAL mode or not
    cur.execute(PHOTS_TABLE)
    cur.execute(HATIDS_TABLE)
    cur.execute(META_TABLE)
    db.commit()

    # first, figure out the directories
    if not photdir:
        photdir = framedir

    # send these to the database
    cur.execute(META_INSERT_CMD, (photdir, framedir))
    db.commit()

    if isinstance(framedir, list):
        framelist = framedir
    else:
        # first, find all the frames
        framelist = glob.glob(os.path.join(os.path.abspath(framedir),
                                           frameglob))

    # restrict to maxframes max frames
    if maxframes:
        framelist = framelist[:maxframes]

    # go through all the frames
    for frame in framelist:

        print('%sZ: working on frame %s' %
              (datetime.utcnow().isoformat(), frame))

        # generate the names of the associated phot and sourcelist files
        frameinfo = FRAMEREGEX.findall(os.path.basename(frame))

        photsearch = photglob % ('%s-%s_%s' % (frameinfo[0][0],
                                               frameinfo[0][1],
                                               frameinfo[0][2]))

        originalframe = '%s-%s_%s.fits' % (frameinfo[0][0],
                                           frameinfo[0][1],
                                           frameinfo[0][2])

        photmatch = glob.glob(os.path.join(os.path.abspath(photdir),
                                           photsearch))
        originalframe = os.path.join(os.path.abspath(framedir),
                                     originalframe)

        # check these files exist, and populate the dict if they do
        if (photmatch and os.path.exists(photmatch[0])
            and os.path.exists(originalframe)):

            phot = photmatch[0]

            # get the JD from the FITS file.

            # NOTE: this is the ORIGINAL FITS frame, since the subtracted one
            # contains some weird JD header (probably inherited from the photref
            # frame)
            framerjd = get_header_keyword(originalframe, 'JD')

            # update the DB with this info
            cur.execute(PHOTS_INSERT_CMD,
                        (os.path.basename(phot),
                         framerjd,
                         os.path.basename(originalframe)))

            # get the phot file
            photf = open(phot, 'rb')
            phothatids = [x.split()[0] for x in photf]
            photf.close()

            for ind, hatid in enumerate(phothatids):

                # update the DB with phot info
                cur.execute(HATIDS_INSERT_CMD,
                            (hatid,
                             os.path.basename(phot),
                             ind))

        # if some associated files don't exist for this frame, ignore it
        else:

            print('WRN! %sZ: ignoring frame %s, '
                  'photometry for this frame is not available!' %
                  (datetime.utcnow().isoformat(), frame))


    # make the indices for fast lookup
    print('%sZ: making photometry index DB indices...' %
          (datetime.utcnow().isoformat(),))
    cur.execute(PHOTS_INDEX_CMD)
    cur.execute(HATIDS_INDEX_CMD)
    cur.execute(HATIDS_PHOT_INDEX_CMD)

    # commit the DB at the end of writing
    db.commit()

    print('%sZ: done. photometry index DB written to %s' %
          (datetime.utcnow().isoformat(), outfile))

    return outfile



def get_iphot_line(iphot, linenum, lcobject, iphotlinechars=338):
    """
    This gets a random iphot line out of the file iphot.
    """

    iphotf = open(iphot, 'rb')
    filelinenum = iphotlinechars*linenum

    if filelinenum > 0:
        iphotf.seek(filelinenum - 100)
    else:
        iphotf.seek(filelinenum)

    iphotline = iphotf.read(iphotlinechars + 100)

    linestart = iphotline.index(lcobject)
    iphotline = iphotline[linestart:]
    lineend = iphotline.index('\n')
    iphotline = iphotline[:lineend]

    iphotf.close()

    return iphotline



def get_iphot_line_linecache(iphot, linenum, lcobject, iphotlinechars=260):
    """
    This uses linecache's getline function to get the line out of the file
    iphot.
    """

    return getline(iphot, linenum)



def get_iphot_line_sed(iphot, linenum, lcobject, iphotlinechars=260):
    """
    This uses the sed utility to pull the line out of the iphot.

    Following: http://stackoverflow.com/questions/6022384

    sed '{linenum}q;d' file
    """

    try:

        pout = subprocess.check_output("sed '%sq;d' %s" % (linenum+1, iphot),
                                       shell=True)
        return pout

    except subprocess.CalledProcessError:

        return ''




def iphot_line_tail(iphot, linenum, lcobject, iphotlinechars=338):
    """
    Uses head | tail.
    """

    try:
        cmd = 'head -n {headline} {iphot} | tail -n 1'.format(
            headline=linenum+1,
            iphot=iphot
        )

        pout = check_output(cmd, shell=True)
        return poutput.rstrip(' \n')
    except:
        return ''


def collect_imagesubphot_lightcurve(hatid,
                                    photindex,
                                    outdir,
                                    skipcollected=True,
                                    iphotlinefunc=get_iphot_line,
                                    iphotlinechars=338,
                                    mindetections=50):
    """
    This collects the imagesubphot lightcurve of a single object into a .ilc
    file.

    hatid -> the hatid of the object to collect the light-curve for

    photindexfile -> the file containing the master index of which .iphot,
                      .sourcelist, and .fits contain the lines corresponding to
                      this HATID. this way, we can look these lines up
                      super-fast using the linecache module.

    outdir -> the directory where to the place the collected lightcurve

    skipcollected -> if True, looks for an existing LC for this hatid in
                       outdir. if found, returns the path to that LC instead of
                       actually processing. if this is False, redoes the
                       processing for this LC anyway.

    iphotlinefunc -> this is the function to use for getting a specific line out
                     of the specified iphot file.

    The collected LC is similar to the aperturephot LC, but some extra columns
    added by fiphot running on the subtracted frames. columns are:

    00 rjd    Reduced Julian Date (RJD = JD - 2400000.0)
    01 rstfc  Unique frame key ({STID}-{FRAMENUMBER}_{CCDNUM})
    02 hat    HAT ID of the object
    03 xcc    original X coordinate on CCD before shifting to astromref
    04 ycc    original y coordinate on CCD before shifting to astromref
    05 xic    shifted X coordinate on CCD after shifting to astromref
    06 yic    shifted Y coordinate on CCD after shifting to astromref
    07 fsv    Measured S value
    08 fdv    Measured D value
    09 fkv    Measured K value
    10 bgv    Background value
    11 bge    Background measurement error
    12 irm1   Instrumental magnitude in aperture 1
    13 ire1   Instrumental magnitude error for aperture 1
    14 irq1   Instrumental magnitude quality flag for aperture 1 (0/G OK, X bad)
    15 irm2   Instrumental magnitude in aperture 2
    16 ire2   Instrumental magnitude error for aperture 2
    17 irq2   Instrumental magnitude quality flag for aperture 2 (0/G OK, X bad)
    18 irm3   Instrumental magnitude in aperture 3
    19 ire3   Instrumental magnitude error for aperture 3
    20 irq3   Instrumental magnitude quality flag for aperture 3 (0/G OK, X bad)
    """

    # connect to the photindex sqlite3 database
    indexdb = sqlite3.connect(photindex)
    cur = indexdb.cursor()

    # first, look up the metainfo
    cur.execute(META_SELECT_CMD)
    metarow = cur.fetchone()
    photdir, framedir = metarow

    # look up the hatid and its info in the photindex db
    cur.execute(PHOT_SELECT_CMD, (str(hatid),))
    rows = cur.fetchall()

    # make sure we have enough observations for this object to make it
    # worthwhile
    if rows and len(rows) >= mindetections:

        # prepare the output file
        outfile = os.path.join(os.path.abspath(outdir), '%s.ilc' % hatid)

        # if the file already exists and skipcollected is True, then return
        # that file instead of processing any further
        if os.path.exists(outfile) and skipcollected:

            print('WRN! %sZ: object %s LC already exists, not overwriting: %s' %
                  (datetime.utcnow().isoformat(), hatid, outfile))

            return outfile

        # otherwise, open the file and prepare to write to it
        outf = open(outfile, 'wb')

        # go through the phots and sourcelists, picking out the timeseries
        # information for this hatid
        for row in rows:

            # unpack the row to get our values
            framerjd, phot, photline = row

            try:

                # open the phot
                infd = open(os.path.join(photdir, phot),'rb')
                for ind, line in enumerate(infd):
                    if ind == photline:
                        photelemline = line.rstrip(' \n')
                        break
                infd.close()

                phot_elem = photelemline.split()

                if len(phot_elem) > 0:

                    # parse these lines and prepare the output
                    rstfc_elems = FRAMEREGEX.findall(os.path.basename(phot))
                    rstfc = '%s-%s_%s' % (rstfc_elems[0])
                    out_line = '%s %s %s\n' % (framerjd, rstfc,
                                               ' '.join(phot_elem))
                    outf.write(out_line.encode('utf-8'))

            # if this frame isn't available, ignore it
            except Exception as e:

                print('WRN! %sZ: phot %s isn\'t available (error: %s)'
                      ', skipping...' %
                      (datetime.utcnow().isoformat(), phot, e))
                continue

        # close the output LC once we're done with it
        outf.close()

        print('%sZ: object %s -> %s' %
              (datetime.utcnow().isoformat(), hatid, outfile))

        returnf = outfile

    # if the hatid isn't found in the photometry index or it has less than the
    # required number of detections, then we can't do anything
    else:

        print('ERR! %sZ: object %s is not in the '
              'photometry index or has less than %s detections, ignoring...' %
              (datetime.utcnow().isoformat(), hatid, mindetections))

        returnf = None

    # at the end, close the DB and return
    indexdb.close()
    return returnf



def imagesublc_collection_worker(task):
    """
    This wraps collect_imagesuphot_lightcurve for parallel_collect_lightcurves
    below.

    task[0] -> hatid
    task[1] -> photindex DB name
    task[2] -> outdir
    task[3] -> {skipcollected, iphotlinefunc, iphotlinechars, mindetections}
    """

    try:

        return task[0], collect_imagesubphot_lightcurve(task[0],
                                                        task[1],
                                                        task[2],
                                                        **task[3])

    except Exception as e:

        print('ERR! %sZ: failed to get LC for %s, error: %s' %
              (datetime.utcnow().isoformat(), task[0], e ))
        return task[0], None



def parallel_collect_imagesub_lightcurves(
        framedir,
        outdir,
        photindexdb=None,
        frameglob='rsub-*-xtrns.fits',
        photglob='rsub-*-%s.iphot',
        photdir=None,
        maxframes=None,
        overwritephotindex=False,
        skipcollectedlcs=True,
        iphotlinefunc=get_iphot_line,
        iphotlinechars=338,
        mindetections=50,
        nworkers=16,
        maxworkertasks=1000
    ):
    """
    This collects all .iphot files into lightcurves.
    """

    # first, check if the output directory exists
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # next, check if we have to make a photometry index DB, and launch the
    if not photindexdb:

        photdbf = os.path.join(framedir,'TM-imagesubphot-index.sqlite')

        # this picks up framedir/framelist for make_photometry_indexdb
        photindexdb = make_photometry_indexdb(framedir,
                                              photdbf,
                                              frameglob=frameglob,
                                              photglob=photglob,
                                              photdir=photdir,
                                              maxframes=maxframes,
                                              overwrite=overwritephotindex)

    # only proceed if the photometry index DB exists
    if os.path.exists(photindexdb):

        # get the list of distinct HATIDs from the photindexdb
        db = sqlite3.connect(photindexdb)
        cur = db.cursor()
        cur.execute(DISTINCT_HATIDS_CMD)
        rows = cur.fetchall()
        hatids = [x[0] for x in rows]
        db.close()

        # generate the task list
        tasks = [(hatid,
                  photindexdb,
                  outdir,
                  {'skipcollected':skipcollectedlcs,
                   'iphotlinefunc':iphotlinefunc,
                   'iphotlinechars':iphotlinechars,
                   'mindetections':mindetections}) for hatid in hatids]

        # now start up the parallel collection
        print('%sZ: %s HATIDs to get LCs for, starting...' %
              (datetime.utcnow().isoformat(), len(hatids), ))
        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        # fire up the pool of workers
        results = pool.map(imagesublc_collection_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}

    # if the photometry index DB doesn't exist, nothing we can do
    else:

        print('ERR! %sZ: specified photometry index DB does not exist!' %
              (datetime.utcnow().isoformat(), ))



def lc_concatenate_worker(task):
    """
    This is the parallel worker for the function below.

    task is a tuple:

    task[0] = base LC file
    task[1] = new LC file to be concatenated to base
    task[2] = zero-indexed column number to use to sort resulting LC
              (if None, then no sorting is done)

    FIXME: this currently uses a dumb method, make it faster by streaming newlc
    lines to baselc file directly

    FIXME: add sorting by any column
    """

    try:

        baselcfile = task[0]
        newlcfile = task[1]

        baselcf = open(baselcfile, 'rb')
        baselclines = baselcf.readlines()
        baselcf.close()
        baselcndet = len(baselclines)

        newlcf = open(newlcfile, 'rb')
        newlclines = newlcf.readlines()
        newlcf.close()
        newlcndet = len(newlclines)

        baselclines.extend(newlclines)
        finalndet = len(baselclines)

        with open(baselcfile,'wb') as outfd:
            for line in baselclines:
                outfd.write(line.encode('utf-8'))

        print('%sZ: concat LC OK: %s with ndet %s -> %s with ndet %s' %
              (datetime.utcnow().isoformat(),
               baselcfile, baselcndet, baselcfile, finalndet ))
        return baselcfile, baselcfile

    except Exception as e:

        print('ERR! %sZ: concat LC task failed: %s, error: %s' %
              (datetime.utcnow().isoformat(), repr(task), e ))
        return task[0], None



def parallel_concatenate_lightcurves(baselcdir,
                                     newlcdir,
                                     lcext='ilc',
                                     sortcol=0,
                                     nworkers=16,
                                     maxworkertasks=1000):
    """
    This concatenates light curves in baselcdir with newlcdir.

    Searches for light curves in the baselcdir, then searches for LCs in the
    newlcdir. Concatenates the light curves of matching objects together and
    writes them to the baselcdir.

    Useful for adding night-by-night reductions to a set of already reduced
    LCs. The proposed workflow is:

    - reduce the night to photometry files using the same photometry base
      (cmrawphot) used for all the nights before
    - create a photindexdb for the night's photometry
    - collect night's photometry into light curves
    - use parallel_concatenate_lightcurves to add night's LCs to base LCs
    - rerun post-processing, EPD, TFA, etc.
    """

    baselcpaths = sorted(glob.glob(os.path.join(baselcdir),
                                  '*.{lcext}'.format(lcext=lcext)))
    newlcpaths = sorted(glob.glob(os.path.join(newlcdir),
                                  '*.{lcext}'.format(lcext=lcext)))

    baselcbnames = set([os.path.basename(x) for x in baselcpaths])
    newlcbnames = set([os.path.basename(x) for x in newlcpaths])

    # find the intersection of the base and new LC sets
    print('%sZ: finding light curves... ' %
          (datetime.utcnow().isoformat(), ))
    worklcbnames = list(baselcbnames.intersect(newlcbnames))

    if len(worklcbnames) > 0:

        print('%sZ: %s LCs match between base: %s, new: %s ' %
              (datetime.utcnow().isoformat(),
               len(worklcbnames),
               baselcdir,
               newlcdir))

        tasks = [(os.path.join(baselcdir,x),
                  os.path.join(newlcdir,x),
                  sortcol) for x in worklcbnames]

        # now start up the parallel collection
        print('%sZ: starting...' %
              (datetime.utcnow().isoformat(), ))
        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        # fire up the pool of workers
        results = pool.map(lc_concatenate_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}


    else:

        print('ERR! %sZ: no LCs match between base: %s, new: %s' %
              (datetime.utcnow().isoformat(), baselcdir, newlcdir ))
        return None


############################################
## SPECIAL EPD FUNCTIONS FOR IMAGESUB LCS ##
############################################

def epd_diffmags_imagesub_linearmodel(coeff, fsv, fdv, fkv, xcc, ycc, mag,
                          temperatures=None,
                          observatory='hatpi'):

    """
    This calculates the difference in mags after EPD coefficients are calculated
    for imagesub lightcurves. The only difference is that we don't use the
    background or background error to figure out the fit.

    final EPD mags = median(magseries) + epd_diffmags()
    """

    if observatory=='hatpi':
        return -(coeff[0]*fsv**2. +
                 coeff[1]*fsv +
                 coeff[2]*fdv**2. +
                 coeff[3]*fdv +
                 coeff[4]*fkv**2. +
                 coeff[5]*fkv +
                 coeff[6] +
                 coeff[7]*fsv*fdv +
                 coeff[8]*fsv*fkv +
                 coeff[9]*fdv*fkv +
                 coeff[10]*np.sin(2*np.pi*xcc) +
                 coeff[11]*np.cos(2*np.pi*xcc) +
                 coeff[12]*np.sin(2*np.pi*ycc) +
                 coeff[13]*np.cos(2*np.pi*ycc) +
                 coeff[14]*np.sin(4*np.pi*xcc) +
                 coeff[15]*np.cos(4*np.pi*xcc) +
                 coeff[16]*np.sin(4*np.pi*ycc) +
                 coeff[17]*np.cos(4*np.pi*ycc) +
                 coeff[18]*(xcc - np.floor(xcc)) +
                 coeff[19]*(ycc - np.floor(ycc))
                 -
                 mag)
    elif observatory=='tess':
        return -(coeff[0]*fsv**2. +
                 coeff[1]*fsv +
                 coeff[2]*fdv**2. +
                 coeff[3]*fdv +
                 coeff[4]*fkv**2. +
                 coeff[5]*fkv +
                 coeff[6] +
                 coeff[7]*fsv*fdv +
                 coeff[8]*fsv*fkv +
                 coeff[9]*fdv*fkv +
                 coeff[10]*np.sin(2*np.pi*xcc) +
                 coeff[11]*np.cos(2*np.pi*xcc) +
                 coeff[12]*np.sin(2*np.pi*ycc) +
                 coeff[13]*np.cos(2*np.pi*ycc) +
                 coeff[14]*np.sin(4*np.pi*xcc) +
                 coeff[15]*np.cos(4*np.pi*xcc) +
                 coeff[16]*np.sin(4*np.pi*ycc) +
                 coeff[17]*np.cos(4*np.pi*ycc) +
                 coeff[18]*(xcc - np.floor(xcc)) +
                 coeff[19]*(ycc - np.floor(ycc)) +
                 coeff[20]*temperatures
                 -
                 mag)


def rolling_window(a, window):
    # a cheap way to pass a rolling window over an array
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def epd_magseries_imagesub(mag, fsv, fdv, fkv, xcc, ycc,
                           smooth=21, sigmaclip=3.0, observatory='hatpi',
                           temperatures=None, times=None, isfull=True):
    """
    For each star, ask how the flux correlates with "external parameters" (for
    instance the intra-pixel position, or the CCD temperature, or the shape
    parameters S,D,K, or the hour angle / zenith distance from the ground).

    Detrend these correlations with EPD External parameters are specific to
    each star (TFA decorrelates remaining "unknown parameters" by looking for
    correlations between similar stars).

    A traditional fit for each stars from HATS/HATNet looks like:
    \begin{equation}
        mag =
        a_s S + a_s2 S^2 + a_d D + a_d2 D^2 + a_k K + a_k2 K^2 +
        const +
        a_sd S * D + a_sk S * K + a_dk D * K +
        a_x \sin(2\pi x) + b_x \cos(2\pi x) +
        a_y \sin(2\pi y) + b_y \cos(2\pi y) +
        a_2x \sin(4\pi x) + b_2x \cos(4\pi x) +
        a_2y \sin(4\pi y) + b_2y \cos(4\pi y) +
        a_intrax (x - floor(x)) +
        a_intray (y - floor(y)).
    \end{equation}

    However, different parameters may be correlated with different instruments.
    An appropriate EPD model needs to be determined on an instrument by
    instrument basis, by looking at plots of e.g., magnitude vs (time, S, D, K,
    x, y, {x}, {y}, temperature, hour angle, zenith distance, background level)
    etc. Also, look at "pair plots" (triangle plots scattering each of these
    dimensions against the other).

    For TESS, a linear model is not a good fit to the data. However the
    magnitudes have a secular drift that correlates with that seen in (x, y,
    and CCD temperature). We therefore adopt a nonlinear, spline model.

    References:
        Andras Pal's thesis, Section 2.10.1
        Bakos et al 2010, Appendix of HAT-P-11b paper.
        Zhang, Bakos et al 2016, Section 5.7.

    Args:
        fsv: S value (analog of FWHM from elongated gaussian fit to stars)

        fdv: D value (elongation parameters, see Pal 2009 Eq 31 for definition)

        fkv: K value (ditto)

        xcc: x coordinate centroid

        ycc: y coordinate centroid

    Kwargs:

        smooth (int) : number of cadences used when passing a median filter
        over the magnitude time series. If 21 with HATPI, means a ~10 minute
        median filter.

        sigmaclip (float): number of sigma away from median from which to
        ignore values in the EPD fit (NOTE: this does NOT sigma clip the actual
        time-series.)

        observatory (str): "hatpi" and "tess" implemented. Each uses a
        different equation for EPD detrending.

        temperatures (np.ndarray): CCD temperatures. These are expected for
        TESS data.

    Returns:
        If HATPI: array of "differential" EPD mags (to add back to median mag),
        if the solution succeeds.

        If TESS: same array, also returns a dictionary with the fit details,
        including useful tidbits like the predicted (X,Y,temperature,magnitude)
        vectors.

        (NOTE: we are adding the magnitudes back. In fitting models to LCs, if
        you're modeling the FLUX with some multiplicative function that is
        distorting the number of electrons you are reading out, then dividing
        out your model is best.  However, in magnitudes, that multiplicative
        function becomes additive, because logarithms. So add for magnitudes,
        divide for fluxes).
    """

    # find all the finite values of the magnitude
    finiteind = npisfinite(mag)

    # calculate median and stdev
    mag_median = np.median(mag[finiteind])
    mag_stdev = np.nanstd(mag)

    # if we're supposed to sigma clip, do so
    if sigmaclip:
        excludeind = abs(mag - mag_median) < sigmaclip*mag_stdev
        finalind = finiteind & excludeind
    else:
        finalind = finiteind

    final_mag = mag[finalind]
    final_len = len(final_mag)

    if DEBUG:
        print('final epd fit mag len = %s' % final_len)

    # smooth the signal
    smoothedmag = medfilt(final_mag, smooth)

    if observatory=='hatpi':
        try:
            # make the linear equation matrix.
            epdmatrix = np.c_[fsv[finalind]**2.0,
                              fsv[finalind],
                              fdv[finalind]**2.0,
                              fdv[finalind],
                              fkv[finalind]**2.0,
                              fkv[finalind],
                              np.ones(final_len),
                              fsv[finalind]*fdv[finalind],
                              fsv[finalind]*fkv[finalind],
                              fdv[finalind]*fkv[finalind],
                              np.sin(2*np.pi*xcc[finalind]),
                              np.cos(2*np.pi*xcc[finalind]),
                              np.sin(2*np.pi*ycc[finalind]),
                              np.cos(2*np.pi*ycc[finalind]),
                              np.sin(4*np.pi*xcc[finalind]),
                              np.cos(4*np.pi*xcc[finalind]),
                              np.sin(4*np.pi*ycc[finalind]),
                              np.cos(4*np.pi*ycc[finalind]),
                              xcc[finalind] - np.floor(xcc[finalind]),
                              ycc[finalind] - np.floor(ycc[finalind])
                             ]

            # solve the equation epdmatrix * x = smoothedmag
            # return the EPD differential mags if the solution succeeds
            coeffs, residuals, rank, singulars = lstsq(epdmatrix, smoothedmag)

            if DEBUG:
                print('coeffs = %s, residuals = %s' % (coeffs, residuals))

            return epd_diffmags_imagesub_linearmodel(coeffs, fsv, fdv, fkv,
                                                     xcc, ycc, mag,
                                                     observatory=observatory)
        except Exception as e:
            # if the solution fails, return nothing
            print('ERR! %sZ: EPD solution did not converge! Error was: %s' %
                  (datetime.utcnow().isoformat(), e))
            return None

    elif observatory=='tess':

        ##########################################
        # for each orbit, fit a spline for (x,y,temperature,smoothedmag).  then
        # divide it out.  (note: the time-sort is a sanity check -- it was
        # already done when making the grcollect -> FITS "raw" lightcurves).
        ##########################################
        # cut for sigma-clipped / finite values
        times = times[finalind]
        xcc = xcc[finalind]
        ycc = ycc[finalind]
        temperatures = temperatures[finalind]

        # sort by time
        sort = np.argsort(times)
        final_mag = smoothedmag[sort]
        xcc = xcc[sort]
        ycc = ycc[sort]
        temperatures = temperatures[sort]

        orbitgap = 1. # days
        expected_norbits = 2 if isfull else 1
        norbits, groups = lcmath.find_lc_timegroups(times, mingap=orbitgap)

        if norbits != expected_norbits:
            outmsg = (
                'EPD assumes given two orbits. {} orbits. Time {}, Mag {}'.
                format(norbits, repr(times), repr(final_mag))
            )
            raise AssertionError(outmsg)

        fitdetails = {}
        orbitind = 0
        for group in groups:
            groupind = np.zeros_like(times)
            groupind[group.start : group.stop] += 1
            groupind = groupind.astype(np.bool)

            tg_mag = final_mag[groupind]
            tg_x = xcc[groupind]
            tg_y = ycc[groupind]
            tg_temp = temperatures[groupind]

            theta = [tg_mag, tg_x, tg_y, tg_temp]
            ndim = len(theta)

            # to get the weight vector used in chi^2 estimation, estimate
            # uncertainty in x,y,T,mag as 1-sigma standard deviation from 6-hr
            # timescale window.
            sigma_vec = []
            for _theta in theta:
                windowsize = 12 # 6 hour timescale = 12 time points for FFIs.
                _sigma = np.std(rolling_window(_theta, windowsize))
                sigma_vec.append(np.ones_like(_theta)*np.nanmean(_sigma))

            # Find B-spline representation of N-dimensional curve. Assumption
            # is that \vec{theta}(t) represents a curve in N-dimensional space
            # parametrized by u, for u in [0,1]. We are trying to find a smooth
            # approximating spline curve g(u). The main function wraps the
            # PARCUR routine from FITPACK, a FORTRAN library. Written by Paul
            # Dierckx.  PARCUR is for fitting of parametric open curves. (e.g.,
            # Dierckx, "Curve and surface fitting with splines", (1993, pg 111,
            # sec 6.3.1).  The method, and statement of the optimization
            # problem, are equations 6.46 and 6.47 of the above reference.  The
            # smoothing parameter s must be positive.  For each value of the
            # smoothing parameter, refit for the optimal number of knots.
            # Calculate chi^2 and BIC. Set the maximum number of knots for a
            # 13.7 day time-series (one orbit) to be 13, to prevent fitting out
            # <1 day frequencies unless they involve sharp features.  (This
            # imposes a high-pass filter.) The grid spacing is arbitrary, but
            # should be enough to get some decent resolution, and to bracket
            # the BIC-minimum.
            s_grid = np.logspace(-4,-1,num=50)
            n_knot_max = 13 if isfull else 4 # ~one knot per day.
            korder = 3 # spline order

            spld = {}
            BICs = []
            for s in s_grid:
                (tckp, u), metric, ier, msg = splprep(theta, s=s, k=korder,
                                                      nest=n_knot_max, task=0,
                                                      full_output=1)

                # t_knots: "this array contains the knots of the spline curve,
                # i.e.  the position of the interior knots t(k+2),
                # t(k+3),..,t(n-k-1) as well as the position of the additional
                # t(1)=t(2)=...=t(k+1)=ub and t(n-k)=...=t(n)=ue needed for the
                # b-spline representation"
                #
                # knots: knots of the spline curve; i.e. position of the knots
                # in (x,y,temperature,mag) space.
                #
                # order: b-spline order
                t_knots, knots, order = tckp

                n_dim = 4
                n_data = len(theta[0])
                n_knots = np.atleast_1d(knots).shape[1]
                k_freeparam = n_dim*( n_knots + korder + 1 )

                # Compute chisq by evaluating the spline.

                mag_pred, x_pred, y_pred, T_pred = splev(u, tckp)
                pred_vec = mag_pred, x_pred, y_pred, T_pred

                chisq = 0
                for _theta, _theta_pred, _sigma_theta in zip(
                    theta, pred_vec, sigma_vec
                ):
                    chisq += np.sum( ( (_theta - _theta_pred)/_sigma_theta)**2 )

                BIC = chisq + k_freeparam * np.log(n_data)

                spld[s] = {}
                spld[s]['tckp'] = tckp
                spld[s]['n_knots'] = n_knots
                spld[s]['u'] = u
                spld[s]['metric'] = metric
                spld[s]['chisq'] = chisq
                spld[s]['n_data'] = n_data
                spld[s]['k_freeparam'] = k_freeparam
                spld[s]['BIC'] = BIC

                spld[s]['mag_pred'] = mag_pred
                spld[s]['x_pred'] = x_pred
                spld[s]['y_pred'] = y_pred
                spld[s]['T_pred'] = T_pred

                BICs.append(BIC)

            # using BIC, get the best-fit model (the only one we will keep).
            # for each spacecraft orbit, save the best-fit model, and also the
            # tckp and u vectors used to generate it.

            BICs = nparr(BICs)
            sel = np.argmin(BICs)
            best_s = s_grid[sel]
            fitdetails[orbitind] = {}

            for k in ['tckp', 'u', 'n_knots', 'chisq', 'n_data', 'k_freeparam',
                      'BIC', 'x_pred', 'y_pred', 'T_pred', 'mag_pred']:

                fitdetails[orbitind][k] = spld[best_s][k]

            orbitind += 1

        if DEBUG:
            print('{}'.format(repr(fitdetails)))

        # now stitch together the two orbits, and subtract out the model
        if isfull:
            mag_pred = np.concatenate(([fitdetails[0]['mag_pred'],
                                        fitdetails[1]['mag_pred']]))
        else:
            # for the TUNE case, no need to stitch (because it's a subset of
            # one orbit).
            mag_pred = fitdetails[0]['mag_pred']

        epd_diffmags_imagesub_splinemodel = final_mag - mag_pred

        return epd_diffmags_imagesub_splinemodel, fitdetails



    else:
        raise NotImplementedError


def epd_lightcurve_imagesub(ilcfile,
                            mags=[14,19,24],
                            sdk=[7,8,9],
                            xy=[5,6],
                            smooth=21,
                            sigmaclip=3.0,
                            ilcext='ilc',
                            outfile=None,
                            minndet=200,
                            observatory='hatpi'):
    """
    Runs the EPD process on ilcfile, using columns specified to get the required
    parameters. If outfile is None, the .epdlc will be placed in the same
    directory as ilcfile.

    00 rjd    Reduced Julian Date (RJD = JD - 2400000.0)
    01 rstfc  Unique frame key ({STID}-{FRAMENUMBER}_{CCDNUM})
    02 hat    HAT ID of the object
    03 xcc    original X coordinate on CCD on photref frame
    04 ycc    original y coordinate on CCD on photref frame
    05 xic    shifted X coordinate on CCD on subtracted frame
    06 yic    shifted Y coordinate on CCD on subtracted frame
    07 fsv    Measured S value
    08 fdv    Measured D value
    09 fkv    Measured K value
    10 bgv    Background value
    11 bge    Background measurement error

    12 ifl1   Flux in aperture 1 (ADU)
    13 ife1   Flux error in aperture 1 (ADU)
    14 irm1   Instrumental magnitude in aperture 1
    15 ire1   Instrumental magnitude error for aperture 1
    16 irq1   Instrumental magnitude quality flag for aperture 1 (0/G OK, X bad)

    17 ifl2   Flux in aperture 2 (ADU)
    18 ife2   Flux error in aperture 2 (ADU)
    19 irm2   Instrumental magnitude in aperture 2
    20 ire2   Instrumental magnitude error for aperture 2
    21 irq2   Instrumental magnitude quality flag for aperture 2 (0/G OK, X bad)

    22 ifl3   Flux in aperture 3 (ADU)
    23 ife3   Flux error in aperture 3 (ADU)
    24 irm3   Instrumental magnitude in aperture 3
    25 ire3   Instrumental magnitude error for aperture 3
    26 irq3   Instrumental magnitude quality flag for aperture 3 (0/G OK, X bad)

    27 ep1    EPD magnitude for aperture 1
    28 ep2    EPD magnitude for aperture 2
    29 ep3    EPD magnitude for aperture 3
    """

    # read the lightcurve in
    ilc = np.genfromtxt(ilcfile,
                        usecols=tuple(xy + sdk + mags),
                        dtype='f8,f8,f8,f8,f8,f8,f8,f8',
                        names=['xic','yic',
                               'fsv','fdv','fkv',
                               'rm1','rm2','rm3'])
    if observatory=='hatpi':
        temperatures=None
    elif observatory=='tess':
        raise DeprecationWarning
        raise NotImplementedError(
            'use parallel_run_epd_imagesub_fits instead. '+
            'EPD for text-file lightcurves not implemented for TESS, because '+
            'keeping track of additional time-series vectors is impractical.'
        )
    else:
        raise NotImplementedError('observatory must be "tess" or "hatpi"')

    if len(ilc['xic']) >= minndet:

        # get the indices where all columns are non-nan
        combinedok = (npisfinite(ilc['xic']) &
                      npisfinite(ilc['yic']) &
                      npisfinite(ilc['fsv']) &
                      npisfinite(ilc['fdv']) &
                      npisfinite(ilc['fkv']) &
                      npisfinite(ilc['rm1']) &
                      npisfinite(ilc['rm2']) &
                      npisfinite(ilc['rm3']))

        # calculate the EPD differential mags
        epddiffmag1 = epd_magseries_imagesub(
            ilc['rm1'][combinedok],
            ilc['fsv'][combinedok],
            ilc['fdv'][combinedok],
            ilc['fkv'][combinedok],
            ilc['xic'][combinedok],
            ilc['yic'][combinedok],
            smooth=smooth, sigmaclip=sigmaclip,
            observatory=observatory, temperatures=temperatures
            )

        epddiffmag2 = epd_magseries_imagesub(
            ilc['rm2'][combinedok],
            ilc['fsv'][combinedok],
            ilc['fdv'][combinedok],
            ilc['fkv'][combinedok],
            ilc['xic'][combinedok],
            ilc['yic'][combinedok],
            smooth=smooth, sigmaclip=sigmaclip,
            observatory=observatory, temperatures=temperatures
            )

        epddiffmag3 = epd_magseries_imagesub(
            ilc['rm3'][combinedok],
            ilc['fsv'][combinedok],
            ilc['fdv'][combinedok],
            ilc['fkv'][combinedok],
            ilc['xic'][combinedok],
            ilc['yic'][combinedok],
            smooth=smooth, sigmaclip=sigmaclip,
            observatory=observatory, temperatures=temperatures
            )

        # add the EPD diff mags back to the median mag to get the EPD mags
        if epddiffmag1 is not None:
            mag_median = np.nanmedian(ilc['rm1'])
            epdmag1 = epddiffmag1 + mag_median
        else:
            epdmag1 = np.array([np.nan for x in ilc['rm1'][combinedok]])
            print('WRN! %sZ: no EP1 mags available for %s!' %
                  (datetime.utcnow().isoformat(), ilcfile))

        if epddiffmag2 is not None:
            mag_median = np.nanmedian(ilc['rm2'])
            epdmag2 = epddiffmag2 + mag_median
        else:
            epdmag2 = np.array([np.nan for x in ilc['rm2'][combinedok]])
            print('WRN! %sZ: no EP2 mags available for %s!' %
                  (datetime.utcnow().isoformat(), ilcfile))

        if epddiffmag3 is not None:
            mag_median = np.nanmedian(ilc['rm3'])
            epdmag3 = epddiffmag3 + mag_median
        else:
            epdmag3 = np.array([np.nan for x in ilc['rm3'][combinedok]])
            print('WRN! %sZ: no EP3 mags available for %s!' %
                  (datetime.utcnow().isoformat(), ilcfile))

        # now write the EPD LCs out to the outfile
        if not outfile:
            outfile = '%s.epdlc' % ilcfile.rstrip('.%s' % ilcext)

        inf = open(ilcfile,'rb')
        inflines = inf.readlines()
        inf.close()

        # get only the lines that have no nans in the epd input columns
        inflines = (np.array(inflines))[combinedok]

        outf = open(outfile,'wb')

        # only these lines can be attached to the output epd mags
        for line, epd1, epd2, epd3 in zip(inflines, epdmag1, epdmag2, epdmag3):
            outline = '%s %.6f %.6f %.6f\n' % (
                line.decode('utf-8').rstrip('\n'), epd1, epd2, epd3)
            outf.write(outline.encode('utf-8'))

        outf.close()

        print('%sZ: ilc %s with %s dets -> epdlc %s with %s dets' %
              (datetime.utcnow().isoformat(),
               ilcfile, len(ilc['xic']), outfile, len(inflines)))

        return outfile

    else:
        print('not running EPD for %s, ndet = %s < min ndet = %s' %
              (ilcfile, len(ilc['xic']), minndet))
        return None


def serial_run_epd_imagesub(ilcdir,
                            ilcglob='*.ilc',
                            outdir=None,
                            smooth=21,
                            sigmaclip=3.0,
                            minndet=200):
    """
    This runs EPD on the lightcurves from the pipeline.

    00 rjd    Reduced Julian Date (RJD = JD - 2400000.0)
    01 rstfc  Unique frame key ({STID}-{FRAMENUMBER}_{CCDNUM})

    02 hat    HAT ID of the object
    03 xcc    original X coordinate on CCD on photref frame
    04 ycc    original y coordinate on CCD on photref frame
    05 xic    shifted X coordinate on CCD on subtracted frame
    06 yic    shifted Y coordinate on CCD on subtracted frame
    07 fsv    Measured S value
    08 fdv    Measured D value
    09 fkv    Measured K value
    10 bgv    Background value
    11 bge    Background measurement error

    12 ifl1   Flux in aperture 1 (ADU)
    13 ife1   Flux error in aperture 1 (ADU)
    14 irm1   Instrumental magnitude in aperture 1
    15 ire1   Instrumental magnitude error for aperture 1
    16 irq1   Instrumental magnitude quality flag for aperture 1 (0/G OK, X bad)

    17 ifl2   Flux in aperture 2 (ADU)
    18 ife2   Flux error in aperture 2 (ADU)
    19 irm2   Instrumental magnitude in aperture 2
    20 ire2   Instrumental magnitude error for aperture 2
    21 irq2   Instrumental magnitude quality flag for aperture 2 (0/G OK, X bad)

    22 ifl3   Flux in aperture 3 (ADU)
    23 ife3   Flux error in aperture 3 (ADU)
    24 irm3   Instrumental magnitude in aperture 3
    25 ire3   Instrumental magnitude error for aperture 3
    26 irq3   Instrumental magnitude quality flag for aperture 3 (0/G OK, X bad)

    27 ep1    EPD magnitude for aperture 1
    28 ep2    EPD magnitude for aperture 2
    29 ep3    EPD magnitude for aperture 3

    30 tf1    TFA magnitude for aperture 1
    31 tf2    TFA magnitude for aperture 2
    32 tf3    TFA magnitude for aperture 3
    """

    if not outdir:
        outdir = ilcdir

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # pick up the list/dir of input light curves
    if isinstance(ilcdir, list):
        ilcfiles = ilcdir
    else:
        ilcfiles = glob.glob(os.path.join(ilcdir, ilcglob))

    for ilc in ilcfiles:

        outepd = os.path.join(outdir,
                              os.path.basename(ilc).replace('.ilc','.epdlc'))

        print('%sZ: doing EPD for %s...' %
              (datetime.utcnow().isoformat(), ilc))

        try:
            outfilename = epd_lightcurve_imagesub(
                ilc,
                outfile=outepd,
                smooth=smooth,
                sigmaclip=sigmaclip,
                ilcext=os.path.splitext(ilcglob)[-1],
                minndet=minndet
                )
        except Exception as e:
            print('EPD failed for %s, error was: %s' % (ilc, e))



def parallel_epd_worker(task):
    """
    This is the worker for the function below.

    task[0] = ilc
    task[1] = outdir
    task[2] = smooth
    task[3] = sigmaclip
    task[4] = minndet
    task[5] = observatory
    """

    try:

        ilc, outdir, smooth, sigmaclip, minndet, observatory = task
        ilcext = os.path.splitext(ilc)[-1]
        outepd = os.path.join(
            outdir, os.path.basename(ilc).replace(ilcext,'.epdlc')
        )

        outf = epd_lightcurve_imagesub(
            ilc,
            outfile=outepd,
            smooth=smooth,
            sigmaclip=sigmaclip,
            ilcext=ilcext,
            minndet=minndet,
            observatory=observatory
        )

        print('%sZ: %s -> %s EPD OK' %
              (datetime.utcnow().isoformat(), ilc, outf))
        return ilc, outf

    except Exception as e:

        print('EXC! %sZ: %s EPD failed because: %s' %
              (datetime.utcnow().isoformat(), ilc, e))

        return ilc, None



def parallel_run_epd_imagesub(ilcdir,
                              ilcglob='*.ilc',
                              outdir=None,
                              smooth=21,
                              sigmaclip=3.0,
                              nworkers=16,
                              overwrite=False,
                              maxworkertasks=1000,
                              minndet=200,
                              observatory='hatpi'):
    """
    This runs EPD in parallel.
    """

    if not outdir:
        outdir = ilcdir

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # pick up the list/dir of input light curves
    if isinstance(ilcdir, list):
        ilcfiles = ilcdir
    else:
        ilcfiles = glob.glob(os.path.join(ilcdir, ilcglob))

    # if overwrite is false, only run EPD on LCs that don't have it
    existingepd = glob.glob(os.path.join(outdir,'*.epdlc'))
    ilcext = os.path.splitext(ilcglob)[-1]

    if len(existingepd) > 0 and not overwrite:

        requested = list(map(os.path.basename, glob.glob(ilcdir+ilcglob)))
        alreadyexists = list(map(os.path.basename, existingepd))

        # substitute out the hash string
        alreadyexists = [ae.replace('.epdlc','') for ae in alreadyexists]
        requested = [r.replace(ilcext,'') for r in requested]

        setdiff = np.setdiff1d(requested, alreadyexists)

        if len(setdiff) == 0:
            print('WRN! %sZ: EPD already run on selected lighcurves.' %
                  (datetime.utcnow().isoformat(),))
            return

        else:
            ilcfiles = [os.path.join(ilcdir,sd+ilcext) for sd in setdiff]

    tasks = [(x, outdir, smooth, sigmaclip, minndet, observatory)
             for x in ilcfiles]

    print('%sZ: starting EPD...' % (datetime.utcnow().isoformat(),))
    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(parallel_epd_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}




###################
## TFA FUNCTIONS ##
###################

def run_tfa_singlelc(epdlc,
                     templatefiles,
                     outfile=None,
                     epdlc_jdcol=0,
                     epdlc_magcol=(27,28,29),
                     template_sigclip=5.0,
                     epdlc_sigclip=5.0,
                     keep_epdlc=True):
    """
    This runs TFA for all apertures defined in epdlc_magcol for the input
    epdlc file, given an existing TFA template list in templatefile. If outfile
    is None, the output TFA LC will be in the same directory as epdlc but with
    an extension of .tfalc.

    Args:
        keep_epdlc (bool): If False, removes the epdlc file (since the same information
            is contained in the tfalc anyway). If TFA failed, will simply rename
            epdlc to tfalc.
    """

    tfacmdstr = ("tfa -i {epdlc} -t {templatefile} "
                 "--col-jd {epdlc_jdcol} "
                 "--col-mag {epdlc_magcol} "
                 "--col-mag-out {tfalc_magcol} "
                 "--epsilon-time 1e-9 "
                 "--join-by-time "
                 "--templ-filter --templ-outl-limit {template_sigclip} "
                 "--lc-filter --lc-outl-limit {epdlc_sigclip} "
                 "--log-svn-version "
                 "-o {out_tfalc}")

    if not outfile:
        outfile = epdlc.replace('.epdlc','.tfalc')

    tfalc_output = []

    # figure out the number of templates for each aperture; only the stars with
    # ndets > 2 x number of templates will have TFA light-curves generated

    # get the ndets for this epdlc
    with open(epdlc,'rb') as epdfile:
        epdlines = epdfile.readlines()
        epdlen = len(epdlines)

    tfarunnable = False

    # check if the number of detections in this LC is more than 2 x ntemplates
    # for each aperture

    for tfatempf in templatefiles:
        with open(tfatempf,'rb') as tfatemplist:
            templistlines = tfatemplist.readlines()
            tfatemplen = len(templistlines)
            if epdlen >= 2*tfatemplen:
                tfarunnable = True

    if tfarunnable:

        # run tfa for each aperture
        for templatef, magcol, magind in zip(templatefiles,
                                             epdlc_magcol,
                                             range(len(epdlc_magcol))):

            in_jdcol = epdlc_jdcol + 1
            in_magcol = magcol + 1
            out_magcol = in_magcol + 3

            aperture_outfile = outfile + ('.TF%s' % (magind+1))

            tfacmd = tfacmdstr.format(epdlc=epdlc,
                                      templatefile=templatef,
                                      epdlc_jdcol=in_jdcol,
                                      epdlc_magcol=in_magcol,
                                      tfalc_magcol=out_magcol,
                                      template_sigclip=template_sigclip,
                                      epdlc_sigclip=epdlc_sigclip,
                                      out_tfalc=aperture_outfile)

            if DEBUG:
                print(tfacmd)

            # execute the tfa shell command
            tfaproc = subprocess.Popen(shlex.split(tfacmd),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            # get results
            tfa_stdout, tfa_stderr = tfaproc.communicate()

            # get results if succeeded, log outcome, and return path of outfile
            if tfaproc.returncode == 0:

                tfalc_output.append(aperture_outfile)

            else:

                print('%sZ: aperture %s TFA failed for %s! Error was: %s' % (
                      datetime.utcnow().isoformat(), magind+1, epdlc,
                      tfa_stderr))

                tfalc_output.append(None)

        try:
            # make the .tfalc file using an outer join.
            # pandas read_csv is significantly faster than np.genfromtxt
            # merge on epd file
            epd_df = pd.read_csv(epdlc, delim_whitespace=True, comment='#',
                                 header=None, dtype=str)

            # Get number of columns in tfalc files
            last_col = len(pd.read_csv(outfile+'.TF1', delim_whitespace=True,
                                       comment='#', header=None,
                                       nrows=1).columns) - 1

            for i in range(len(epdlc_magcol)):
                tfa_df = pd.read_csv(outfile+('.TF%d' % (i+1)), header=None,
                                     delim_whitespace=True, comment='#',
                                     dtype=str, usecols=[1,last_col])
                epd_df = epd_df.merge(tfa_df, on=1, how='outer')

            epd_df.sort_values(by=0, inplace=True)
            epd_df.to_csv(outfile, sep='\t', na_rep='nan', header=False,
                          index=False)

            print('%sZ: made %s' % (
                  datetime.utcnow().isoformat(), outfile))

            # Remove .TF{1, 2, 3}
            for i in range(len(epdlc_magcol)):
                os.remove(outfile+('.TF%d' % (i+1)))

            if not keep_epdlc:
                os.remove(epdlc)

        except Exception as e:

            print('%sZ: .tfalc.TF{1..3} merge failed for %s! Error was: %s' %
                  (datetime.utcnow().isoformat(), outfile, e))

        return tfalc_output

    else:

        print('ERR! %sZ: no TFA possible for %s! ndet < 2 x n(TFA templates)' %
              (datetime.utcnow().isoformat(), epdlc))

        if not keep_epdlc:
            os.rename(epdlc, outfile)

        return None



def parallel_tfa_worker(task):
    """
    This wraps run_tfa_singlelc above.
    """

    try:

        result = run_tfa_singlelc(task[0], task[1], **task[2])

    except Exception as e:

        print('%sZ: TFA failed for %s! Error was: %s' %
              (datetime.utcnow().isoformat(), task[0], e))
        result = None

    return result


def parallel_run_tfa(lcdir,
                     templatefiles,
                     epdlc_glob='*.epdlc',
                     epdlc_jdcol=0,
                     epdlc_magcol=(27,28,29),
                     template_sigclip=5.0,
                     epdlc_sigclip=5.0,
                     nworkers=16,
                     overwrite=False,
                     workerntasks=1000,
                     tfalc_glob='*.tfalc',
                     keep_epdlc=True):
    """
    This runs TFA on the EPD lightcurves.
    """

    # pick up the list/dir of EPD LCs to run TFA on
    if isinstance(lcdir, list):
        epdlcfiles = lcdir
    else:
        epdlcfiles = glob.glob(os.path.join(lcdir, epdlc_glob))

    # only run TFA on lightcurves that need it
    existing = glob.glob(os.path.join(lcdir, tfalc_glob))

    if len(existing) > 0 and not overwrite:

        requested = list(map(os.path.basename, epdlcfiles))
        alreadyexists = list(map(os.path.basename, existing))

        # substitute out the appropriate strings
        requested = np.sort(np.array([r.replace('.epdlc','') for r in
                                      requested]))

        # works for TFA output naming convention either HAT-528-0000017.tfalc
        # or HAT-528-0000017.tfalc.TF1
        alreadyexists = np.sort(np.unique(
                        [re.sub('\.TF[1-3]','',ae).replace('.tfalc','') for ae
                         in alreadyexists]))

        setdiff = np.setdiff1d(requested, alreadyexists)

        if len(setdiff) == 0:
            print('WRN! %sZ: TFA already done on all lightcurves.' %
                  (datetime.utcnow().isoformat(),))
            return

        else:
            epdlcfiles = [lcdir+sd+'.epdlc' for sd in setdiff]

    tasks = [(x, templatefiles, {'epdlc_jdcol':epdlc_jdcol,
                                 'epdlc_magcol':epdlc_magcol,
                                 'template_sigclip':template_sigclip,
                                 'epdlc_sigclip':epdlc_sigclip,
                                 'keep_epdlc':keep_epdlc})
             for x in epdlcfiles]

    print('%sZ: %s objects to process, starting parallel TFA...' %
          (datetime.utcnow().isoformat(), len(epdlcfiles)))

    pool = mp.Pool(nworkers, maxtasksperchild=workerntasks)
    results = pool.map(parallel_tfa_worker, tasks)
    pool.close()
    pool.join()

    print('%sZ: done. %s LCs processed.' %
          (datetime.utcnow().isoformat(), len(epdlcfiles)))

    return {x:y for x,y in zip(epdlcfiles, results)}
