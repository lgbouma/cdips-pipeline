#!/usr/bin/env python

'''imagesubphot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - March 2015

This contains functions to do image subtraction photometry.

GENERAL ORDER OF THINGS

0. you need reduced frames with accompanying fiphot (photometry) files, and
   fistar (source detection) files. see aperturephot.py's docstring (steps 1
   through 5) for how to get to this state. also see framecalib.py for how to
   get from raw frames to calibrated reduced frames.

   (see /nfs/phs3/ar1/S/HP0/PHOT_WB/ccd5-work on phs3 for an example of a
    directory that has most of the needed stuff in it)

1. the first order of business is to select an astrometric reference frame
   (astromref) using select_astromref_frame. this should be a frame that has the
   sharpest and roundest stars. see select_astromref_frame below to see other
   useful selectors that are applied.

2. next, use get_smoothed_xysdk_coeffs to generate files that contain the
   smoothed S, D, K coefficients for each source detection list. this is needed
   later when we do photometry on the subtracted frames.

3. use get_astromref_shifts to calculate the X and Y coordinate shifts between
   the selected astromref and all other frames we're working on. these will be
   used to shift these other frames to the coordinate system of the astromref,
   which is required to subtract these frames cleanly.

4. use transform_frames_to_astromref to do the actual shifting of all frames to
   the coordinate system of the astromref. this produces new FITS files with the
   '-xtrns.fits' postfix in their filenames. these are the files we'll use from
   now on.

5. use generate_astromref_registration_info to generate a file that grids up the
   astromref source detections into a 30 x 30 grid based on their x and y
   coordinates. not sure what this does exactly, but the file it generates is
   required by the actual convolution steps later.

6. the next thing to do is to select a bunch of frames that can serve as
   photometric reference frames (photrefs). use select_photref_frames for
   this. see the docstring there for the list of selectors used. we'll then
   stack these photrefs into a combined photref later.

7. now that we have photrefs, we have to convolve their PSFs to the best
   available PSF we have (from the astromref). in this way, we match both the
   PSF of the image and the coordinates; these are needed to combine the frames
   correctly. use covolve_photref_frames for this task. this function produces
   FITS with PHOTREF- prefixes to indicate that these are the convolved photref
   frames.

8. use combine_frames to combine all PHOTREF-*-xtrns.fits frames. this creates a
   single high quality photometric reference frame that we'll subtract from all
   other frames to produce difference images.

9. get raw photometry on this combined photref by using
   photometry_on_combined_photref. this produces the base photometry values that
   we'll be diffing from those found in the difference images to get difference
   magnitudes.

10. use convolve_and_subtract_frames to convolve all other frames to the
    combined photref, and generate difference images. the produced images will
    have a subtracted- prefix in their filenames.

11. finally, use photometry_on_subtracted_frames to do photometry on the
    subtracted frames to produce difference magnitudes for each image. these
    calculated mags are put into .iphot files.

12. use parallel_collect_lightcurves to collect the .iphot files into .ilc
    lightcurve files containing image subtraction photometric timeseries for
    each star.

the next few steps are common between imagesubphot.py and aperturephot.py, so
you can use the functions there for them (but that might have issues with
differing column numbers, so I'll probably end up remaking them for
imagesubphot.py):

13. run serial_run_epd or parallel_run_epd to do EPD on all LCs.

14. run parallel_lc_statistics to collect stats on .epdlc files.

15. run choose_tfa_template to choose TFA template stars using the .epdlc stats.

16. run parallel_run_tfa for TFA to get .tfalc.TF{1,2,3} files (FIXME: still
    need to collect into single .tfalc files for all apertures)

17. run parallel_lc_statistics to collect stats on .tfalc files.

18. run parallel_bin_lightcurves to bin LCs to desired time-bins.

19. run parallel_binnedlc_statistics to collect stats for the binned LCs.

20. run plot_stats_file to make RMS vs. mag plots for all unbinned and binned
    LCs.

21. run plot_magrms_comparison to compare the mag-RMS relation for various CCDs.

22. run plot_ismphot_comparison to compare against ISM photometry statistics for
    the same field (requires common stars).

'''

#############
## IMPORTS ##
#############

import os
import os.path
import glob
import multiprocessing as mp
import subprocess
import shlex
from datetime import datetime
import re
import json
import shutil
import random
import cPickle as pickle

import numpy as np

from scipy.spatial import cKDTree as kdtree
from scipy.signal import medfilt
from scipy.linalg import lstsq
from scipy.stats import sigmaclip as stats_sigmaclip
from scipy.optimize import curve_fit

import scipy.stats
import numpy.random as nprand

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import pyfits

import imageutils
from imageutils import get_header_keyword

# get fiphot binary reader
try:
    from HATpipepy.Common.BinPhot import read_fiphot
    HAVEBINPHOT = True
except:
    print("can't import binary fiphot reading functions from "
          "HATpipe, binary fiphot files will be unreadable!")
    HAVEBINPHOT = False


#################
## DEFINITIONS ##
#################

# set this to show extra info
DEBUG = False

# CCD minimum and maximum X,Y pixel coordinates
# used to strip things outside FOV from output of make_frame_sourcelist
CCDEXTENT = {'x':[0.0,2048.0],
             'y':[0.0,2048.0]}

# zeropoint mags for the HATPI lenses given exp time of 30 seconds
# from Chelsea's src directory on phs3: run_phot_astrom.py (2014-12-15)
# FIXME: check where these came from and fix if out of date, especially if
# cameras moved around
ZEROPOINTS = {5:17.11,
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

SUBFRAMEPHOTCMD = (
    "fiphot --input-subtracted {subtractedframe} "
    "--input-raw-photometry {photrefrawphot} "
    "--sky-fit mode,iterations=2,sigma=3 "
    "--format IXY-----,sMm "
    "--mag-flux {zeropoint},{exptime} "
    "--gain {ccdgain} "
    "--disjoint-radius {disjointradius} "
    "--magfit orders=4:2,niter=3,sigma=3 "
    "--input-kernel {subtractedkernel} "
    "--comment --output - | "
    "grtrans --col-xy 2,3 "
    "--input-transformation {subtracteditrans} "
    "--col-out 4,5 "
    "--output - | "
    "grtrans --col-xy 4,5 "
    "--input-transformation {subtractedxysdk} "
    "--col-out 6,7,8 "
    "--output {outiphot}"
)

##################################
## ASTROMETRIC REFERENCE FRAMES ##
##################################

def select_astromref_frame(fitsdir,
                           fitsglob,
                           srclistdir=None,
                           srclistext='.fistar',
                           photdir=None,
                           photext='.fiphot'):
    '''
    This picks an astrometric reference frame.

    We're looking for (in order):

    - highest median S (smallest FWHM)
    - median D value closest to zero (roundest stars)
    - lowest median background
    - largest number of sources with good extractions


    '''

    # first, get the frames
    fitslist = glob.glob(os.path.join(fitsdir, fitsglob))

    if not srclistdir:
        srclistdir = fitsdir
    if not photdir:
        photdir = fitsdir

    print('%sZ: %s FITS files found in %s matching glob %s, '
          'finding photometry and source lists...' %
          (datetime.utcnow().isoformat(),
           len(fitslist), fitsdir, fitsglob))

    goodframes = []
    goodphots = []
    goodsrclists = []

    # associate the frames with their fiphot files
    for fits in fitslist:

        photpath = os.path.join(
            photdir,
            os.path.basename(fits).strip('.fits.fz') + photext
            )
        srclistpath = os.path.join(
            srclistdir,
            os.path.basename(fits).strip('.fits.fz') + srclistext
            )

        if os.path.exists(photpath) and os.path.exists(srclistpath):
            goodframes.append(fits)
            goodphots.append(photpath)
            goodsrclists.append(srclistpath)


    # we only work on goodframes now
    print('%sZ: selecting an astrometric reference frame...' %
          (datetime.utcnow().isoformat(),))


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
            header = photf.read(600)

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

            print('%sZ: %s is a binary phot file, '
                  'but no binary phot reader is present, skipping...' %
                  (datetime.utcnow().isoformat(), phot))
            continue

        else:

            # read in the phot file
            photdata = np.genfromtxt(
                phot,
                usecols=(12,13,14),
                dtype='f8,f8,S5',
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
    median_sval_ind = median_sval_ind[:200]
    median_dval_ind = median_dval_ind[:200]
    median_background_ind = median_background_ind[:200]
    good_detections_ind = good_detections_ind[:200]

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

        return selectedreference

    # otherwise, fall back to to the frames with the best values of S, D and
    # a large number of detections
    elif len(sdndet_ind) > 0:

        selectedreference = goodframes[sdndet_ind[0]]

        print('WRN! %sZ: selected best astrometric reference frame '
              '(using S, D, and ndet only) is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        return selectedreference


    # otherwise, fall back to to the frames with the best values of S and D
    elif len(sd_ind) > 0:

        selectedreference = goodframes[sd_ind[0]]

        print('WRN! %sZ: selected best astrometric reference frame '
              '(using S and D only) is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        return selectedreference

    # if that fails, fail back to the best S value frame
    elif len(median_sval_ind) > 0:

        selectedreference = goodframes[median_sval_ind[0]]

        print('WRN! %sZ: selected best astrometric reference frame '
              '(using S only) is %s' %
              (datetime.utcnow().isoformat(), selectedreference))

        return selectedreference

    else:

        print('ERR! %sZ: could not select a good astrometric reference frame!' %
              (datetime.utcnow().isoformat(), ))

        return



def xysdk_coeffs_worker(task):
    '''
    This is a parallel worker to run the xysdk coeff operation.

    '''

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
                              maxworkertasks=1000):
    '''
    This generates smoothed xy and sdk coefficents for use with iphot later
    (these go into the output photometry file or something).

    grtrans ${APPHOT}/$base.${EXT_FISTAR} --col-xy 2,3 --col-fit 6,7,8 \
                  --col-weight 10 --order 4 \
                  --iterations 3 --rejection-level 3 \
                  --comment --output-transformation ${IPHOT}/$base.xysdk

    '''

    fistarlist = glob.glob(os.path.join(os.path.abspath(fistardir), fistarglob))

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
    '''
    This is a parallel worker for getting the shifts between the astromref frame
    and the frame in the task definition.

    task[0] = target frame fistar
    task[1] = astromref frame fistar
    task[2] = outdir

    '''

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
                         astromrefsrclist,
                         fistarglob='*.fistar',
                         outdir=None,
                         nworkers=16,
                         maxworkertasks=1000):
    '''
    This gets shifts between the astrometric reference frame and all other
    frames.

    '''

    fistarlist = glob.glob(os.path.join(os.path.abspath(fistardir), fistarglob))

    print('%sZ: %s files to process in %s' %
          (datetime.utcnow().isoformat(), len(fistarlist), fistardir))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, astromrefsrclist, outdir) for x in fistarlist]

    # fire up the pool of workers
    results = pool.map(astromref_shift_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



def frame_to_astromref_worker(task):
    '''
    This is a parallel worker for the frame shift to astromref frame operation.

    task[0] = FITS frame to shift
    task[1] = directory where transform files are
    task[2] = output directory

    '''

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
    '''
    This shifts all frames to the astrometric reference.

    '''

    fitslist = glob.glob(os.path.join(os.path.abspath(fitsdir), fitsglob))

    print('%sZ: %s files to process in %s' %
          (datetime.utcnow().isoformat(), len(fitslist), fitsdir))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x, itransdir, outdir) for x in fitslist]

    # fire up the pool of workers
    results = pool.map(frame_to_astromref_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}


def generate_astromref_registration_info(astromrefsrclist,
                                         outfile,
                                         xycols=(1,2)):
    '''This generates a registration information file using the astrometry
    reference frame. This file is then used by the convolution step somehow to
    figure out the convolution kernel? In any case, it's needed for:

    - generating convolved reference frames to be ultimately stacked into a
      single photometric reference frame

    - do the convolution of the reference frame to each -xtrns target frame when
      doing the image subtraction

    '''

    # get the x and y coordinate columns from the source list (fistar)
    srcxy = np.genfromtxt(astromrefsrclist,
                          usecols=xycols,
                          dtype='f8,f8',
                          names=['x','y'])

    # set up the grid (this weirdness is transcribed directly from Chelsea's
    # regslct.py) TODO: figure out WTF this does

    BX = 30.; BY = 30.
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

    for i in xrange(int(BX*BY)):
        outf.write("%8.0f %8.0f %8.0f\n" % (mx[i],my[i],20))

    outf.close()


##################################
## PHOTOMETRIC REFERENCE FRAMES ##
##################################

def select_photref_frames(fitsdir,
                          fitsglob='*-xtrns.fits',
                          photdir=None,
                          photext='.fiphot',
                          srclistdir=None,
                          srclistext='.fistar',
                          minframes=80,
                          maxhourangle=3.0,
                          maxmoonphase=25.0,
                          maxmoonelev=-10.0,
                          maxzenithdist=30.0,
                          forcecollectinfo=False):
    '''This selects a group of photometric reference frames that will later be
    stacked and medianed to form the single photometric reference frame.

    0. this is run on the transformed frames (the ones with -xtrns.fits)

    1. returns a list that is at least minframes long of frames suitable for
    combining into a median reference frame, using the following list of
    selectors (on the original versions (?; probably)):

    - best median scatter of photometry
    - lowest median error in photometry
    - lowest median background measurement
    - low zenith distance
    - high moon distance and lowest moon elevation
    - hour angle with +/- 3 hours
    - large number of stars detected with good flags

    for all selected frames, we will get the median of the background values
    near the center 512x512 pixels. then, we'll enforce that the median of
    background values of each frames be within some delta of the overall
    median. this is basically a slacker way to get rid of cloudy nights.

    2. we convolve all of these to the astrometric reference frame's PSF
    (they're already in the same coordinates as the astromref).

    3. now that all the frames are in the same coordinate system, and have been
    convolved to the same PSF, we can median-stack them (using scipy or
    ficombine)

    '''
    # first, get the frames
    fitslist = glob.glob(os.path.join(fitsdir, fitsglob))

    if not srclistdir:
        srclistdir = fitsdir
    if not photdir:
        photdir = fitsdir

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

    if (not os.path.exists(os.path.join(fitsdir,
                                         'TM-imagesub-photref.pkl')) or
        forcecollectinfo):

        # from the FITS
        zenithdist, moondist, moonelev, moonphase, hourangle = [], [], [], [], []

        # from the fiphot files
        ngoodobjects, medmagerr, magerrmad, medsrcbg = [], [], [], []


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
                header = photf.read(600)

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
                    dtype='f8,f8,S5',
                    names=['mag','err','flag']
                    )

            # 3. get the data fro mthe fistar file
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

            # fistar data
            medsrcbg.append(np.nanmedian(srcdata['background']))

        #
        # done with collecting data, choose the best photometric reference frames
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
            'medsrcbkg':medsrcbg
        }

        # write this info dict to a file so we can quickly load it later
        outpf = open(os.path.join(fitsdir, 'TM-imagesub-photref.pkl'), 'wb')
        pickle.dump(infodict, outpf, pickle.HIGHEST_PROTOCOL)
        outpf.close()

    # if the imagesub photref info file exists already, load it up
    else:

        print('%sZ: loading existing photref select info from %s' %
              (datetime.utcnow().isoformat(),
               os.path.join(fitsdir, 'TM-imagesub-photref.pkl')))

        inpf = open(os.path.join(fitsdir, 'TM-imagesub-photref.pkl'), 'rb')
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

    # this is the final operating set of frames that will be sorted for the
    # following tests
    selectind = haind & moonind & zenithind

    selected_frames = infodict['frames'][selectind]

    selected_ngoodobj = infodict['ngoodobjs'][selectind]
    selected_medmagerr = infodict['medmagerr'][selectind]
    selected_magerrmad = infodict['magerrmad'][selectind]
    selected_medsrcbkg = infodict['medsrcbkg'][selectind]

    print('%sZ: selected %s frames with acceptable '
          'HA, Z, moon phase, and elevation for further filtering...' %
          (datetime.utcnow().isoformat(), len(selected_frames)))

    # do the more strict selection only if we have at least 2 x minframes
    if len(selected_frames) >= 2*minframes:

        # now sort these by the required order
        sorted_ngoodobj_ind = (np.argsort(selected_ngoodobj)[::-1])[:2*minframes]
        sorted_medmagerr_ind = (np.argsort(selected_medmagerr))[:2*minframes]
        sorted_magerrmad_ind = (np.argsort(selected_magerrmad))[:2*minframes]
        sorted_medsrcbkg_ind = (np.argsort(selected_medsrcbkg))[:2*minframes]

        select_ind1 = np.intersect1d(
            sorted_medmagerr_ind,
            sorted_magerrmad_ind,
            assume_unique=True
        )
        select_ind2 = np.intersect1d(
            sorted_ngoodobj_ind,
            sorted_medsrcbkg_ind,
            assume_unique=True
        )

        best_ind = np.intersect1d(
            select_ind1,
            select_ind2,
            assume_unique=True
        )

        if len(best_ind) >= minframes:

            print('%sZ: selecting frames based on '
                  'detections, med mag err, med mag MAD, background' %
                  (datetime.utcnow().isoformat(), ))

            final_ind = best_ind[:minframes]

        elif len(select_ind2) >= minframes:

            print('WRN! %sZ: selecting frames based on '
                  'detections, background' %
                  (datetime.utcnow().isoformat(), ))

            final_ind = select_ind2[:minframes]

        elif len(select_ind1) >= minframes:

            print('WRN! %sZ: selecting frames based on '
                  'med mag err, med mag MAD' %
                  (datetime.utcnow().isoformat(), ))

            final_ind = select_ind1[:minframes]

        elif len(sorted_medsrcbkg_ind) >= minframes:

            print('WRN! %sZ: selecting frames based on '
                  'background only' %
                  (datetime.utcnow().isoformat(), ))

            final_ind = sorted_medsrcbkg_ind[:minframes]

        else:

            print('ERR! %sZ: not enough frames to select photref!.' %
                  (datetime.utcnow().isoformat(), ))
            return None, infodict

        # the final set of frames
        final_frames = selected_frames[final_ind]

    else:

        print('WRN! %sZ: not enough pre-selected frames to cut down '
              'to a minimum set, selecting those with lowest background...' %
              (datetime.utcnow().isoformat(), ))

        final_ind = (np.argsort(selected_medsrcbkg))[:minframes]
        final_frames = selected_frames[final_ind]

    # finally return the best frames for use with the photref convolution and
    # the infodict
    return final_frames.tolist(), infodict



def photref_convolution_worker(task):
    '''This is a parallel worker to convolve the photref frames to the astromref
    frame. Used by convolve_photref_frames below.

    task[0] -> the frame to convolve
    task[1] -> the frame to use as the convolution target
    task[2] -> the convolution target's registration info file
    task[3] -> the kernel specification for the convolution
    task[4] -> the output directory where to place the results

    '''

    frametoconvolve, targetframe, convregfile, kernelspec, outdir = task

    if not outdir:
        outfile = os.path.join(os.path.dirname(frametoconvolve),
                               'PHOTREF-%s' % os.path.basename(frametoconvolve))

    else:
        outfile = os.path.join(outdir,
                               'PHOTREF-%s' % os.path.basename(frametoconvolve))

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
        return frametoconvolve, outfile
    else:
        print('ERR! %sZ: photref convolution failed for %s' %
              (datetime.utcnow().isoformat(), frametoconvolve,))
        if os.path.exists(outfile):
            os.remove(outfile)
        return frametoconvolve, None


def convolve_photref_frames(photreflist,
                            targetframe,
                            convregfile,
                            kernelspec='b/4;i/4;d=4/4',
                            nworkers=16,
                            maxworkertasks=1000,
                            outdir=None):

    '''This convolves all photref frames in photreflist to the targetframe. See
    getref() in run_ficonv.py.

    '''

    # make a list of tasks

    tasks = [(x, targetframe, convregfile, kernelspec, outdir)
             for x in photreflist]

    print('%sZ: %s photref files to convolve to %s' %
          (datetime.utcnow().isoformat(), len(photreflist), targetframe))

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
    '''This combines all of the frames in framelist (a list of filenames) using
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

    '''

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
        return framelist, outfile
    else:
        print('ERR! %sZ: framelist combine failed!' %
              (datetime.utcnow().isoformat(),))
        if os.path.exists(outfile):
            os.remove(outfile)
        return framelist, None



def photometry_on_combined_photref(
        photref_frame,
        photref_sourcelist,  # this is the matched source list (.sourcelist)
        srclist_idcol='1',
        srclist_xycol='7,8',
        ccdgain=None,
        zeropoint=None,
        ccdexptime=None,
        apertures='2.95:7.0:6.0,3.35:7.0:6.0,3.95:7.0:6.0',
        outfile=None
):
    '''This runs fiphot in the special iphot mode on the combined photometric
    reference frame. See cmrawphot.sh for the correct commandline to use.

    '''

    # get the required header keywords from the FITS file
    header = imageutils.get_header_keyword_list(photref_frame,
                                                ['GAIN',
                                                 'GAIN1',
                                                 'GAIN2',
                                                 'EXPTIME'])

    # handle the gain and exptime parameters
    if not ccdgain:

        if 'GAIN1' in header and 'GAIN2' in header:
            ccdgain = (header['GAIN1'] + header['GAIN2'])/2.0
        elif 'GAIN' in header:
            ccdgain = header['GAIN']
        else:
            ccdgain = None

    if not ccdexptime:
        ccdexptime = header['EXPTIME'] if 'EXPTIME' in header else None

    if not (ccdgain or ccdexptime):
        print('%sZ: no GAIN or EXPTIME defined for %s' %
              (datetime.utcnow().isoformat(),
               photref_frame))
        return None

    # figure out the fitsbase from the fits sourcelist
    fitsbase = os.path.basename(photref_sourcelist).strip('.sourcelist')

    # handle the zeropoints
    if not zeropoint:

        # if the zeropoint isn't provided and if this is a HAT frame, the ccd
        # number will get us the zeropoint in the ZEROPOINTS dictionary
        frameinfo = FRAMEREGEX.findall(photref_sourcelist)
        if frameinfo:
            zeropoint = ZEROPOINTS[int(frameinfo[0][-1])]
        else:
            print('%sZ: no zeropoint magnitude defined for %s' %
                  (datetime.utcnow().isoformat(),
                   photref_frame))
            return None

    # figure out the output path
    if not outfile:
        outfile = os.path.abspath(photref_frame.strip('.fits.fz') + '.cmrawphot')

    # now assemble the command
    cmdtorun = COMBINEDREFPHOTCMD.format(
        photref=photref_frame,
        srclist=photref_sourcelist,
        srclist_idcol=srclist_idcol,
        srclist_xycol=srclist_xycol,
        ccdgain=ccdgain,
        zeropoint=zeropoint,
        exptime=ccdexptime,
        aperturestring=apertures,
        photrefbase=fitsbase,
        outfile=outfile
    )

    if DEBUG:
        print(cmdtorun)

    returncode = os.system(cmdtorun)

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



#################################
## CONVOLUTION AND SUBTRACTION ##
#################################

def subframe_convolution_worker(task):
    '''This is a parallel worker to convolve the combined photref frame to each
input frame, subtract them, and then return the subtracted frame. Used by
convolve_and_subtract_frames below.

    task[0] -> the frame to convolve
    task[1] -> the frame to use as the convolution target
    task[2] -> the convolution target's registration info file
    task[3] -> the kernel specification for the convolution
    task[4] -> the output directory where to place the results

    '''

    frametoconvolve, targetframe, convregfile, kernelspec, outdir = task

    if not outdir:
        outfile = os.path.join(os.path.dirname(frametoconvolve),
                               'subtracted-%s' % os.path.basename(frametoconvolve))
        outkernel = os.path.join(os.path.dirname(frametoconvolve),
                                 '%s-kernel' % os.path.basename(frametoconvolve))

    else:
        outfile = os.path.join(outdir,
                               'subtracted-%s' % os.path.basename(frametoconvolve))
        outkernel = os.path.join(outdir,
                                 '%s-kernel' % os.path.basename(frametoconvolve))

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
        return frametoconvolve, outfile
    else:
        print('ERR! %sZ: convolution and subtraction failed for %s' %
              (datetime.utcnow().isoformat(), frametoconvolve,))
        if os.path.exists(outfile):
            os.remove(outfile)
        return frametoconvolve, None


def convolve_and_subtract_frames(fitsdir,
                                 combinedphotref,
                                 photrefregfile,
                                 fitsglob='*-xtrns.fits',
                                 kernelspec='b/4;i/4;d=4/4',
                                 nworkers=16,
                                 maxworkertasks=1000,
                                 outdir=None):
    '''
    This convolves the photometric reference to each frame, using the specified
    kernel, then subtracts the frame from the photometric reference to produce
    the subtracted frames.

    '''

    # find all the files to process
    transframelist = glob.glob(os.path.join(os.path.abspath(fitsdir),
                                            fitsglob))

    # make a list of tasks
    tasks = [(x, combinedphotref, photrefregfile, kernelspec, outdir)
             for x in transframelist]

    print('%sZ: %s frames to convolve to %s and subtract' %
          (datetime.utcnow().isoformat(), len(transframelist), combinedphotref))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(subframe_convolution_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



def subframe_photometry_worker(task):
    '''This runs the special version of fiphot in subtracted image mode to
    calculate the ISM magnitudes.

    task[0] -> subtracted frame FITS
    task[1] -> photometric reference frame .cmrawphot file
    task[2] -> disjoint radius
    task[3] -> subtracted frame kernel file
    task[4] -> subtracted frame itrans file
    task[5] -> subtracted frame xysdk file
    task[6] -> output directory

    '''

    # get the info out of the task
    (subframe, photrefrawphot, disjointraid,
     subframekernel, subframeitrans, subframexysdk,
     outdir) = task


    # get the CCD info out of the subframe
    header = imageutils.get_header_keyword_list(photref_frame,
                                                ['GAIN',
                                                 'GAIN1',
                                                 'GAIN2',
                                                 'EXPTIME'])

    # handle the gain and exptime parameters
    if not ccdgain:

        if 'GAIN1' in header and 'GAIN2' in header:
            ccdgain = (header['GAIN1'] + header['GAIN2'])/2.0
        elif 'GAIN' in header:
            ccdgain = header['GAIN']
        else:
            ccdgain = None

    if not ccdexptime:
        ccdexptime = header['EXPTIME'] if 'EXPTIME' in header else None

    if not (ccdgain or ccdexptime):
        print('%sZ: no GAIN or EXPTIME defined for %s' %
              (datetime.utcnow().isoformat(),
               photref_frame))
        return None



    zeropoint, exptime, ccdgain = ccdinfo

    frameinfo = re.findall(os.path.basename(subframe))
    frameiphot = '%s-%s_%s.iphot' % (frameinfo[0][0],
                                     frameinfo[0][1],
                                     frameinfo[0][2])

    if outdir:
        outfile = os.path.join(os.path.abspath(outdir),
                               frameiphot)
    else:
        outfile = os.path.join(os.path.abspath(os.path.dirname(subframe)),
                               frameiphot)


    cmdtorun = SUBFRAMEPHOTCMD.format(
        subtractedframe=subframe,
        photrefrawphot=photrefrawphot,
        zeropoint=zeropoint,
        exptime=exptime,
        ccdgain=ccdgain,
        disjointradius=disjointrad,
        subtractedkernel=subframekernel,
        subtracteditrans=subframeitrans,
        subtractedxysdk=subframexysdk,
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



def photometry_on_subtracted_frames(subframedir,
                                    photrefrawphot,
                                    subframeglob='subtracted-*.fits',
                                    subframekerneldir=None,
                                    subframeitransdir=None,
                                    subframexysdkdir=None,
                                    photdisjointradius=2,
                                    zeropoint=None,
                                    exptime=None,
                                    ccdgain=None,
                                    outdir=None):

    '''
    This runs photometry on the subtracted frames and finally produces the ISM
    magnitudes.

    See run_iphot.py and IMG-3-PHOT_st5.sh for what this is supposed to do.

    '''

    # find all the subtracted frames
    subframelist = glob.glob(os.path.join(os.path.abspath(subframedir),
                                          subframeglob))


    # figure out the CCD parameters


    # we need to find the accompanying kernel, itrans, and xysdk files for each
    # subtracted frame for the tasks list
    tasks = []

    if not subframekerneldir:
        subframekerneldir = subframedir
    if not subframeitransdir:
        subframeitransdir = subframedir
    if not subframexysdkdir:
        subframeitransdir = subframedir

    # find matching kernel, itrans, and xysdk files for each subtracted frame
    for subframe in subframelist:

        frameinfo = re.findall(os.path.basename(subframe))
        kernel = '%s-%s_%s-xtrns.fits-kernel' % (frameinfo[0][0],
                                                 frameinfo[0][1],
                                                 frameinfo[0][2])
        itrans = '%s-%s_%s.itrans' % (frameinfo[0][0],
                                      frameinfo[0][1],
                                      frameinfo[0][2])

        xysdk = '%s-%s_%s.xysdk' % (frameinfo[0][0],
                                    frameinfo[0][1],
                                    frameinfo[0][2])

        if (os.path.exists(os.path.join(subframekerneldir, kernel)) and
            os.path.exists(os.path.join(subframekerneldir, itrans)) and
            os.path.exists(os.path.join(subframekerneldir, xysdk))):

            tasks.append((os.path.abspath(subframe),
                          os.path.abspath(photrefrawphot),
                          (


###########################
## LIGHTCURVE COLLECTION ##
###########################

def parallel_collect_lightcurves(photdir):
    '''
    This collects all .iphot objects into lightcurves.

    '''
