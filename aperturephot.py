#!/usr/bin/env python

"""
aperturephot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2014

Contains aperture photometry routines for HATPI. Needs reduced frames.

The usual sequence is:

1. run parallel_extract_sources on all frames with threshold ~ 10000 to get
   bright stars for astrometry.

2. run parallel_anet to get WCS headers for all frames.

3. run make_fov_catalog to get a FOV source catalog for the field.

4. run reform_fov_catalog to cut this down to the columns needed for magfit
   only.

5. run parallel_fitsdir_photometry for photometry on all frames

6. run get_magfit_frames to select a single magfit photometry reference and set
   up per-CCD work directories, symlinks, etc. for the next steps.

7. run make_magfit_config to generate magfit config files for
   MagnitudeFitting.py

8. run make_fiphot_list to make lists of fiphot files for each CCD.

9. run MagnitudeFitting.py in single reference mode.

10. run do_masterphotref.py to get the master mag fit reference.

11. run MagnitudeFitting.py in master reference mode.

12. run parallel_collect_lightcurves to collect all lightcurves into .rlc files.

13. run serial_run_epd or parallel_run_epd to do EPD on all LCs.

14. run parallel_lc_statistics to collect stats on .epdlc files.

15. run choose_tfa_template to choose TFA template stars using the .epdlc stats.

16. run parallel_run_tfa for TFA to get .tfalc files (and .tfalc.TF{1,2,3}
    files).

17. run parallel_lc_statistics to collect stats on .tfalc files.

18. run parallel_bin_lightcurves to bin LCs to desired time-bins.

19. run parallel_binnedlc_statistics to collect stats for the binned LCs.

20. run plot_stats_file to make MAD vs. mag plots for all unbinned and binned
    LCs.

21. run plot_magrms_comparison to compare the mag-RMS relation for various CCDs.

22. run plot_ismphot_comparison to compare against ISM photometry statistics for
    the same field (requires common stars).
"""

#############
## IMPORTS ##
#############

import os, os.path, glob, sys
import multiprocessing as mp

try:
    import subprocess32 as subprocess
except:
    import subprocess

import shlex
from datetime import datetime
import re, json, shutil, random
try:
    import cPickle as pickle
except:
    import pickle

import sqlite3, gzip

import numpy as np, pandas as pd

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

import astropy.io.fits as pyfits

import imageutils
from imageutils import get_header_keyword, read_fits, extract_img_background

from shared_variables import FITS_TAIL
import shared_variables as sv

from astropy.table import Table
from astropy.io import fits

# get fiphot binary reader
try:
    from HATpipepy.Common.BinPhot import read_fiphot
    HAVEBINPHOT = True
except:
    print("can't import binary fiphot reading functions from "
          "HATpipe, binary fiphot files will be unreadable!")
    HAVEBINPHOT = False

########################
## USEFUL DEFINITIONS ##
########################

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

# command string to use gaia2read for specified FOV
GAIADR2READCMD = ("gaia2read -r {ra:f} -d {dec:f} -s {boxlen:f} "
                  "--mR {brightrmag:f} --MR {faintrmag:f} "
                  "--xieta-coords --header --extra "
                  "--idrequest {idrequest:s} -o {outfile}")

# command string to do a 2massread for a specified FOV
TWOMASSREADCMD = ("2massread -r {ra:f} -d {dec:f} -s {boxlen:f} "
                  "--cat {catalogpath} -mr {brightrmag:f} -Mr {faintrmag:f} "
                  "--xieta-coords {ra:f} {dec:f} -o {outfile}")

# command string to do a 2massread for a specified FOV
UCAC4READCMD = ("ucac4read -r {ra:f} -d {dec:f} -s {boxlen:f} "
                  "--cat {catalogpath} -mr {brightrmag:f} -Mr {faintrmag:f} "
                  "-o {outfile}")

# locations of catalogs
CATALOGS = {
    '2MASS':{'cmd':TWOMASSREADCMD,
             'path':sv.TWOMASSPATH},
    'UCAC4':{'cmd':UCAC4READCMD,
             'path':sv.UCAC4PATH},
    'GAIADR2': {'cmd':GAIADR2READCMD,
                'path':sv.GAIADR2PATH}
    }

# command string to run fistar
# parameters:
# {frame}
# {extractedlist}
# {ccdgain}
# {zeropoint}
# {exptime}
# {fluxthreshold}
# {ccdsection}
FISTARCMD = ("{fistarexec} -i {frame} -o {extractedlist} "
             "--model elliptic --iterations symmetric=4,general=2 "
             "--algorithm uplink --format id,x,y,bg,amp,s,d,k,flux,s/n "
             "-g {ccdgain} --mag-flux {zeropoint},{exptime} --sort flux "
             "--flux-threshold {fluxthreshold} --section {ccdsection} "
             "--comment")

# command string to run transformation from RA/Dec in 2MASS catalogs to X/Y in
# FITs image to eventually run photometry at those locations using the
# transformations noted in each frame's .wcs file. we do the transform, then
# remove any objects outside the [0:2048, 0:2048] box for the CCD. we then use
# the resulting source list as input to fiphot. the parameters are:

# {transformer}:       'anrd2xy' or similar, coord -> pix converter executable
# {framewcsfile}:      tWCS transformation file associated with frame
# {catalogsourcelist}: 2MASS catalog file associated with camera FOV
# {outfile}:           temporary output file
TRANSFORMCMD = ("{transformer} -w {framewcsfile} "
                "-o {outputfile} "
                "-c 2,3 "
                "{catalogsourcelist} ")
WCSRD2XYCMD = ( "wcs-rd2xy -w {framewcsfile} -o {outputfile} "
               "-i {rdlsfile} -R 'ra' -D 'dec' " )

# fiphot command string to run fiphot. requires a sourcelist obtained from
# running TRANSFORMCMD and removing objects outside the CCD. the parameters are:

# {fits}:         name of input FITS frame
# {sourcelist}:   name of the source list file
# {zeropoint}:    zeropoint magnitude from ZEROPOINTS above
# {xycols}:       comma-separated 1-indexed column numbers of x and y coords
# {ccdgain}:      gain of the CCD
# {ccdexptime}:   exposure time of the CCD in seconds
# {aperturelist}: aperture list in the following format (all units are pixels):
#                 aper1rad:sky1inner:sky1rad,...,aperNrad,skyNinner,skyNrad
# {fitsbase}:     the FITS base filename without any extensions
#                 e.g. for 1-377741e_5.fits, this is 1-377741e_5
#                 (used for later collection of fiphot files into an LC)
# {outfile}:      name of the output .phot file (binary format)
# {format}:       for text output, e.g.,'ISXY,BbMms', see fiphot manpage
#
# single-background: a fiphot I/O parameter for which the number sets different
# aspects of the output formatting for the .fiphot file that is produced. It
# is a HATpipebin-specific dependency; the public `fitsh` downloadable from
# Andras' webpage does not have this parameter. See `fiphot-io.c` and
# `fiphot.c` in the HATpipebin source. The corresponding `singlebg` parameter
# is a switch used in `write_comment`, `write_photometry_text`,
# `output_phot_columns`, and similar formatting functions.
#
FIPHOTCMD = ("fiphot --input {fits} --input-list {sourcelist} "
             "--col-id 1 --col-xy {xycols} --gain {ccdgain:f} "
             "--mag-flux {zeropoint:f},{ccdexptime:f} "
             "--apertures {aperturelist} "
             "--sky-fit 'mode,sigma=3,iterations=2' --disjoint-radius 2 "
             "--serial {fitsbase} "
             "--format {formatstr} --nan-string 'NaN' "
             "--aperture-mask-ignore 'saturated' --comment '--comment' "
             "--single-background 3 {binaryout} --output {outfile} -k")


# MagnitudeFitting.py commandline
# {magfitexec}:         path to the executable to run for magfit (usually
#                       MagnitudeFitting.py)
# {network}:            HATNet or HATSouth
# {fit_type}:           single for single reference mag fit, master for
#                       master mag fit
# {sphotref_frame}:     FITS to use as the single photometric reference
# {sphotref_phot} :     fiphot file for the single photometric reference
# {nprocs}        :     number of processors to use
# {magfit_config_file}: path to the magfit config file
# {magfit_frame_list}:  path to the magfit frame list produced by
#                       get_magfit_frames
MAGFITCMD = ("python {magfitexec} {network} {fit_type} "
             "{sphotref_frame} {sphotref_phot} "
             "-p {nprocs} --config-file={magfit_config_file} "
             "--manual-frame-list={magfit_frame_list} --stat")

# command to generate master photometric reference
# {mphotrefexec}:        path to executable to run (usually do_masterphotref.py)
# {network}:             HATNet or HATSouth
# {sphotref_frame}:      FITS used for the single photometric reference
# {fiphot_list}:         list of fiphot files containing all photometry
# {magfit_config_file}:  path to the magfit config file
MPHOTREFCMD = ("python {mphotrefexec} {network} {sphotref_frame} "
               "--manual-frame-list={fiphot_list} "
               "--config-file={magfit_config_file} --nostat")



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


###############################
## ANET ASTROMETRY FUNCTIONS ##
###############################


def reform_fistars(fistardir,
                   fistarglob='1-*_?.fistar',
                   linestokeep=300,
                   outpostfix='astrometry'):
    """
    This truncates all fistars in the directory fistardir to linestokeep
    lines. This is useful for astrometry since the fistar files we produce are
    sorted by decreasing flux, and we only only need a couple of thousand bright
    sources to do astrometry with anet.

    Mostly so we don't need to do source extraction over again
    """

    fistars = glob.glob(os.path.join(os.path.abspath(fistardir),
                                     fistarglob))

    for fistar in fistars:

        inf = open(fistar,'rb')
        outfname = os.path.join(os.path.dirname(fistar),
                                '%s-%s' % (os.path.basename(fistar),
                                           outpostfix))

        outf = open(outfname, 'wb')

        for ind, line in enumerate(inf):

            if ind < linestokeep:
                outf.write(line.encode('utf-8'))

        print('%s -> %s' % (fistar, outfname))

        outf.close()
        inf.close()


def fistarfile_to_xy(fistarfile):
    """
    Takes a single fistar file and convert it to a binary fits table of the
    source positions. The latter is readable by astrometry.net.
    """

    if not isinstance(fistarfile, str):
        raise AssertionError('called fistarfile_to_xy on not a path')

    if sys.version_info[0] == 2:
        # used for python 2.X
        df = pd.read_csv(fistarfile, comment='#',
                         names=['ident', 'x', 'y', 'bg', 'amp', 's', 'd', 'k',
                                'flux', 's/n'],
                         delimiter=r"\s*", engine='python')
    elif sys.version_info[0] == 3:
        # used for python 3.X
        df = pd.read_csv(fistarfile, comment='#',
                         names=['ident', 'x', 'y', 'bg', 'amp', 's', 'd', 'k',
                                'flux', 's/n'],
                         delim_whitespace=True)

    if not len(df) > 1:
        print('skipping %s, did not get any sources' % fistarfile)
        return 0

    else:
        col1 = fits.Column(name='ximage', format='D', array=np.array(df['x']))
        col2 = fits.Column(name='yimage', format='D', array=np.array(df['y']))
        coldefs = fits.ColDefs([col1, col2])
        hdu = fits.BinTableHDU.from_columns(coldefs)

        outfname = fistarfile.replace('.fistar','.fistar-fits-xy')
        hdu.writeto(outfname, overwrite=True)

        print('%s -> %s' % (fistarfile, outfname))


def fistardir_to_xy(fistardir, fistarglob='1-*_?.fistar'):
    """
    Convert a directory of fistar outputs to binary fits table of x,y source
    positions. The latter is readable by astrometry.net.
    """

    fistars = glob.glob(
        os.path.join(os.path.abspath(fistardir), fistarglob)
    )

    for fistar in fistars:
        if not os.path.exists(fistar.replace('.fistar','.fistar-fits-xy')):
            fistarfile_to_xy(fistar)
        else:
            print('found {:s}, continue.'.format(
            fistar.replace('.fistar','.fistar-fits-xy'))
            )



def astrometrydotnet_solve_frame(srclist,
                                 wcsout,
                                 ra,
                                 dec,
                                 radius=30,
                                 scalelow=1,
                                 scalehigh=30,
                                 scaleunits='arcsecperpix',
                                 tweakorder=6,
                                 nobjs=200,
                                 xpix=2048,
                                 ypix=2048,
                                 xcolname='ximage',
                                 ycolname='yimage',
                                 useimagenotfistar=False,
                                 downsample=4,
                                 pixelerror=1,
                                 uniformize=10
                                ):
    """
    This uses astrometry.net to solve frame astrometry. This is the
    free version of anet_solve_frame.

    Uses the frame extracted sources (.fistar-fits-xy file, see
    aperturephot.fistar_to_xy) and returns a .wcs file containing the
    astrometric transformation between frame x,y and RA/DEC.

    Optionally, if `useimagenotfistar` is true, uses the fits image
    corresponding to the frame extracted sources and the astrometry.net
    in-built source extraction to produce the solution, along with sick
    constellation plots.

    Example astrometry.net frame-solve command (using the fits table of x,y
    positions):

        solve-field --ra 274.5 --dec 58.0 --radius 30 --scale-low 1
        --scale-high 30 --scale-units arcsecperpix --tweak-order 2
        --wcs /dirname/tess2019135090826-4-2-0016_cal_img_bkgdsub.wcs
        --overwrite --objs 200 -w 2048 -e 2048 --x-column ximage --y-column yimage
        /dirname/tess2019135090826-4-2-0016_cal_img_bkgdsub.fistar-fits-xy

    For astrometry.net to work, you need to install it, and get all the index
    files. See http://astrometry.net/doc/readme.html.

    If you care about your astrometric projection's precision at the < 1 pixel
    level, you will likely need to tweak downsample, pixelerror, and
    uniformize. `wcsqualityassurance.py` implements some relevant tests.
    """

    if useimagenotfistar:

        ASTROMETRYDOTNETCMD = (
            "solve-field --ra {ra} --dec {dec} --radius {radius} "
            "--scale-low {scalelow} --scale-high {scalehigh} "
            "--scale-units {scaleunits} --tweak-order {tweakorder} "
            "--wcs {wcsout} --downsample {downsample} "
            "--overwrite --fits-image --no-verify "
            "--pixel-error {pixelerror} "
            "--uniformize {uniformize} "
            "{srcimage}"
        )

        astrometrycmd = ASTROMETRYDOTNETCMD.format(
            ra=ra,
            dec=dec,
            radius=radius,
            scalelow=scalelow,
            scalehigh=scalehigh,
            scaleunits=scaleunits,
            tweakorder=tweakorder,
            downsample=downsample,
            wcsout=wcsout,
            pixelerror=pixelerror,
            uniformize=uniformize,
            srcimage=srclist.replace('.fistar-fits-xy','.fits')
        )


    else:

        ASTROMETRYDOTNETCMD = (
            "solve-field --ra {ra} --dec {dec} --radius {radius} "
            "--scale-low {scalelow} --scale-high {scalehigh} "
            "--scale-units {scaleunits} --tweak-order {tweakorder} "
            "--wcs {wcsout} "
            "--overwrite --objs {nobjs} --width {xpix} --height {ypix} "
            " --x-column {xcolname} --y-column {ycolname} --no-plot "
            "{srclist}"

        )

        astrometrycmd = ASTROMETRYDOTNETCMD.format(
            ra=ra,
            dec=dec,
            radius=radius,
            scalelow=scalelow,
            scalehigh=scalehigh,
            scaleunits=scaleunits,
            tweakorder=tweakorder,
            xpix=xpix,
            ypix=ypix,
            nobjs=nobjs,
            xcolname=xcolname,
            ycolname=ycolname,
            wcsout=wcsout,
            srclist=srclist
        )

    if DEBUG:
        print(astrometrycmd)

    # execute the anet shell command
    anetproc = subprocess.Popen(shlex.split(astrometrycmd),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

    # get results
    anet_stdout, anet_stderr = anetproc.communicate()

    if DEBUG:
        for l in anet_stdout.decode('utf-8').split('\n'):
            print(l)
        for l in anet_stderr.decode('utf-8').split('\n'):
            print(l)

    # get results if succeeded, log outcome, and return path of outfile
    if (anetproc.returncode == 0 and
        os.path.exists(os.path.abspath(wcsout)) and
        os.stat(os.path.abspath(wcsout)).st_size > 0):

        print('%sZ: astrometrydotnet %s generated for frame sourcelist %s' %
              (datetime.utcnow().isoformat(),
               os.path.abspath(wcsout), os.path.abspath(srclist)))

        return os.path.abspath(wcsout)

    # if astrometry did not succeed, complain and remove the zero-size wcs file
    # anet inexplicably makes anyway
    else:

        print('{:s}Z: astrometrydotnet {:s} failed for frame sourcelist {:s}! '
              .format(datetime.utcnow().isoformat(),
                      os.path.abspath(wcsout),
                      os.path.abspath(srclist)
              )+
              'Error was: {:s}'.format(repr(anet_stderr))
        )

        # remove the broken wcs if astrometry failed
        if os.path.exists(os.path.abspath(wcsout)):
            os.remove(os.path.abspath(wcsout))

        return None


def anet_solve_frame(srclist,
                     wcsout,
                     ra,
                     dec,
                     infofromframe=False,
                     width=13,
                     tweak=6,
                     radius=13,
                     xpix=2048,
                     ypix=2048,
                     cols=(2,3),
                     scale=None,
                     usescalenotwidth=False):
    """
    This uses anet to solve frame astrometry.

    Uses the frame extracted sources (.fistar file) and returns a .wcs file
    containing the astrometric transformation between frame x,y and RA/DEC.

    Example anet command:

    anet --ra 60. --dec -22.5 --radius 4 --width 13 --tweak 6 --cols 2,3 1-383272f_5.fistar --wcs 1-383272f_5.wcs

    assuming an ~/.anetrc file with the following contents:

    xsize = 2048  # the width of the frame in pixels
    ysize = 2048  # the height of the frame in pixels
    tweak = 3     # the order of polynomial fit to frame distortion
    xcol = 2      # the column to be used for x coord (1-indexed)
    ycol = 3      # the column to be used for y coord (1-indexed)
    verify = 1    # number of verify iterations to run
    log = 1       # set to 1 to log operations
    indexpath = /P/HP0/CAT/ANET_INDEX/ucac4_2014/   # path to the indexes

    otherwise, we'll need to provide these as kwargs to the anet executable.

    If usescalenotwidth, instead executes

    anet --ra 60. --dec -22 --radius 12 --scale 21.1 -s 2048,2048 --tweak 6 --cols 2,3 foo.fistar --wcs foo.wcs

    The input sourcelist can come from fistar, with a fluxthreshold set to 10000
    to just get the bright stars. This makes anet way faster.
    """

    if infofromframe:

        # find the frame
        srcframe = os.path.basename(srclist)

        srcframe = os.path.splitext(srcframe)[0] + '.fits'
        srcframepath = os.path.join(os.path.dirname(srclist), srcframe)

        srcframefz = os.path.splitext(srcframe)[0] + FITS_TAIL
        srcframefzpath = os.path.join(os.path.dirname(srclist), srcframefz)

        # get the RA, DEC, and FOV header keywords
        if os.path.exists(srcframepath):

            ra = get_header_keyword(srcframepath,'rac')
            dec = get_header_keyword(srcframepath,'decc')
            fov = get_header_keyword(srcframepath,'fov')
            xpix = get_header_keyword(srcframepath,'naxis1')
            ypix = get_header_keyword(srcframepath,'naxis2')

            if fov is not None:
                width = fov

            ra = ra*360.0/24.0

        elif os.path.exists(srcframefzpath):

            # ext 1 is the header for the fpacked image
            ra = get_header_keyword(srcframefzpath,'rac',ext=1)
            dec = get_header_keyword(srcframefzpath,'decc',ext=1)
            fov = get_header_keyword(srcframefzpath,'fov',ext=1)
            xpix = get_header_keyword(srcframefzpath,'naxis1',ext=1)
            ypix = get_header_keyword(srcframefzpath,'naxis2',ext=1)

            if fov is not None:
                width = fov

            ra = ra*360.0/24.0

    if usescalenotwidth:

        ANETCMDSTR = ("anet -r {ra} -d {dec} --scale {scale} "
                      "--tweak {tweak} --radius {radius} -s {xpix},{ypix} "
                      "--cols {colx},{coly} --wcs {outwcsfile} {sourcelist}")

        anetcmd = ANETCMDSTR.format(ra=ra,
                                    dec=dec,
                                    scale=scale,
                                    tweak=tweak,
                                    radius=radius,
                                    xpix=xpix,
                                    ypix=ypix,
                                    colx=cols[0],
                                    coly=cols[1],
                                    outwcsfile=wcsout,
                                    sourcelist=srclist)

    else:

        ANETCMDSTR = ("anet -r {ra} -d {dec} -w {width} "
                      "--tweak {tweak} --radius {radius} -s {xpix},{ypix} "
                      "--cols {colx},{coly} --wcs {outwcsfile} {sourcelist}")

        anetcmd = ANETCMDSTR.format(ra=ra,
                                    dec=dec,
                                    width=width,
                                    tweak=tweak,
                                    radius=radius,
                                    xpix=xpix,
                                    ypix=ypix,
                                    colx=cols[0],
                                    coly=cols[1],
                                    outwcsfile=wcsout,
                                    sourcelist=srclist)

    if DEBUG:
        print(anetcmd)

    # execute the anet shell command
    anetproc = subprocess.Popen(shlex.split(anetcmd),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

    # get results
    anet_stdout, anet_stderr = anetproc.communicate()

    # get results if succeeded, log outcome, and return path of outfile
    if (anetproc.returncode == 0 and
        os.path.exists(os.path.abspath(wcsout)) and
        os.stat(os.path.abspath(wcsout)).st_size > 0):

        print('%sZ: anet WCS %s generated for frame sourcelist %s' %
              (datetime.utcnow().isoformat(),
               os.path.abspath(wcsout), os.path.abspath(srclist)))

        return os.path.abspath(wcsout)

    # if astrometry did not succeed, complain and remove the zero-size wcs file
    # anet inexplicably makes anyway
    else:

        print('%sZ: anet WCS %s failed for frame sourcelist %s! Error was: %s' %
              (datetime.utcnow().isoformat(),
               os.path.abspath(wcsout),
               os.path.abspath(srclist),
               anet_stderr))

        # remove the broken wcs if astrometry failed
        if os.path.exists(os.path.abspath(wcsout)):
            os.remove(os.path.abspath(wcsout))

        return None


def parallel_anet_worker(task):
    """
    This expands the task arg into the args and kwargs necessary for
    anet_solve_frame.
    """

    return (
        task[0],
        anet_solve_frame(
            task[0], task[1], task[2], task[3], **task[4]
        )
    )


def parallel_astrometrydotnet_worker(task):
    """
    This expands the task arg into the args and kwargs necessary for
    astrometrydotnet_solve_frame.
    """

    return (
        task[0],
        astrometrydotnet_solve_frame(
            task[0], task[1], task[2], task[3], **task[4]
        )
    )


def parallel_anet(srclistdir,
                  outdir,
                  ra, dec,
                  fistarglob='?-*_?.fistar-astrometry',
                  nworkers=16,
                  maxtasksperworker=1000,
                  infofromframe=True,
                  width=13,
                  tweak=6,
                  radius=13,
                  xpix=2048,
                  ypix=2048,
                  cols=(2,3),
                  overwrite=True):
    """
    This does parallel anet astrometry for all frames in srclistdir and
    generates their wcs files.
    """

    # get a list of all fits files in the directory
    fistarlist = glob.glob(os.path.join(srclistdir, fistarglob))
    if not overwrite:
        existing = glob.glob(
            os.path.join(fitsdir, fistarglob.replace('.fistar', '.wcs'))
        )
        requested = list(map(os.path.basename, fistarlist))
        requested = [r.replace('.fistar','') for r in requested]
        alreadyexists = list(map(os.path.basename, existing))
        alreadyexists = [ae.replace('.wcs','') for ae in alreadyexists]

        setdiff = np.setdiff1d(requested, alreadyexists)
        if len(setdiff) == 0:
            print("%sZ: astrometry already done for all .fistar files..." %
                  datetime.utcnow().isoformat())
            return
        fistarlist = [os.path.join(fitsdir, sd+'.fistar') for sd in setdiff]


    print('%sZ: found %s fistar files in %s, starting astrometry...' %
          (datetime.utcnow().isoformat(),
           len(fistarlist), srclistdir))

    if outdir and not os.path.exists(outdir):

        print('%sZ: making new output directory %s' %
              (datetime.utcnow().isoformat(),
               outdir))
        os.mkdir(outdir)

    # get the files for which astrometry hasn't already been done
    fistarlist = check_files(fistarlist,
                            'astrometry',
                            outdir,
                            intailstr='.fistar',
                            outtailstr='.wcs',
                            skipifpartial=True)
    if type(fistarlist) == int:
        if fistarlist == -1:
            return -1

    pool = mp.Pool(nworkers, maxtasksperchild=maxtasksperworker)

    inpostfix = os.path.splitext(fistarglob)[-1]

    tasks = [
        [x, os.path.join(
                outdir, os.path.basename(
                    x.replace(inpostfix, '.wcs')
                    )
                ),
         ra, dec, {'width':width,
                   'tweak':tweak,
                   'radius':radius,
                   'xpix':xpix,
                   'ypix':ypix,
                   'cols':cols,
                   'infofromframe':infofromframe}]
        for x in fistarlist
        ]

    # fire up the pool of workers
    results = pool.map(parallel_anet_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict


def parallel_astrometrydotnet(
        srclistdir,
        outdir,
        ra, dec,
        fistarfitsxyglob='tess2019135090826-4-2-0016_cal_img_bkgdsub.fistar-fits-xy',
        nworkers=10,
        maxtasksperworker=1000,
        radius=30,
        scalelow=1,
        scalehigh=30,
        scaleunits='arcsecperpix',
        tweakorder=6,
        nobjs=200,
        xpix=2048,
        ypix=2048,
        xcolname='ximage',
        ycolname='yimage',
        useimagenotfistar=False,
        downsample=4,
        pixelerror=1,
        uniformize=10
    ):

    """
    Uses astrometrydotnet_solve_frame to do parallel astrometry for all frames
    in srclistdir and generate their wcs files.
    """

    # get a list of all fits files in the directory
    if useimagenotfistar:
        fistarfitsxylist = glob.glob(
            os.path.join(srclistdir,
                         fistarfitsxyglob.replace('fistar-fits-xy','fits'))
        )
        intailstr = '.fits'
    else:
        fistarfitsxylist = glob.glob(
            os.path.join(srclistdir, fistarfitsxyglob)
        )
        intailstr = '.fistar-fits-xy'

    print('%sZ: found %s fistar files in %s, starting astrometry...' %
          (datetime.utcnow().isoformat(),
           len(fistarfitsxylist), srclistdir))

    if outdir and not os.path.exists(outdir):

        print('%sZ: making new output directory %s' %
              (datetime.utcnow().isoformat(),
               outdir))
        os.mkdir(outdir)

    # get the files for which astrometry hasn't already been done
    fistarfitsxylist = check_files(fistarfitsxylist, 'astrometry', outdir,
                                   intailstr=intailstr,
                                   outtailstr='.wcs', skipifpartial=False)
    if type(fistarfitsxylist) == int:
        if fistarfitsxylist == -1:
            return -1

    inpostfix = os.path.splitext(fistarfitsxylist[0])[-1]

    tasks = [
        [x, os.path.join(
                outdir, os.path.basename(
                    x.replace(inpostfix, '.wcs')
                    )
                ),
         ra, dec, {'radius':radius,
                   'scalelow':scalelow,
                   'scalehigh':scalehigh,
                   'scaleunits':scaleunits,
                   'tweakorder':tweakorder,
                   'nobjs':nobjs,
                   'xpix':xpix,
                   'ypix':ypix,
                   'xcolname':xcolname,
                   'ycolname':ycolname,
                   'useimagenotfistar':useimagenotfistar,
                   'pixelerror':pixelerror,
                   'uniformize':uniformize
                   }
        ]
        for x in fistarfitsxylist
    ]

    pool = mp.Pool(nworkers, maxtasksperchild=maxtasksperworker)

    # fire up the pool of workers
    results = pool.map(parallel_astrometrydotnet_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict


def parallel_anet_list(srclistlist,
                       outdir,
                       ra, dec,
                       fistarglob='?-*_?.fistar-astrometry',
                       nworkers=16,
                       maxtasksperworker=1000,
                       infofromframe=True,
                       width=13,
                       tweak=6,
                       radius=13,
                       xpix=2048,
                       ypix=2048,
                       cols=(2,3)):
    """
    This runs anet on a list of frames in parallel.
    """

    # get a list of all fits files in the directory
    fistarlist = [x for x in srclistlist if os.path.exists(x)]

    print('%sZ: found %s fistar files, starting astrometry...' %
          (datetime.utcnow().isoformat(), len(fistarlist)))

    if outdir and not os.path.exists(outdir):

        print('%sZ: making new output directory %s' %
              (datetime.utcnow().isoformat(),
               outdir))
        os.mkdir(outdir)

    pool = mp.Pool(nworkers, maxtasksperchild=maxtasksperworker)

    inpostfix = os.path.splitext(fistarglob)[-1]

    tasks = [
        [x, os.path.join(
                outdir, os.path.basename(
                    x.replace(inpostfix, '.wcs')
                    )
                ),
         ra, dec, {'width':width,
                   'tweak':tweak,
                   'radius':radius,
                   'xpix':xpix,
                   'ypix':ypix,
                   'cols':cols,
                   'infofromframe':infofromframe}]
        for x in fistarlist
        ]

    # fire up the pool of workers
    results = pool.map(parallel_anet_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict



##########################
## PHOTOMETRY FUNCTIONS ##
##########################

def make_fov_catalog(ra=None, dec=None, size=None,
                     brightrmag=sv.FIELDCAT_BRIGHT,
                     faintrmag=sv.FIELDCAT_FAINT,
                     fits=None,
                     outfile=None,
                     outdir=None,
                     catalog='2MASS',
                     catalogpath=None,
                     columns=None,
                     observatory='hatpi',
                     gaiaidrequest='HAT'):
    """
    This function gets all the sources in the field of view of the frame, given
    its central pointing coordinates and plate-scale from either 2MASS or
    UCAC4. Makes a catalog file that can then be used as input to project
    catalog (ra,dec) to frame (x,y).

    if ra, dec, size are None, fits must not be None. fits is the filename of
    the FITS file to get the center RA, DEC, and platescale values from.

    Kwargs:
        ra, dec (float): field center, in degrees

        size (float): size of box, in degrees

        brightrmag (float): bright cutoff from catalog. If 2MASS is used,
        "rmag" is 2MASS r. If GAIADR2 is used, it's Gaia R.

        faintrmag (float): faint cutoff from catalog

        fits (str): path to fits file containing center RA, DEC, and
        platescale (see preamble).

        outfile (str): path to write the output catalog

        catalog (str): 'UCAC4', '2MASS', or 'GAIADR2'. You should usually use
        GAIADR2.

        gaiaidrequest (str): if catalog is GAIADR2, then you can request
        either "GAIA", "HAT", or "TMASS" identifiers. These are collected from
        a crossmatch; if you request "HAT" identifiers, the output catalog may
        not include exclusively HAT-XXX-XXXXXXX ID's (there may also be some
        GAIA ID's). The default is set to "HAT".

    Returns:

        path of the catalog file produced.
    """

    if ra and dec and size:

        catra, catdec, catbox = ra, dec, size

    elif fits:

        frame, hdr = read_fits(fits)
        catbox = sv.FIELDCAT_FOV

        if observatory=='hatpi':
            catra = float(hdr['RACA'])   # RA [DECIMAL hr] (averaged field center)
            catdec = float(hdr['DECCA']) # Dec [decimal deg] (averaged field center)
            print('WRN! %sZ: converting decimal hour RA to decimal degree' %
                  (datetime.utcnow().isoformat()))
            from astropy.coordinates import Angle
            import astropy.units as units
            tempra = Angle(str(catra)+'h')
            catra = tempra.to(units.degree).value
        else:
            raise NotImplementedError

    else:
        print('%sZ: need a FITS file to work on, or center coords and size' %
              (datetime.utcnow().isoformat(),))
        return


    if not outfile:
        outfile = '%s-RA%s-DEC%s-SIZE%s.catalog' % (catalog,
                                                    catra,
                                                    catdec,
                                                    catbox)
    if outdir:
        outfile = os.path.join(outdir, outfile)

    print('%sZ: making FOV catalog for '
          'center RA, DEC = %.5f, %.5f with size = %.5f deg' %
          (datetime.utcnow().isoformat(),
           catra, catdec, catbox))

    if catalog == 'GAIADR2':
        if gaiaidrequest not in ['GAIA','HAT','TMASS']:
            raise ValueError(
                'expected gaiaidrequest one of "GAIA", "HAT", "TMASS"')
        catalogcmd = CATALOGS[catalog]['cmd'].format(
            ra=catra,
            dec=catdec,
            boxlen=catbox,
            catalogpath=catalogpath if catalogpath else CATALOGS[catalog]['path'],
            brightrmag=brightrmag,
            faintrmag=faintrmag,
            outfile=outfile,
            idrequest=gaiaidrequest)
    elif catalog in ['2MASS', 'UCAC4']:
        catalogcmd = CATALOGS[catalog]['cmd'].format(
            ra=catra,
            dec=catdec,
            boxlen=catbox,
            catalogpath=catalogpath if catalogpath else CATALOGS[catalog]['path'],
            brightrmag=brightrmag,
            faintrmag=faintrmag,
            outfile=outfile)
    else:
        raise ValueError('catalog must be one of GAIADR2,2MASS,UCAC4')

    if DEBUG:
        print(catalogcmd)

    if os.path.exists(outfile):

        print('%sZ: found FOV catalog %s, continuing... ' %
              (datetime.utcnow().isoformat(), os.path.abspath(outfile)))

        return os.path.abspath(outfile)

    # execute the cataloger shell command
    catalogproc = subprocess.Popen(shlex.split(catalogcmd),
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

    # get results
    catalog_stdout, catalog_stderr = catalogproc.communicate()

    # get results if succeeded, log outcome, and return path of outfile
    if catalogproc.returncode == 0:

        print('%sZ: FOV catalog %s generated for '
              'center RA, DEC = %.5f, %.5f with size = %.5f deg' %
              (datetime.utcnow().isoformat(),
               os.path.abspath(outfile), catra, catdec, catbox))

        return os.path.abspath(outfile)

    else:

        print('%sZ: FOV catalog %s generation failed for '
              'center RA, DEC = %.5f, %.5f with size = %.5f deg!' %
              (datetime.utcnow().isoformat(),
               os.path.abspath(outfile), catra, catdec, catbox))

        return None


def reform_gaia_fov_catalog(
    incat, outcat, columns='id,ra,dec,xi,eta,G,Rp,Bp,plx,pmra,pmdec,varflag'
):
    """
    This converts the output catalog for gaia2read to the format required
    for magfit.

    columns is a CSV string containing the required columns.
    """
    allcolumns = ['id','ra','dec','raerr','decerr','plx','plxerr','pmra',
                  'pmdec','pmraerr','pmdecerr','epoch','astexcnoise',
                  'astexcnoisesig','astpriflag','G_nobs','G_flux',
                  'G_fluxerr','G_fluxovererr','G','Bp_nobs','Bp_flux',
                  'Bp_fluxerr','Bp_fluxovererr','Bp','Rp_nobs','Rp_flux',
                  'Rp_fluxerr','Rp_fluxovererr','Rp','BpRp_excess','RV',
                  'RV_err','varflag','Teff','Teff_lowq','Teff_highq',
                  'extinction','extinction_lowq','extinction_highq',
                  'reddening','reddening_lowq','reddening_highq',
                  'Rstar','Rstar_lowq','Rstar_highq','L','L_lowq','L_highq',
                  'xi','eta']

    columns = columns.split(',')
    colstoget = [allcolumns.index(x) for x in columns]

    inf = open(incat,'r')
    outf = open(outcat,'wb')

    for line in inf:
        if '#' not in line:
            sline = line.split()
            outcols = [sline[x] for x in colstoget]
            outline = ' '.join(outcols)
            outf.write('{:s}\n'.format(outline).encode('utf-8'))


def reform_fov_catalog(incat,
                       outcat,
                       columns='id,ra,dec,xi,eta,J,K,qlt,I,r,i,z'):
    """
    This converts the full output catalog from 2massread, etc. to the format
    required for magfit. Also useful for general reforming of the columns.

    columns is a CSV string containing columns needed from allcolumns below.
    """

    allcolumns = ['id','ra','dec','xi','eta','arcdis','J',
                  'Junc','H','Hunc','K','Kunc','qlt','B',
                  'V','R','I','u','g','r','i','z','field','num']

    columns = columns.split(',')
    colstoget = [allcolumns.index(x) for x in columns]

    inf = open(incat,'rb')
    outf = open(outcat,'wb')

    for line in inf:
        if '#' not in line:
            sline = line.split()
            outcols = [sline[x] for x in colstoget]
            outline = ' '.join(outcols)
            outf.write('%s\n' % outline)



def extract_frame_sources(fits,
                          outfile,
                          fistarexec='fistar',
                          ccdextent='0:2048,0:2048',
                          ccdgain=2.725,
                          fluxthreshold=1000,
                          zeropoint=17.11,
                          exptime=30.0):
    """
    This uses fistar to extract sources from the image.

    fistar -i 1-377741e_5.fits -o test.fistar --model elliptic \
            --iterations symmetric=4,general=2 --algorithm uplink \
            --format id,x,y,bg,amp,s,d,k,flux,s/n -g 2.725 \
            --mag-flux 17,30 --sort flux \
            --flux-threshold 1000 --section 0:0,2048:2048
    """

    if not os.path.exists(fits):

        print('%sZ: no FITS file to work on!' %
              (datetime.utcnow().isoformat(),))
        return None

    if not outfile:
        outfile = re.sub(sv.FITS_TAIL, '.fistar', fits)

    fistarcmd = FISTARCMD.format(
        fistarexec=fistarexec, # assuming fistar is in the path
        frame=fits,
        extractedlist=outfile,
        ccdgain=ccdgain,
        zeropoint=zeropoint,
        exptime=exptime,
        fluxthreshold=fluxthreshold,
        ccdsection=ccdextent
        )

    if DEBUG:
        print(fistarcmd)

    print('%sZ: starting fistar for %s...' %
          (datetime.utcnow().isoformat(), fits))

    # execute the fistar shell command
    fistarproc = subprocess.Popen(shlex.split(fistarcmd),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

    # get results
    fistar_stdout, fistar_stderr = fistarproc.communicate()

    # get results if succeeded, log outcome, and return path of outfile
    if fistarproc.returncode == 0:

        print('%sZ: fistar completed for %s -> %s' %
              (datetime.utcnow().isoformat(),fits, outfile))

        return outfile

    else:

        print('%sZ: fistar failed for %s!' %
              (datetime.utcnow().isoformat(), fits))

        return None


def parallel_sourceextract_worker(task):
    """
    This expands the task arg into the args and kwargs necessary for
    extract_frame_sources.
    """

    return (task[0][0], extract_frame_sources(*task[0],**task[1]))


def check_files(inlist, operationstr, outdir, intailstr='.fits',
                outtailstr='.fistar', skipifpartial=False):
    """
    You have a list of files you think you want to run an operation on.

    However, you do not want to repeat the operation if you have already run
    it.

    Further, sometimes you want to skip the operation entirely if it failed
    last time you tried it.

    input:

        inlist: a list of paths that you want to perform an operation on

        operationstr: a string for logging, e.g., "source extraction"

        outdir: path of the directory being written to

        intailsstr: the tail string of the input files

        outtailstr: the tail string of the output files

        skipifpartial: if we find ANY matches between the inlist and outlist,
            then return -1 and do not perform any operations at all.

    output:

        default: the list of files on which to operate. If there are no files
        on which to operate, or skipifpartial==True and there are some matches,
        returns -1.
    """

    # construct list of file names that would be created for the operation to
    # be performed
    outlist = [os.path.join(outdir,
                            os.path.basename(re.sub(intailstr, outtailstr, x))) for
              x in inlist]

    
    exists = np.array(outlist)[np.array([os.path.exists(f) for f in outlist])]
    _exists = [re.sub(outtailstr,'',e) for e in exists]
    _inlist = [re.sub(intailstr,'',w) for w in inlist]

    if len(exists) > 0:
        to_operate_on = np.array(inlist)[~np.in1d(_inlist, _exists)]
    else:
        to_operate_on = np.array(inlist)

    print('%sZ: found %s FITS files to %s...' %
          (datetime.utcnow().isoformat(), len(to_operate_on), operationstr))

    if len(to_operate_on) == 0:
        print('%sZ: escaping %s because no new files...' %
              (datetime.utcnow().isoformat(), operationstr))
        return -1

    if skipifpartial and (len(to_operate_on) < len(inlist)):
        print('%sZ: escaping %s because got partial list of files...' %
              (datetime.utcnow().isoformat(), operationstr))
        return -1

    else:
        return to_operate_on


def parallel_extract_sources(fitsdir,
                             outdir,
                             nworkers=8,
                             maxtasksperworker=1000,
                             fistarexec='fistar',
                             ccdextent='0:0,2048:2048',
                             ccdgain=2.725,
                             fluxthreshold=1000,
                             zeropoint=17.11,
                             exptime=30.0,
                             tailstr=FITS_TAIL,
                             fnamestr='*_?.fits',
                             overwrite=False):
    """
    This does parallel source extraction from all FITS in fitsdir, and puts the
    results in outdir.
    """

    # get a list of all fits files in the directory
    fitslist = glob.glob(os.path.join(fitsdir,fnamestr))
    print('%sZ: found %s FITS files in %s, starting source extraction...' %
          (datetime.utcnow().isoformat(),
           len(fitslist), fitsdir))

    if outdir and not os.path.exists(outdir):

        print('%sZ: making new output directory %s' %
              (datetime.utcnow().isoformat(),
               outdir))
        os.mkdir(outdir)

    if not overwrite:
        # get the files for which source extraction hasn't already been done
        toextract = check_files(fitslist,
                                'source extraction',
                                outdir,
                                intailstr=tailstr,
                                outtailstr='.fistar')
    else:
        toextract = fitslist
    if type(toextract) == int:
        if toextract == -1:
            return -1

    pool = mp.Pool(nworkers, maxtasksperchild=maxtasksperworker)

    tasks = [
        [(x, os.path.join(outdir,
                          os.path.basename(re.sub(tailstr,'.fistar',x)))),
         {'fistarexec':fistarexec,
          'ccdextent':ccdextent,
          'ccdgain':ccdgain,
          'fluxthreshold':fluxthreshold,
          'zeropoint':zeropoint,
          'exptime':exptime,}]
        for x in toextract
        ]

    # fire up the pool of workers
    results = pool.map(parallel_sourceextract_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict



def parallel_srcextract_list_worker(task):
    """
    This is the worker for the function below.

    task[0] = fits
    task[1] = {'fistarexec','ccdextent','ccdgain','fluxthreshold',
               'zeropoint', 'exptime'}
    """

    try:

        fits, kwargs = task

        if not os.path.exists(fits):
            return fits, None

        # get the required header keywords from the FITS file
        header = imageutils.get_header_keyword_list(fits,
                                                    ['GAIN',
                                                     'GAIN1',
                                                     'GAIN2',
                                                     'EXPTIME'])

        # handle the gain and exptime parameters
        if 'GAIN1' in header and 'GAIN2' in header:
            ccdgain = (header['GAIN1'] + header['GAIN2'])/2.0
        elif 'GAIN' in header:
            ccdgain = header['GAIN']
        else:
            ccdgain = None

        ccdexptime = header['EXPTIME'] if 'EXPTIME' in header else None

        if not (ccdgain or ccdexptime):
            print('ERR! %sZ: no GAIN or EXPTIME defined for %s' %
                  (datetime.utcnow().isoformat(),
                   fits))
            return fits, None

        # figure out the outputfile
        outfile = fits.replace('.fits','.fistar')

        # figure out the input kwargs to fistar
        kwargs['exptime'] = ccdexptime
        kwargs['ccdgain'] = ccdgain

        # figure out this frame's CCD and therefore zeropoint
        frameinfo = FRAMEREGEX.findall(fits)
        if frameinfo:
            kwargs['zeropoint'] = ZEROPOINTS[int(frameinfo[0][-1])]
        elif not frameinfo and not kwargs['zeropoint']:
            print('ERR! %sZ: no zeropoint mag defined for %s' %
                  (datetime.utcnow().isoformat(),
                   fits))
            return fits, None

        # run fistar
        fistar = extract_frame_sources(fits,
                                       outfile,
                                       **kwargs)

        if fistar and os.path.exists(fistar):
            return fits, fistar
        else:
            return fits, None

    except Exception as e:

        print('ERR! %sZ: could not extract sources for %s, error: %s' %
              (datetime.utcnow().isoformat(),
               fits, e))
        return fits, None



def parallel_extract_sources_for_list(fitslist,
                                      nworkers=16,
                                      maxworkerstasks=1000,
                                      fistarexec='fistar',
                                      ccdextent='0:0,2048:2048',
                                      ccdgain=2.725,
                                      fluxthreshold=1000,
                                      zeropoint=17.11,
                                      exptime=30.0):
    """
    This runs a parallel fistar operation on all sources in fitslist.

    Puts the results in the same directories as the FITS themselves.
    """

    pool = mp.Pool(nworkers, maxtasksperchild=maxtasksperworker)

    tasks = [
        (x,
         {'fistarexec':fistarexec,
          'ccdextent':ccdextent,
          'ccdgain':ccdgain,
          'fluxthreshold':fluxthreshold,
          'zeropoint':zeropoint,
          'exptime':exptime,}) for x in fitslist
        ]

    # fire up the pool of workers
    results = pool.map(parallel_srcextract_list_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}
    return returndict


def match_fovcatalog_framesources(frame_extracted_sourcelist,
                                  frame_projected_fovcatalog,
                                  outfile,
                                  srclist_cols=(0,1,2,5,6,7),
                                  fovcat_cols=(0,1,2,12,13),
                                  match_pixel_distance=0.5):
    """
    Does frame_projected_fovcatalog and frame_extracted_sourcelist matching.

        frame_extracted_sourcelist: *.fistar file
        frame_projected_fovcatalog: *.projcatalog file
        outfile: *.sourcelist file to be created by this function. Each line
        looks like:

     ID              RA       DEC     x          y         phot_id   x       y          s,d,k
    HAT-381-0000008 249.39070 5.27754 1261.64440 517.39870 4620 1261.75400 517.42700 1.39400 0.24400 -0.17900

    This matches the fovcatalog transformed to pixel coordinates to the
    extracted source list pixel coordinates and gets the IDs of the sources to
    be later used when creating the fiphot file.

    The procedure is:

    1. read frame sourcelist xy cols into ndarray
    2. read projected fovcatalog xy cols into ndarray
    3. vstack both ndarrays, combined = [sourcelist, fovcatalog]
    4. create a kd-Tree using tree = cKDTree(combined)
    5. find pairs using tree.query_pairs(match_pixel_distance, p=2) [p=2 ->
       standard euclidean distance for the Minkowski norm]
    6. convert result to list of index pairs
    7. get indices of fovcatalog for each pair and use to get ID from fovcatalog
    8. get indices of framelist for each pair and use to get S, D, K info
    9. make sure fovcatalog IDs are unique
    10. write to a combined fiphot file.
    """

    srclist = np.genfromtxt(frame_extracted_sourcelist,
                           dtype='S17,f8,f8,f8,f8,f8',
                           names=['id','x','y','s','d','k'],
                           usecols=srclist_cols)

    fovcat = np.genfromtxt(frame_projected_fovcatalog,
                            dtype='S17,f8,f8,f8,f8',
                            names=['id','ra','dec','x','y'],
                            usecols=fovcat_cols)

    srclist_xys = np.column_stack((srclist['x'],srclist['y']))
    fovcat_xys = np.column_stack((fovcat['x'],fovcat['y']))

    fovcat_tree = kdtree(fovcat_xys)

    distances, fovcat_indices = fovcat_tree.query(srclist_xys)

    # now that we have matches, put the two files together based on the indices
    # the output format will be:
    # srcext = source extracted frame list from fistar
    # fovcat = frame projected FOV catalog object list from
    # make_frame_sourcelist
    # fovcat HAT-ID, fovcat RA, fovcat DEC, fovcat x, fovcat y,
    # srcext id, srcext x, srcext y, srcext S, srcext D, srcext K
    if not outfile:
        outfile = (re.sub(sv.FITS_TAIL, '.matched-sources',
                          frame_extracted_sourcelist))


    outf = open(outfile,'wb')

    outlinestr = '%s %.5f %.5f %.5f %.5f %s %.5f %.5f %.5f %.5f %.5f\n'

    for dist, fovcat_ind, srclist_ind in zip(distances,
                                             fovcat_indices,
                                             range(len(srclist))):

        if dist < match_pixel_distance:
            outf.write(outlinestr %
                       (fovcat['id'][fovcat_ind],
                        fovcat['ra'][fovcat_ind],
                        fovcat['dec'][fovcat_ind],
                        fovcat['x'][fovcat_ind],
                        fovcat['y'][fovcat_ind],
                        srclist['id'][srclist_ind],
                        srclist['x'][srclist_ind],
                        srclist['y'][srclist_ind],
                        srclist['s'][srclist_ind],
                        srclist['d'][srclist_ind],
                        srclist['k'][srclist_ind]))

    outf.close()

    return outfile



def make_frameprojected_catalog(fits,
                                catalog,
                                catalogxycols=(12,13),
                                transformer='anrd2xy',
                                wcs=None,
                                ccdextent=None,
                                out=None,
                                removetemp=True,
                                pixborders=0.0,
                                observatory='hatpi'):

    """
    This makes the projected catalog for the frame to use with fiphot using the
    anet rdtoxy transform stored in the filename wcs and projects the catalog
    into pixel coordinates of the image. If wcs is None, a file with .wcs
    extension in the same directory as fits will be used as input. catalog is
    the path to the FOV catalog generated by 2massread or ucac4read. out is the
    name of the output sourcelist. Uses the executable defined in the
    transformer kwarg (usually a variant of the rd2xy binary from
    astrometry.net). ccdextent is a dictionary like CCDEXTENT above noting the
    extent of the CCD and tells us which objects are outside the FOV so they're
    removed from the output source list.

    Returns the path of the source list file produced if successful, otherwise
    returns None.
    """

    fitspath = os.path.abspath(fits)

    if wcs:
        framewcsfile = wcs
    else:
        wcspath = re.sub(sv.FITS_TAIL,'',fitspath)
        wcspath = wcspath + '.wcs'
        framewcsfile = wcspath

    if out:
        outfile = out
        temppath = out + '.projcattemp'
    else:
        outpath = re.sub(sv.FITS_TAIL,'',fitspath)
        temppath = outpath + '.projcattemp'
        outpath = outpath + '.projcatalog'
        outfile = outpath

    # format the transformer shell command
    transformcmd = TRANSFORMCMD.format(transformer=transformer,
                                       framewcsfile=framewcsfile,
                                       catalogsourcelist=catalog,
                                       outputfile=temppath)

    if DEBUG:
        print(transformcmd)

    # make sure the wcs file makes sense before trying the transform
    if (framewcsfile and
        os.path.exists(os.path.abspath(framewcsfile)) and
        os.stat(os.path.abspath(framewcsfile)).st_size > 0):

        # execute the transformer shell command
        transformproc = subprocess.Popen(shlex.split(transformcmd),
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        print('%sZ: %s' % (datetime.utcnow().isoformat(), transformcmd))

        # get results
        transform_stdout, transform_stderr = transformproc.communicate()

        # get results if succeeded, log outcome, and return path of outfile
        if transformproc.returncode == 0:

            if observatory=='hatpi':

                # now we need to take out sources outside the CCD extent
                sourcelist_x, sourcelist_y = np.loadtxt(temppath,
                                                        usecols=catalogxycols,
                                                        unpack=True)

                # get the extent of the CCD
                if not ccdextent:
                    ccdextent = CCDEXTENT

                # get indices for the lines to be kept optionally, remove all
                # sources within pixborders pixels of the edges of the image.
                keep_ind = np.where(
                    (sourcelist_x > (ccdextent['x'][0] + pixborders)) &
                    (sourcelist_x < (ccdextent['x'][1] - pixborders)) &
                    (sourcelist_y > (ccdextent['y'][0] + pixborders)) &
                    (sourcelist_y < (ccdextent['y'][1] - pixborders))
                )[0].tolist()

                # output the lines to be kept
                outf = open(outfile, 'wb')
                with open(temppath,'rb') as tempf:

                    templines = tempf.readlines()
                    templines = [x.decode('utf-8') for x in templines if '#' not in
                                 x.decode('utf-8')]
                    for ind in keep_ind:
                        outf.write(templines[ind].encode('utf-8'))

                outf.close()

            elif observatory=='tess':

                df = pd.read_csv(
                    temppath,
                    delim_whitespace=True,
                    names='id,ra,dec,xi,eta,G,Rp,Bp,plx,pmra,pmdec,varflag,x_proj,y_proj'.split(',')
                )

                sourcelist_x, sourcelist_y = (np.array(df['x_proj']),
                                              np.array(df['y_proj']))

                # get the extent of the CCD
                if not ccdextent:
                    ccdextent = CCDEXTENT

               # do a hack for TESS-SPOC WCS. the WCS is derived pre-trim of the
                # rows and columns. this corrects it.
                if ('proj' in fitspath and 's0' in fitspath and 'cam' in fitspath
                    and 'ccd' in fitspath and 'photref' in fitspath):

                    print('WRN! doing hack for TESS/SPOC WCS to fix the trim '+
                          'x/y CRPIX offset')

                    hdulist = pyfits.open(fitspath)
                    hdr = hdulist[0].header

                    sourcelist_x -= (hdr['SCCSA']-1)
                    sourcelist_y -= (hdr['SCIROWS']-1)

                    hdulist.close()

                    # the WCS-project and measured positions differ by this because
                    # of a coordinate offset problem, somewhere. this is
                    # CHECKED formally by wcsqualityassurance, and visually by
                    # inspecting  tessutils.plot_apertures_on_frame
                    sourcelist_x -= 0.5
                    sourcelist_y -= 0.5

                df['x_proj'] = sourcelist_x
                df['y_proj'] = sourcelist_y

                # get indices for the lines to be kept optionally, remove all
                # sources within pixborders pixels of the edges of the image.
                keep_ind = np.where(
                    (sourcelist_x > (ccdextent['x'][0] + pixborders)) &
                    (sourcelist_x < (ccdextent['x'][1] - pixborders)) &
                    (sourcelist_y > (ccdextent['y'][0] + pixborders)) &
                    (sourcelist_y < (ccdextent['y'][1] - pixborders))
                )[0].tolist()

                outdf = df.iloc[keep_ind]
                outdf.to_csv(outfile, index=False, header=False, sep=' ')
                print('TESS wrote {}'.format(outfile))
                print(outdf.describe())

            if removetemp:
                os.remove(temppath)

            print('%sZ: frame source list generation OK for %s' %
                  (datetime.utcnow().isoformat(),
                   fits))

            return outfile
        else:
            print('%sZ: frame source list generation '
                  'failed for %s: error was %s' %
                  (datetime.utcnow().isoformat(),
                   fits,
                   transform_stderr))
            if removetemp:
                try:
                    os.remove(temppath)
                except:
                    pass
            return None

    # the wcs file doesn't make sense for this FITS image, complain and return
    else:
        print('%sZ: WCS transform file does not work for %s, '
              'skipping this frame...' %
              (datetime.utcnow().isoformat(), fits))
        return None


def run_fiphot(fits,
               sourcelist=None,
               xycols='25,26', # set for full 2MASS fov catalog output format
               ccdgain=None,
               zeropoint=None,
               ccdexptime=None,
               aperturelist='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
               formatstr='ISXY,BbMms',
               outfile=None,
               removesourcelist=False,
               binaryoutput=True,
               observatory='hatpi'):
    """
    Thus runs fiphot for a single frame. Only the fits filename is required. If
    other parameters are not provided, they will be obtained from the image
    header and defaults.

    Returns the path of the .fiphot file produced if successful, otherwise
    returns None.
    """

    # get the required header keywords from the FITS file
    if observatory=='hatpi':
        headerlist = ['GAIN', 'GAIN1', 'GAIN2', 'EXPTIME', 'RAC', 'DECC',
                      'FOV']
    elif observatory=='tess':
        headerlist = ['GAINA', 'TELAPSE', 'CRVAL1', 'CRVAL2']

    header = imageutils.get_header_keyword_list(fits, headerlist)

    # handle the gain and exptime parameters
    if not ccdgain:

        # FIXME: is this right? should we use separate gain values for each side
        # of the CCD? what about stars that straddle the middle?
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
               fits))
        return None

    # figure out the fitsbase from the fits filename
    fitsbase = re.sub(sv.FITS_TAIL,'',os.path.basename(fits))

    # handle the zeropoints
    if not zeropoint:

        # if the zeropoint isn't provided and if this is a HAT frame, the ccd
        # number will get us the zeropoint in the ZEROPOINTS dictionary
        frameinfo = FRAMEREGEX.findall(fits)
        if frameinfo:
            zeropoint = ZEROPOINTS[int(frameinfo[0][-1])]
        else:
            print('%sZ: no zeropoint magnitude defined for %s' %
                  (datetime.utcnow().isoformat(),
                   fits))
            return None


    # figure out the output path
    if not outfile:
        outfile = os.path.abspath(re.sub(sv.FITS_TAIL,'.fiphot',fits))

    # figure out the sourcelist path
    if not sourcelist:
        sourcelist = os.path.abspath(re.sub(sv.FITS_TAIL,'.sourcelist',fits))
        if not os.path.exists(sourcelist):

            print("%sZ: can't find a source list for %s" %
                  (datetime.utcnow().isoformat(),
                   fits))
            return None

    # figure out if we want binary output or not
    if binaryoutput:
        binaryout = '--binary-output'
    else:
        binaryout = ''


    # assemble the fiphot command string
    fiphotcmd = FIPHOTCMD.format(fits=fits,
                                 sourcelist=sourcelist,
                                 xycols=xycols,
                                 zeropoint=zeropoint,
                                 ccdgain=ccdgain,
                                 ccdexptime=ccdexptime,
                                 aperturelist=aperturelist,
                                 fitsbase=fitsbase,
                                 formatstr=formatstr,
                                 binaryout=binaryout,
                                 outfile=outfile)

    if DEBUG:
        print(fiphotcmd)

    # execute the fiphot command
    fiphotproc = subprocess.Popen(shlex.split(fiphotcmd),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

    # get results
    fiphot_stdout, fiphot_stderr = fiphotproc.communicate()

    # get results if succeeded, log outcome, and return path of outfile
    if fiphotproc.returncode == 0:

        print('%sZ: photometry using fiphot OK for %s' %
              (datetime.utcnow().isoformat(),
               fits))

        if removesourcelist:
            os.remove(sourcelist)

        return outfile


    else:
        print('%sZ: fiphot failed for %s: error was %s' %
              (datetime.utcnow().isoformat(),
               fits,
               fitphot_stderr))
        return None



def do_photometry(fits,
                  reformedfovcatalog,
                  extractsources=True,
                  fluxthreshold=500.0,
                  fovcat_xycols=(12,13),
                  projcat_xycols=(24,25),
                  fiphot_xycols='7,8', # set for matched source list
                  outdir=None,
                  ccdextent=None,
                  ccdgain=None,
                  zeropoint=None,
                  ccdexptime=None,
                  removesourcetemp=True,
                  pixborders=0.0,
                  aperturelist='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
                  formatstr='ISXY,BbMms',
                  removesourcelist=False,
                  binaryoutput=True,
                  minsrcbgv=100.0,
                  maxmadbgv=20.0,
                  maxframebgv=2000.0,
                  minnstars=500,
                  observatory='hatpi',
                  extractforsdk=False,
                  overwrite=True):
    """
    This rolls up the sourcelist and fiphot functions above.

    Runs both stages on fits, and puts the output in outdir if it exists. If it
    doesn't or is None, then puts the output in the same directory as fits.

    kwargs:

        reformedfovcatalog is the path to the 2MASS/UCAC4 catalog for all sources in the
        observed field. (12 columns!)

        extractforsdk (bool): if you want to run fistar source extraction on
        the frame to get SDK values, even though you're really forcing the
        photometry from a projected catalog.
    """

    outprojcat = re.sub(sv.FITS_TAIL,'.projcatalog',os.path.basename(fits))
    outsourcelist = re.sub(sv.FITS_TAIL,'.sourcelist',os.path.basename(fits))
    outfiphot = re.sub(sv.FITS_TAIL,'.fiphot',os.path.basename(fits))

    if outdir and os.path.exists(outdir):

        outprojcat = os.path.join(outdir, outprojcat)
        outsourcelist = os.path.join(outdir,outsourcelist)
        outfiphot = os.path.join(outdir, outfiphot)

    elif outdir and not os.path.exists(outdir):

        print('%sZ: making new output directory %s' %
              (datetime.utcnow().isoformat(),
               outdir))
        os.mkdir(outdir)
        outprojcat = os.path.join(outdir, outprojcat)
        outsourcelist = os.path.join(outdir,outsourcelist)
        outfiphot = os.path.join(outdir, outfiphot)

    else:

        outprojcat, outsourcelist, outfiphot = None, None, None

    if DEBUG:
        print('output projected catalog will be %s' % outprojcat)
        print('output source list will be %s' % outsourcelist)
        print('output fiphot will be %s' % outfiphot)

    # make the .projcatalog files (projects catalog onto the frame)
    if overwrite or not os.path.exists(outprojcat):
        projcatfile = make_frameprojected_catalog(fits,
                                                  reformedfovcatalog,
                                                  ccdextent=ccdextent,
                                                  out=outprojcat,
                                                  removetemp=removesourcetemp,
                                                  pixborders=pixborders,
                                                  observatory=observatory)
    else:
        print('%sZ: %s already exists, skipping making projcatalog.' %
              (datetime.utcnow().isoformat(), outprojcat))
        projcatfile = outprojcat

    if projcatfile:

        # if we're supposed to extract sources and run photometry on them
        # instead of just the sources in the projected fovcatalog, do so
        if extractsources:
            outfistar = os.path.join(outdir, re.sub(sv.FITS_TAIL, '.fistar',
                    os.path.basename(fits)))

            # extract sources
            if observatory=='hatpi':
                if overwrite or not os.path.exists(outfistar):
                    framesources = extract_frame_sources(
                        fits, outfistar,
                        fluxthreshold=fluxthreshold
                    )
                else:
                    print('%sZ: %s already exists, skipping fistar.' %
                          (datetime.utcnow().isoformat(), outfistar))
                    framesources = outfistar

            elif observatory=='tess':
                framesources = extract_frame_sources(
                    fits,
                    os.path.join(
                        outdir,
                        re.sub(sv.FITS_TAIL,'.fistar',os.path.basename(fits))
                        ),
                    fluxthreshold=fluxthreshold,
                    ccdgain=ccdgain,
                    zeropoint=zeropoint,
                    exptime=ccdexptime
                )

            if framesources:

                # match extracted frame sources to the projected fovcatalog.
                # this makes a .sourcelist file, named "outsourcelist". 
                matchedsources = match_fovcatalog_framesources(
                    framesources, # *.fistar file
                    projcatfile,  # *.projcatalog file
                    outsourcelist # *.sourcelist file, created by this function
                    )

                fiphot_xycols = '7,8'

            else:

                print('%sZ: extracting sources failed for %s!' %
                      (datetime.utcnow().isoformat(), fits))
                return None, None

        else:

            # even if you don't want to extract sources for *photometry*, you
            # might want to extract them for their SDK values. for example,
            # this is needed so that you can select photometric reference
            # frames, while still running forced photometry from a base
            # catalog.
            if extractforsdk:

                _ = extract_frame_sources(
                    fits,
                    os.path.join(
                        outdir,
                        re.sub(sv.FITS_TAIL,'.fistar',os.path.basename(fits))
                        ),
                    fluxthreshold=fluxthreshold,
                    ccdgain=ccdgain,
                    zeropoint=zeropoint,
                    exptime=ccdexptime
                )

            outsourcelist = projcatfile
            fiphot_xycols = '13,14'


        # run fiphot on the source list
        fiphotfile = run_fiphot(fits,
                                sourcelist=outsourcelist,
                                aperturelist=aperturelist,
                                outfile=outfiphot,
                                xycols=fiphot_xycols,
                                ccdgain=ccdgain,
                                zeropoint=zeropoint,
                                formatstr=formatstr,
                                ccdexptime=ccdexptime,
                                removesourcelist=removesourcelist,
                                binaryoutput=binaryoutput,
                                observatory=observatory)

        if fiphotfile:

            return fiphotfile, None

        else:

            print('%sZ: photometry failed for %s!' %
                  (datetime.utcnow().isoformat(), fits))

    else:

        print('%sZ: creating a projected source catalog failed for %s!' %
              (datetime.utcnow().isoformat(), fits))
        return None, None


def parallel_photometry_worker(task):
    """
    This is the parallel photometry worker function for use with
    parallel_fitsdir_photometry below. Just calls do_photometry with expanded
    args and kwargs from the two element task list. task[0] is a tuple of args,
    and task[1] is a dictionary of kwargs. task[2] is a boolean indicating if we
    should kill bad frames.

    Returns a tuple of form: (fits, fiphot)
    """

    try:

        # first, do the photometry
        framephot, frameinfo = do_photometry(*task[0], **task[1])

        badframesdir = os.path.join(task[1]['outdir'],'badframes')

        # make sure all is OK with this frame
        if framephot:
            result = (framephot, frameinfo)

        # if the frame photometry is bad or the frame isn't OK, delete its
        # fiphot so we don't have to deal with it later
        else:
            # if the fiphot exists and we're allowed to kill it, do so
            if os.path.exists(framephot) and task[2]:

                filestomove = glob.glob(
                    os.path.join(
                        os.path.dirname(framephot),
                        os.path.basename(framephot).replace('.fiphot','.*')
                    )
                )

                for filetomove in filestomove:
                    shutil.move(filetomove, badframesdir)

            # tell the user what happened
            print('WRN! frame %s rejected, %s. %s' %
                  (task[0][0],
                   'files moved' if task[2] else 'fiphot %s' % framephot,
                   frameinfo if frameinfo else 'photometry failed!'))

            result = (framephot, frameinfo)

    except Exception as e:

        print('ERR! photometry failed! reason: %s' % e)
        result = (None, None)

    return result


def parallel_fitsdir_photometry(
        fitsdir,
        outdir,
        fovcatalog,
        fluxthreshold=500.0,
        ccdextent=None,
        pixborders=0.0,
        aperturelist='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
        removesourcetemp=True,
        removesourcelist=False,
        binaryoutput=True,
        nworkers=16,
        maxtasksperworker=1000,
        saveresults=True,
        rejectbadframes=True,
        minsrcbgv=200.0,
        maxmadbgv=150.0,
        maxframebgv=2000.0,
        minnstars=500,
        formatstr='ISXY,BbMms',
        ccdgain=None,
        ccdexptime=None,
        zeropoint=None,
        fitsglob='?-*_?.fits',
        extractsources=True,
        fovcat_xycols=(12,13),
        projcat_xycols=(24,25),
        fiphot_xycols='7,8',
        observatory='hatpi',
        overwrite=True
        ):
    """
    This does photometry for all FITS files in a directory using nworkers
    parallel workers.
    """

    # get a list of all fits files in the directory
    fitslist = glob.glob(os.path.join(fitsdir,fitsglob))

    if not overwrite:
        existing = glob.glob(
            os.path.join(fitsdir, fitsglob.replace('.fits', '.fiphot'))
        )
        requested = list(map(os.path.basename, fitslist))
        requested = [r.replace('.fits','') for r in requested]
        alreadyexists = list(map(os.path.basename, existing))
        alreadyexists = [ae.replace('.fiphot','') for ae in alreadyexists]

        setdiff = np.setdiff1d(requested, alreadyexists)
        if len(setdiff) == 0:
            print("%sZ: aperture photometry already done for all FITS files..." %
                  datetime.utcnow().isoformat())
            return
        fitslist = [os.path.join(fitsdir, sd+'.fits') for sd in setdiff]


    print('%sZ: found %s FITS files in %s, starting photometry...' %
          (datetime.utcnow().isoformat(),
           len(fitslist), fitsdir))

    if outdir and not os.path.exists(outdir):

        print('%sZ: making new output directory %s' %
              (datetime.utcnow().isoformat(),
               outdir))
        os.mkdir(outdir)

    pool = mp.Pool(nworkers,maxtasksperchild=maxtasksperworker)

    tasks = [[(x, fovcatalog),
              {'outdir':outdir,
               'ccdextent':ccdextent,
               'pixborders':pixborders,
               'aperturelist':aperturelist,
               'removesourcetemp':removesourcetemp,
               'removesourcelist':removesourcelist,
               'fluxthreshold':fluxthreshold,
               'binaryoutput':binaryoutput,
               'minsrcbgv':minsrcbgv,
               'maxmadbgv':maxmadbgv,
               'maxframebgv':maxframebgv,
               'minnstars':minnstars,
               'formatstr':formatstr,
               'ccdgain':ccdgain,
               'ccdexptime':ccdexptime,
               'zeropoint':zeropoint,
               'extractsources':extractsources,
               'observatory':observatory,
               'fovcat_xycols':fovcat_xycols,
               'projcat_xycols':projcat_xycols,
               'fiphot_xycols':fiphot_xycols,
               'overwrite':overwrite}, rejectbadframes] for x in fitslist]

    # if the badframes directory doesn't exist, make it
    badframesdir = os.path.join(outdir, 'badframes')

    if not os.path.exists(badframesdir):
        os.mkdir(badframesdir)

    # fire up the pool of workers
    results = pool.map(parallel_photometry_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}

    if saveresults:
        resultsfile = open(os.path.join(outdir,'TM-photometry.pkl'),'wb')
        pickle.dump(returndict, resultsfile)
        resultsfile.close()

    return returndict



def parallel_fitslist_photometry(
        fitslist,
        outdir,
        photokey,
        fovcatalog,
        fluxthreshold=500.0,
        ccdextent=None,
        pixborders=0.0,
        aperturelist='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
        removesourcetemp=True,
        removesourcelist=False,
        binaryoutput=True,
        nworkers=16,
        maxtasksperworker=1000,
        saveresults=True,
        rejectbadframes=True,
        formatstr='ISXY,BbMms',
        minsrcbgv=200.0,
        maxmadbgv=150.0,
        maxframebgv=2000.0,
        minnstars=500
        ):
    """
    This does photometry for all FITS files in the given list using nworkers
    parallel workers.

    photokey is required to set the name of the output photometry info pickle.
    """

    # get a list of all fits files in the directory
    goodlist = [x for x in fitslist if os.path.exists(x)]

    # if we have no files, then bail out
    if not goodlist:
        print('%sZ: no good FITS in list, bailing out...' %
              (datetime.utcnow().isoformat(),))
        return


    print('%sZ: found %s FITS files in input list, starting photometry...' %
          (datetime.utcnow().isoformat(),
           len(goodlist)))

    if outdir and not os.path.exists(outdir):

        print('%sZ: making new output directory %s' %
              (datetime.utcnow().isoformat(),
               outdir))
        os.mkdir(outdir)

    pool = mp.Pool(nworkers,maxtasksperchild=maxtasksperworker)

    tasks = [[(x, fovcatalog),
              {'outdir':outdir,
               'ccdextent':ccdextent,
               'pixborders':pixborders,
               'aperturelist':aperturelist,
               'removesourcetemp':removesourcetemp,
               'removesourcelist':removesourcelist,
               'fluxthreshold':fluxthreshold,
               'binaryoutput':binaryoutput,
               'minsrcbgv':minsrcbgv,
               'formatstr':formatstr,
               'maxmadbgv':maxmadbgv,
               'maxframebgv':maxframebgv,
               'minnstars':minnstars}, rejectbadframes] for x in goodlist]

    # if the badframes directory doesn't exist, make it
    badframesdir = os.path.join(outdir, 'badframes')

    if not os.path.exists(badframesdir):
        os.mkdir(badframesdir)

    # fire up the pool of workers
    results = pool.map(parallel_photometry_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # this is the return dictionary
    returndict = {x:y for (x,y) in results}

    if saveresults:
        resultsfile = open(os.path.join(outdir,
                                        'TM-photometry-%s.pkl' % photokey),'wb')
        pickle.dump(returndict, resultsfile)
        resultsfile.close()

    return returndict



##############################
## FRAME INFO AND FILTERING ##
##############################

def collect_image_info(fits, fistar,
                       minsrcbgv=100.0,
                       minsrcsval=1.5,
                       maxframebgv=2000.0,
                       maxmadbgv=150.0,
                       minnstars=500):
    """
    This collects the following info about a frame.

    - nstars detected
    - median background for all source detections
    - MAD of the background for all source detections
    - the overall image background

    bad frames are usually those with:

    - large MAD for the background
    - median source background < 0
    - nstars < 500

    furthermore, a running average over, say, 10 frames will reject large
    deviations in:

    - nstars
    - median source background
    - overall image background
    """

    frame, hdr = read_fits(fits)

    # the overall image background
    # assume this is the same as the median for now
    # previously, we were assuming that this is 100 ADU below the median
    imgbackg = extract_img_background(frame,median_diffbelow=0.0)

    # at some point this used to return an array, now it doesn't?
    # guard against this madness
    if isinstance(imgbackg,np.ndarray):
        framebgv = float(imgbackg[0])
    elif isinstance(imgbackg,int) or isinstance(imgbackg,float):
        framebgv = float(imgbackg)

    # get the fistar file columns we need
    framecols = np.genfromtxt(fistar,
                              usecols=(3,5),
                              names=['bgv','sval'],
                              dtype='f8,f8',comments='#')

    finitesrcbgvs = framecols['bgv'][np.isfinite(framecols['bgv'])]
    nstars = len(finitesrcbgvs)
    mediansrcbgv = np.median(finitesrcbgvs)
    madsrcbgv = np.median(np.abs(finitesrcbgvs - mediansrcbgv))
    mediansval = np.median(framecols['sval'][np.isfinite(framecols['sval'])])

    # check if the frame was aborted in the middle of the exposure
    if 'ABORTED' in hdr and hdr['ABORTED'] and hdr['ABORTED'] == 1:
        frameaborted = True
    elif 'ABORTED' in hdr and hdr['ABORTED'] and hdr['ABORTED'] == 0:
        frameaborted = False
    else:
        frameaborted = None

    frameok = ((mediansrcbgv > minsrcbgv) and
               (madsrcbgv < maxmadbgv) and
               (-2*minsrcbgv < framebgv < maxframebgv) and
               (nstars >= minnstars) and
               (mediansval > minsrcsval) and
               (frameaborted is not True))

    frameinfo = {'fits':fits,
                 'fistar':fistar,
                 'nstars':nstars,
                 'medsrcbgv':mediansrcbgv,
                 'madsrcbgv':madsrcbgv,
                 'medsrcsval':mediansval,
                 'framebgv':framebgv,
                 'frameok':frameok}

    return frameinfo



def frame_filter_worker(task):
    """
    This wraps collect_image_info above and removes the fiphot for image if it's
    rejected.

    task[0] = fits
    task[1] = fistar
    task[2] = {'minsrcbgv', 'maxframebgv', 'maxmaxbgv',
               'minnstars', 'minsrcsval'}

    this returns:

    True: if the frame was not filtered out
    False: if the frame was filtered out
    None: if the frame filtering failed
    """

    try:

        # first, make sure that fits and fistar both exist
        if (task[0] and task[1] and
            os.path.exists(task[0]) and os.path.exists(task[1])):

            frameinfo = collect_image_info(task[0],
                                           task[1],
                                           **task[2])

            # get rid of the frame if we're allowed to do so
            if not frameinfo['frameok']:
                returnval = False

            else:
                returnval = True

            return returnval

        else:
            print("ERR! fits/fiphot don't exist for this frame: %s" % task[0])
            return None

    except Exception as e:
        print("ERR! frame stats collection failed for %s, reason: %s" %
              (task[0], e))
        return None


def parallel_frame_filter(fitsdir,
                          fitsglob='?-*_?.fits',
                          fistarext='.fistar',
                          fiphotext='.fiphot',
                          removebadframes=False,
                          badframesdir=None,
                          minsrcbgv=100.0,
                          maxmadbgv=200.0,
                          minsrcsval=1.5,
                          maxframebgv=2000.0,
                          minnstars=500,
                          nworkers=16,
                          maxworkertasks=1000):
    """
    This goes through a fitsdir and removes bad frames.
    """

    # find all the fits files
    fitslist = glob.glob(os.path.join(os.path.abspath(fitsdir),
                                      fitsglob))

    # make sure all of these have accompanying fistar and fiphot files
    tasks = []

    print('%s total FITS, finding good FITS files...' % len(fitslist))

    for fits in fitslist:

        fistar = fits.replace('.fits',fistarext)

        if os.path.exists(fistar):
            tasks.append((fits, fistar, {'minsrcbgv':minsrcbgv,
                                         'minsrcsval':minsrcsval,
                                         'maxmadbgv':maxmadbgv,
                                         'maxframebgv':maxframebgv,
                                         'minnstars':minnstars}))

    print('%s FITS to work on.' % len(tasks))

    if len(tasks) > 0:

        if not badframesdir:
            badframesdir = os.path.join(fitsdir, 'badframes')
            if not os.path.exists(badframesdir):
                os.mkdir(badframesdir)

        print('checking FITS files...')

        # now start up the workers
        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
        results = pool.map(frame_filter_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        outdict = {}

        # now remove the fiphots if we're asked to do so
        for x, result in zip(tasks, results):

            fits = x[0]

            if (result is False or result is None) and removebadframes:
                filestomove = glob.glob(fits.replace('.fits','.*'))
                for deadfile in filestomove:
                    shutil.move(deadfile, badframesdir)
                print('moved all files for %s to %s' % (fits, badframesdir))

            elif (result is False or result is None) and not removebadframes:
                print('bad frame %s, not moving' % fits)

            else:
                print('frame %s is OK' % fits)

            outdict[fits] = (result, fits.replace('.fits',fiphotext))

        resultsfile = open(os.path.join(fitsdir,
                                        'TM-framerejection.pkl'),'wb')
        pickle.dump(outdict, resultsfile)
        resultsfile.close()

        return outdict



#################################
## MAGNITUDE FITTING FUNCTIONS ##
#################################

def get_magfit_frames(fitsdir,
                      fitsglob,
                      photdir,
                      workdir=None,
                      headerfilters=None,
                      selectreference=True,
                      framestats=False,
                      linkfiles=True,
                      outlistfile=None,
                      observatory='hatpi'):
    """
    fitsdir = directory where the FITS object frames are
    fitsglob = glob to select a subset of the FITS object frames
    photdir = directory where the TEXT fiphot files are (to select a
              singlephotref)
    workdir = directory where to create FITS and fiphot symlinks so
    MagnitudeFitting.py can work there

    This does the following:

    1. gets a list of frames in fitsdir that match fitsglob
    2. associates these files with their photometry files in photdir
    3. optionally filters them by the definitions in headerfilters (TBD)
    4. optionally creates symlinks to the chosen frame files in the workdir
       (photdir by default, directory will be created if non-existent)
    5. optionally selects a frame that might be a good reference for single
       reference magfit
    6. optionally writes the frame list to outlistfile.

    headerfilters is a list of string elements of the form:

    '<FITS header key> <operator> <FITS header value>'

    where <operator> is a standard binary operator. This string will be evaled
    to get the filter results.

    if selectreference is True, the following algorithm chooses a reference
    frame:

    1. from frames, get zenith distances (Z), moon distance (MOONDIST), moon
       elevation (MOONELEV).
    2. from (text) fiphot files, get nsources with G flags, MAD of magnitudes in
       largest aperture, median magnitude error in largest aperture.
    3. sort Z asc, MOONDIST desc, MOONELEV asc, nsources desc, good nsources
       desc, mag MAD asc, magerr asc, use these sort indices to sort framelists
    4. create sets of first 200 frames in each framelist above (or all frames if
       total nframes with phot < 200), and then intersect them all to find a set
       of frames that have the best properties. pick the first one out of the
       final intersection set as the reference frame.

    Returns a dict of the form:

    {'framelist':<list of frames passing fitsglob and headerfilters>,
     'photlist':<list of fitphot files associated with selected frames>,
     'framelistfile':<path to the outlistfile if created>,
     'referenceframe':<path to the chosen reference frame>,
     'referencestats':<stats dictionary for the reference frame>,
     'framestats':<a stats dictionary for all the frames>}
    """

    # first, get the frames
    fitslist = glob.glob(os.path.join(fitsdir, fitsglob))

    print('%sZ: %s FITS files found in %s matching glob %s' %
          (datetime.utcnow().isoformat(),
           len(fitslist), fitsdir, fitsglob))

    # TODO: add filtering by header keywords using headerfilters

    photfits = []
    photlist = []

    # associate the frames with their fiphot files
    for fits in fitslist:

        searchpath = os.path.join(
            photdir,
            re.sub(sv.FITS_TAIL,'.fiphot',os.path.basename(fits))
            )

        if os.path.exists(searchpath):
            photfits.append(fits)
            photlist.append(searchpath)

    # remove all frames with no fiphot files
    fitslist = photfits

    # check if the workdir exists. if it doesn't, create it
    if workdir and not os.path.exists(workdir):
        os.mkdir(workdir)
        print('%sZ: made work directory %s' %
              (datetime.utcnow().isoformat(),
               workdir))

    # if the workdir is not provided, use the photdir as workdir
    if not workdir:
        workdir = photdir

    # make sure the workdir has a stats and MPHOTREF directory
    if not os.path.exists(os.path.join(workdir, 'stats')):
        os.mkdir(os.path.join(workdir,'stats'))
    if not os.path.exists(os.path.join(workdir,'MPHOTREF')):
        os.mkdir(os.path.join(workdir,'MPHOTREF'))

    workfitslist, workphotlist = [], []

    # link the frames in fitsdir to workdir
    if linkfiles:
        # temporarily change the directory so symlinking works
        cwd = os.getcwd()

        try:

            os.chdir(workdir)
            for fits in fitslist:
                os.symlink(fits, os.path.basename(fits))
                workfitslist.append(
                    os.path.join(workdir, os.path.basename(fits))
                    )
            # change back at the end
            os.chdir(cwd)
            print('%sZ: linked frames to workdir %s' %
                  (datetime.utcnow().isoformat(),
                   workdir))

        except Exception as e:
            print('%sZ: linking fiphot files to workdir %s failed!' %
                  (datetime.utcnow().isoformat(),
                   workdir))
            os.chdir(cwd)
            raise

    # if the workdir != photdir, then link the fiphot files too
    if workdir != photdir and linkfiles:
        # temporarily change the directory so symlinking works
        cwd = os.getcwd()

        try:
            os.chdir(workdir)
            for fiphot in photlist:
                os.symlink(fiphot, os.path.basename(fiphot))
                workphotlist.append(
                    os.path.join(workdir, os.path.basename(fiphot))
                    )
            # change back at the end
            os.chdir(cwd)
            print('%sZ: linked fiphot files to workdir %s' %
                  (datetime.utcnow().isoformat(),
                   workdir))

        except Exception as e:
            print('%sZ: linking fiphot files to workdir %s failed!' %
                  (datetime.utcnow().isoformat(),
                   workdir))
            os.chdir(cwd)
            raise

    if not workfitslist:
        workfitslist = fitslist
    if not workphotlist:
        workphotlist = photlist

    # collect the minimum returndict
    returndict = {'origframelist':fitslist,
                  'origphotlist':photlist,
                  'workframelist':workfitslist,
                  'workphotlist':workphotlist,
                  'framelistfile':outlistfile}

    # find a reference frame
    goodframes, goodphots = [], []
    if selectreference:

        print('%sZ: selecting a reference frame...' %
              (datetime.utcnow().isoformat()))

        if observatory=='hatpi':
            zenithdist, moondist, moonelev = [], [], []
        elif observatory=='tess':
            pass
        ngoodobjects, medmagerr, magerrmad = [], [], []

        # for each FITS and fiphot combo, collect stats
        for fits, fiphot in zip(workfitslist, workphotlist):

            if observatory=='hatpi':
                headerdata = imageutils.get_header_keyword_list(
                                            fits, ['Z','MOONDIST','MOONELEV'])

            # decide if the fiphot file is binary or not. read the first 600
            # bytes and look for the '--binary-output' text
            with open(fiphot,'rb') as fiphotf:
                header = fiphotf.read(1000)

            if '--binary-output' in header and HAVEBINPHOT:

                photdata_f = read_fiphot(fiphot)

                if photdata_f:

                    photdata = {
                        'mag':np.array(photdata_f['per aperture'][2]['mag']),
                        'err':np.array(photdata_f['per aperture'][2]['mag err']),
                        'flag':np.array(
                            photdata_f['per aperture'][2]['status flag']
                            )
                        }
                else:

                    print('no photdata in %s, skipping...' % fiphot)
                    continue

                del photdata_f

            elif '--binary-output' in header and not HAVEBINPHOT:

                print('%sZ: %s is a binary fiphot file, '
                      'but no binary fiphot reader is present, skipping...' %
                      (datetime.utcnow().isoformat(), fiphot))
                continue

            else:

                # read in the phot file
                photdata = np.genfromtxt(
                    fiphot,
                    usecols=(12,13,14),
                    dtype='f8,f8,S5',
                    names=['mag','err','flag']
                    )

            # calculate stats for photometry

            # find good frames
            if '--binary-output' in header:
                goodind = np.where(photdata['flag'] == 0)
            else:
                goodind = np.where(photdata['flag'] == 'G')

            ngood = len(goodind[0])
            median_mag = np.median(photdata['mag'][goodind])
            median_magerr = np.median(photdata['err'][goodind])
            medabsdev_mag = np.median(
                np.abs(photdata['mag'][goodind] - median_mag)
                )

            # append to result lists
            goodframes.append(fits)
            goodphots.append(fiphot)
            ngoodobjects.append(ngood)
            medmagerr.append(median_magerr)
            magerrmad.append(medabsdev_mag)
            if observatory=='hatpi':
                zenithdist.append(headerdata['Z'])
                moondist.append(headerdata['MOONDIST'])
                moonelev.append(headerdata['MOONELEV'])

            if DEBUG:
                if observatory=='hatpi':
                    print('frame = %s, phot = %s, Z = %s, MOONDIST = %s, '
                          'MOONELEV = %s, ngood = %s, medmagerr = %.5f, '
                          'magerrmad = %.5f' %
                          (fits, fiphot, headerdata['Z'],
                           headerdata['MOONDIST'], headerdata['MOONELEV'],
                           ngood, median_magerr, medabsdev_mag))

                elif observatory=='tess':
                    print('frame = %s, phot = %s, '
                          'ngood = %s, medmagerr = %.5f, '
                          'magerrmad = %.5f' %
                          (fits, fiphot, ngood, median_magerr, medabsdev_mag))

        # now that we're done collecting data, sort them in orders we want
        goodframes = np.array(goodframes)
        goodphots = np.array(goodphots)
        ngood_ind = np.argsort(ngoodobjects)[::-1]
        mederr_ind = np.argsort(medmagerr)
        magmad_ind = np.argsort(magerrmad)
        if observatory=='hatpi':
            zenithdist_ind = np.argsort(zenithdist)
            moondist_ind = np.argsort(moondist)[::-1]
            moonelev_ind = np.argsort(moonelev)

        # get the first 200 of these or all 200 if n < 200
        if len(goodframes) > 200:
            ngood_ind = ngood_ind[:500]
            mederr_ind = mederr_ind[:500]
            magmad_ind = magmad_ind[:500]
            if observatory=='hatpi':
                zenithdist_ind = zenithdist_ind[:500]
                moondist_ind = moondist_ind[:500]
                moonelev_ind = moonelev_ind[:500]

        # intersect all arrays to find a set of common indices that belong to
        # the likely reference frames

        photgood_ind = np.intersect1d(np.intersect1d(ngood_ind,
                                                     magmad_ind,
                                                     assume_unique=True),
                                      mederr_ind,assume_unique=True)

        if observatory=='hatpi':
            headergood_ind =  np.intersect1d(np.intersect1d(moondist_ind,
                                                            moonelev_ind,
                                                            assume_unique=True),
                                             zenithdist_ind,assume_unique=True)

            allgood_ind = np.intersect1d(photgood_ind, headergood_ind,
                                         assume_unique=True)
        elif observatory=='tess':
            allgood_ind = photgood_ind
        else:
            raise NotImplementedError

        # if the headers and photometry produce a good reference frame, use
        # that. if they don't, use the photometry to choose a good reference
        # frame
        if len(allgood_ind) > 0:
            selectedreference = goodframes[allgood_ind[0]]
            selectedind = allgood_ind[0]
        elif len(photgood_ind) > 0:
            selectedreference = goodframes[photgood_ind[0]]
            selectedind = photgood_ind[0]
        elif len(headergood_ind) > 0:
            selectedreference = goodframes[headergood_ind[0]]
            selectedind = headergood_ind[0]
        else:
            selectedreference = None
            selectedind = None

        print('%sZ: selected reference frame = %s' %
              (datetime.utcnow().isoformat(), selectedreference))
        print('%sZ: selected reference phot = %s' %
              (datetime.utcnow().isoformat(), goodphots[selectedind]))

        # update the returndict with reference frame and stats
        returndict['referenceframe'] = selectedreference
        returndict['referencephot'] = goodphots[selectedind]
        if selectedreference and observatory=='hatpi':
            returndict['referencestats'] = {
                'zenithdist':zenithdist[selectedind],
                'moondist':moondist[selectedind],
                'moonelev':moonelev[selectedind],
                'ngood':ngoodobjects[selectedind],
                'magmad':magerrmad[selectedind],
                'mederr':medmagerr[selectedind],
                }
        elif selectedreference and observatory=='tess':
            returndict['referencestats'] = {
                'ngood':ngoodobjects[selectedind],
                'magmad':magerrmad[selectedind],
                'mederr':medmagerr[selectedind],
                }
        else:
            returndict['referencestats'] = None

        # add all frame stats to the returndict
        if framestats:
            if observatory=='tess':
                raise NotImplementedError
            returndict['framestats'] = {'frame':goodframes,
                                        'phot':goodphots,
                                        'zenithdist':zenithdist,
                                        'moondist':moondist,
                                        'moonelev':moonelev,
                                        'ngood':ngoodobjects,
                                        'magmad':magerrmad,
                                        'mederr':medmagerr}
        # done with reference frame selection #


    # update the workfitslist and workphotlist if we did reference frame
    # selection
    if (len(goodframes) > 0 and len(goodphots) > 0 and
        len(goodframes) == len(goodphots)):
        returndict['workframelist'] = goodframes
        returndict['workphotlist'] = goodphots

    # write the framelistfile using the new frame locations
    if outlistfile:
        outf = open(outlistfile,'wb')
        for fitsline, photline in zip(workfitslist,workphotlist):
            outf.write('%s %s\n' % (photline, fitsline))
        outf.close()

        print('%sZ: wrote good frame list to %s' %
              (datetime.utcnow().isoformat(), os.path.abspath(outlistfile)))

    print('%sZ: done with fitsdir = %s' %
          (datetime.utcnow().isoformat(), fitsdir))

    # at the end, return returndict
    return returndict



def textphot_links_to_binphot_links(workdir,
                                    binphotdir):
    """
    This is used to convert the fiphot links in workdir (which are text fiphot
    files) to the equivalent binary fiphot links in the same directory. Useful
    only for MagnitudeFitting.py.
    """

    # get a list of text fiphot links
    text_fiphot_links = glob.glob(os.path.join(workdir,'*.fiphot'))

    print('%sZ: found %s text fiphot links in %s' %
          (datetime.utcnow().isoformat(),
           len(text_fiphot_links), workdir))

    # temporarily change working directory to the workdir
    cwd = os.getcwd()
    os.chdir(workdir)

    for textphot in text_fiphot_links:

        binphot = os.path.join(os.path.abspath(binphotdir),
                               os.path.basename(textphot))

        if os.path.exists(binphot):
            print('%sZ: converting textphot link %s -> binphot link for %s' %
                  (datetime.utcnow().isoformat(),
                   os.path.basename(textphot), binphot))
            os.remove(os.path.abspath(textphot))
            os.symlink(binphot, os.path.basename(textphot))
        else:
            print('%sZ: textphot link has no '
                  'equivalent binphot file, removing it!' %
                  (datetime.utcnow().isoformat(),
                   os.path.basename(textphot),))
            os.remove(os.path.abspath(textphot))



def make_magfit_config(configoutfile,
                       phot_apertures=3,
                       phot_band='r',
                       phot_brightmag=0.0,
                       phot_faintmag=16.0,
                       phot_fovcatalog='',
                       singlephot_statdir='',
                       masterphot_statdir='',
                       masterphot_outdir=''):
    """
    This creates a magfit config file.
    """

    # open the output file object
    outf = open(configoutfile,'wb')

    # write the top of the config file
    outf.write('num_apertures=%i\n' % phot_apertures)

    #
    # [mcat] section
    #
    outf.write('\n[mcat]\n')
    outf.write('rawphot_ver=0\n')
    # the mcat stat template entry
    templatestr = phot_fovcatalog
    outf.write("template=Template('%s')\n" % templatestr)
    outf.write('faint_mag=%.1f\n' %  phot_faintmag)
    outf.write('bright_mag=%.1f\n' %  phot_brightmag)
    outf.write('round_pointing=1\n')
    outf.write(("columns="
                "['id','ra','dec','xi','eta','J','K','qlt','I','%s']\n") %
               phot_band)
    outf.write('fov_alarm=20.0\n')
    outf.write('fov_safety_fac=1.1\n')

    #
    # [magfit.single] section
    #
    outf.write('\n[magfit.single]\n')
    # the magfit.single stat template entry
    templatestr = '/'.join(
        [singlephot_statdir,'SPHOTREFSTAT_object_${OBJECT}_${CMPOS}']
        )
    outf.write("stat_template=Template('%s')\n" % templatestr)
    outf.write("first_column=15\n")
    outf.write("reference_subpix=True\n")
    outf.write("file_extension='.sphotref'\n")
    outf.write("column_precision=1.0e-5\n")
    outf.write("column_name='sprmag'\n")
    outf.write("version=0\n")
    outf.write("count_weight=1.0\n")
    outf.write("error_avg='weightedmean'\n")
    outf.write("max_rej_iter=20\n")
    outf.write("rej_level=3.0\n")
    outf.write("max_mag_err=0.03\n")
    outf.write("noise_offset=0.0005\n")
    outf.write("ref_frame_stars=10000\n")
    outf.write("max_JmK=1.0\n")
    outf.write("min_JmK=0.0\n")
    outf.write("bright_mag_min=8.5\n")
    outf.write("AAAonly=True\n")
    outf.write("param_str='spatial:4;r:2,2;JmK:1,2;subpix:1,2'\n")
    outf.write("fntype='linear'\n")

    #
    # [magfit.master] section
    #
    outf.write('\n[magfit.master]\n')
    # the magfit.master stat template entry
    templatestr = '/'.join(
        [masterphot_statdir,'MPHOTREFSTAT_object_${OBJECT}_${CMPOS}']
        )
    outf.write("stat_template=Template('%s')\n" % templatestr)
    outf.write("first_column=15\n")
    outf.write("reference_subpix=False\n")
    outf.write("file_extension='.mphotref'\n")
    outf.write("column_precision=1.0e-5\n")
    outf.write("column_name='mprmag'\n")
    outf.write("version=0\n")
    outf.write("count_weight=1.0\n")
    outf.write("error_avg='weightedmean'\n")
    outf.write("max_rej_iter=20\n")
    outf.write("rej_level=3.0\n")
    outf.write("max_mag_err=0.03\n")
    outf.write("noise_offset=0.0005\n")
    outf.write("ref_frame_stars=10000\n")
    outf.write("max_JmK=1.0\n")
    outf.write("min_JmK=0.0\n")
    outf.write("bright_mag_min=8.5\n")
    outf.write("AAAonly=True\n")
    outf.write("param_str='spatial:4;r:2,2;JmK:1,2;subpix:1,2'\n")
    outf.write("fntype='linear'\n")

    #
    # [mphotref] section
    #
    outf.write('\n[mphotref]\n')
    outf.write('version=0\n')
    # the mphotref template entry
    templatestr = '/'.join(
        [masterphot_outdir,'mphotref_${CMPOS}.AP${APERTURE}']
        )
    outf.write("template=Template('%s')\n" % templatestr)
    outf.write("rms_fit_bright_mag_min=9.0\n")
    outf.write("max_rms_quantile=0.1\n")
    outf.write("max_rms_above_fit=4\n")
    outf.write("rms_fit_rej_lvl=3.0\n")
    outf.write("rms_fit_err_avg=median\n")
    outf.write("rms_fit_param='spatial:2;r:2,2'\n")
    outf.write("grcollect_tempdir='/dev/shm'\n")
    outf.write("min_meas_med=0.9\n")
    outf.write("min_measurements=0.01\n")
    outf.write("rej_iterations=20\n")
    outf.write("rej_outliers='iterations=20,median,meddev=8'\n")

    outf.close()

    print('%sZ: wrote magfit config to %s' %
          (datetime.utcnow().isoformat(),
           os.path.abspath(configoutfile),))

    return configoutfile



def make_fiphot_list(searchdirs,
                     searchglob,
                     listfile):
    """
    This makes a list of fiphot files in the list of directories specified in
    searchdirs, using the searchglob to filter by filename. Returns a list of
    absolute paths to the fiphot files and writes this to the file specified in
    listfile.
    """

    fiphotlist = []

    for fdir in searchdirs:
        fiphotfiles = glob.glob(os.path.join(fdir, searchglob))
        fiphotlist.extend([os.path.abspath(x) for x in fiphotfiles])

    # write to the output file
    outf = open(listfile,'wb')

    for fdir in fiphotlist:
        outf.write('%s\n' % fdir)

    outf.close()

    return listfile



def run_magfit(sphotref_frame,
               sphotref_phot,
               magfit_frame_list,
               magfit_config_file,
               magfit_type,
               nprocs=16,
               hatnetwork='HATSouth',
               magfitexec='MagnitudeFitting.py'):
    """
    This runs magfit in single/master photometric reference mode.

    lcohpsrv1 invocation:

    single photref

    python /home/hatuser/wbhatti/src/MagnitudeFittingOrig.py HATSouth single /nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/G557-ccd5-work/1-470798a_5.fits /nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/G557-ccd5-work/1-470798a_5.fiphot -p 8 --log-config=/home/hatuser/wbhatti/src/logging.conf --config-file=/nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/ccd5-magfit.cfg --manual-frame-list=/nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/ccd5-magfit-frames.list --stat

    master photref:

    nohup python /home/hatuser/wbhatti/src/MagnitudeFittingOrig.py HATSouth master /nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/G557-ccd6-work/1-470789a_6.fits /nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/G557-ccd6-work/1-470789a_6.fiphot -p 8 --log-config=/home/hatuser/wbhatti/src/logging.conf --config-file=/nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/ccd6-magfit.cfg --manual-frame-list=/nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid16/photometry-ap/ccd6-magfit-frames.list --stat > ccd6-mmagfit.log
    """

    if not (os.path.exists(magfit_frame_list) and
            os.path.exists(sphotref_frame) and
            os.path.exists(sphotref_phot) and
            os.path.exists(magfit_config_file)):

        print('%sZ: some required files are missing!' %
              (datetime.utcnow().isoformat(),))
        return None

    magfitcmd = MAGFITCMD.format(
        magfitexec=os.path.abspath(magfitexec),
        network=hatnetwork,
        fit_type=magfit_type,
        sphotref_frame=sphotref_frame,
        sphotref_phot=sphotref_phot,
        nprocs=nprocs,
        magfit_config_file=magfit_config_file,
        magfit_frame_list=magfit_frame_list
        )

    if DEBUG:
        print(magfitcmd)

    print('%sZ: starting %s magfit with %s processes...' %
          (datetime.utcnow().isoformat(), magfit_type, nprocs))

    # execute the magfit shell command
    magfitproc = subprocess.Popen(shlex.split(magfitcmd),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

    # get results
    magfit_stdout, magfit_stderr = magfitproc.communicate()

    # get results if succeeded, log outcome, and return path of outfile
    if magfitproc.returncode == 0:

        print('%sZ: %s magfit completed.' %
              (datetime.utcnow().isoformat(), magfit_type))

        return True

    else:

        print('%sZ: %s magfit failed!' %
              (datetime.utcnow().isoformat(), magfit_type))
        print('%sZ: error returned was %s' % magfit_stderr)

        return False



def get_master_photref(sphotref_frame,
                       fiphot_list,
                       magfit_config_file,
                       hatnetwork='HATSouth',
                       photrefexec='do_masterphotref.py'):
    """
    Generates the master photometric reference from single ref photometry.

    lcohpsrv1 invocation:

    nohup python /home/hatuser/wbhatti/src/do_masterphotref.py HATSouth /nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid8/ccd5-fits/1-404411d_5.fits --manual-frame-list=/nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid8/photometry-ap/ccd5-fiphot.list --config-file=/nfs/lcohpsrv1/ar1/scratch/PHOT_WB/projid8/photometry-ap/ccd5-magfit.cfg --log-config=/home/hatuser/wbhatti/src/logging.conf --nostat > ccd5-masterphotref.log &
    """

    if not (os.path.exists(fiphot_list) and
            os.path.exists(sphotref_frame) and
            os.path.exists(magfit_config_file)):

        print('%sZ: some required files are missing!' %
              (datetime.utcnow().isoformat(),))
        return None

    photrefcmd = MPHOTREFCMD.format(
        mphotrefexec=os.path.abspath(photrefexec),
        network=hatnetwork,
        sphotref_frame=sphotref_frame,
        fiphot_list=fiphot_list,
        magfit_config_file=magfit_config_file
        )

    if DEBUG:
        print(photrefcmd)

    print('%sZ: starting photref...' %
          (datetime.utcnow().isoformat(),))

    # execute the photref shell command
    photrefproc = subprocess.Popen(shlex.split(photrefcmd),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

    # get results
    photref_stdout, photref_stderr = photrefproc.communicate()

    # get results if succeeded, log outcome, and return path of outfile
    if photrefproc.returncode == 0:

        print('%sZ: photref completed.' %
              (datetime.utcnow().isoformat(),))

        return True

    else:

        print('%sZ: photref failed!' %
              (datetime.utcnow().isoformat(),))

        return False



###########################
## FIPHOT DUMP FUNCTIONS ##
###########################

def dump_binary_fiphot(fiphot,
                       sourcelist,
                       outfile):
    """
    This dumps all columns from a fiphot binary format file to a text fiphot
    file. This also needs the sourcelist file for the same frame to get the S,
    D, K values correctly for each detection. This assumes that the sourcelist
    source detections are ordered in the same way as the source detections in
    the fiphot file (this appears to be valid, but need a workaround).

    keys to dump and in which order:

    HAT-field-sourceid
    serial
    x
    y
    bg
    bg err
    per aperture[0] mag
    per aperture[0] mag err
    per aperture[0] status flag
    per aperture[1] mag
    per aperture[1] mag err
    per aperture[1] status flag
    per aperture[2] mag
    per aperture[2] mag err
    per aperture[2] status flag
    mprmag[0]
    mprmag[1]
    mprmag[2]

    NOTE: each line has a length of 210 characters (this will be useful as input
    to the fast parallel LC collection function in imagesubphot.py).
    """

    # first, read the fiphot in
    binphot = read_fiphot(fiphot)

    srclist = np.genfromtxt(sourcelist,
                            usecols=(8,9,10),
                            dtype='f8,f8,f8',
                            names=['fsv','fdv','fkv'])

    # get all the columns

    source = binphot['source']
    serial = binphot['serial']
    field = binphot['field']

    srcx = binphot['x']
    srcy = binphot['y']

    bkg = binphot['bg']
    bkgerr = binphot['bg err']

    im1 = binphot['per aperture'][0]['mag']
    ie1 = binphot['per aperture'][0]['mag err']
    iq1 = binphot['per aperture'][0]['status flag']

    im2 = binphot['per aperture'][1]['mag']
    ie2 = binphot['per aperture'][1]['mag err']
    iq2 = binphot['per aperture'][1]['status flag']

    im3 = binphot['per aperture'][2]['mag']
    ie3 = binphot['per aperture'][2]['mag err']
    iq3 = binphot['per aperture'][2]['status flag']

    rm1 = binphot['mprmag[0]'] if 'mprmag[0]' in binphot else [np.nan for x in srcx]
    rm2 = binphot['mprmag[1]'] if 'mprmag[1]' in binphot else [np.nan for x in srcx]
    rm3 = binphot['mprmag[2]'] if 'mprmag[2]' in binphot else [np.nan for x in srcx]

    # format the output line
    lineform = (
        'HAT-%3i-%07i %12s '      # hatid, rstfc
        '%12.5f %12.5f '          # srcx, srcy
        '%12.5f %12.5f '          # bkg, bkgerr
        '%12.5f %12.5f %12.5f '   # fsv, fdv, fkv
        '%12.5f %12.5f %3i '      # im1, ie1, iq1
        '%12.5f %12.5f %3i '      # im2, ie2, iq2
        '%12.5f %12.5f %3i '      # im3, ie3, iq3
        '%12.5f %12.5f %12.5f\n'  # rm1, rm2, rm3
        )

    # open the outfile
    outf = open(outfile, 'wb')

    for ind in xrange(len(srcx)):

        outf.write(lineform % (field[ind], source[ind], serial,
                               srcx[ind], srcy[ind],
                               bkg[ind], bkgerr[ind],
                               srclist['fsv'][ind],
                               srclist['fdv'][ind],
                               srclist['fkv'][ind],
                               im1[ind], ie1[ind], iq1[ind],
                               im2[ind], ie2[ind], iq2[ind],
                               im3[ind], ie3[ind], iq3[ind],
                               rm1[ind], rm2[ind], rm3[ind]))

    outf.close()

    return outfile



def dump_binary_worker(task):
    """
    This is a worker for parallelization of binary fiphot dumping.

    task[0] -> path to input binary fiphot
    task[1] -> path to accompanying sourcelist file
    task[2] -> output directory
    task[3] -> output fiphot extension to use
    """

    try:

        outbasename = task[0].replace('fiphot',task[3])
        outfile = os.path.join(os.path.abspath(task[2]), outbasename)

        print('%sZ: binary fiphot %s -> text fiphot %s OK' %
              (datetime.utcnow().isoformat(), task[0], outfile))

        return task[0], dump_binary_fiphot(task[0], task[1], outfile)

    except Exception as e:

        print('ERR! %sZ: could not dump '
              'binary fiphot %s to text fiphot, error was: %s' %
              (datetime.utcnow().isoformat(), task[0], e))

        return task[0], None



def parallel_dump_binary_fiphots(fiphotdir,
                                 fiphotglob='*.fiphot',
                                 sourcelistdir=None,
                                 sourcelistext='.sourcelist',
                                 outdir=None,
                                 textfiphotext='text-fiphot',
                                 nworkers=16,
                                 maxworkertasks=1000):
    """
    This dumps all binary fiphots found in fiphotdir (we check if the file is
    binary or not) to text fiphots with all the same row lengths in outdir. This
    is needed if we want to use the fast LC collection method implemented in
    imagesubphot.py.
    """

    if not sourcelistdir:
        sourcelistdir = fiphotdir

    fiphotlist = glob.glob(os.path.join(os.path.abspath(fiphotdir), fiphotglob))
    fiphotext = os.path.splitext(fiphotglob)[-1]

    sourcelistlist = [x.replace(fiphotext, sourcelistext) for x in fiphotlist]

    print('%sZ: %s files to process in %s' %
          (datetime.utcnow().isoformat(), len(fiphotlist), fiphotdir))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    if not outdir:
        outdir = fiphotdir

    tasks = [(x, y, outdir, textfiphotext)
             for (x,y) in zip(fiphotlist, sourcelistlist)]

    # fire up the pool of workers
    results = pool.map(dump_binary_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}


#############################
## NEW STYLE LC COLLECTION ##
#############################

def make_photometry_indexdb(framedir,
                            outfile,
                            frameglob='*_5.fits',  # avoid ISM FITS products
                            photdir=None,
                            photext='text-fiphot',
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

        phot = '%s-%s_%s.%s' % (frameinfo[0][0],
                                frameinfo[0][1],
                                frameinfo[0][2],
                                photext)
        originalframe = '%s-%s_%s.fits' % (frameinfo[0][0],
                                frameinfo[0][1],
                                frameinfo[0][2])

        phot = os.path.join(os.path.abspath(photdir), phot)
        originalframe = os.path.join(os.path.abspath(framedir),
                                     originalframe)

        # check these files exist, and populate the dict if they do
        if os.path.exists(phot) and os.path.exists(originalframe):


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


def get_fiphot_line(fiphot, linenum, fiphotlinechars=249):
    """
    This gets a random fiphot line out of the file fiphot.
    """

    fiphotf = open(fiphot, 'rb')
    filelinenum = fiphotlinechars*linenum
    fiphotf.seek(filelinenum)
    fiphotline = fiphotf.read(fiphotlinechars)
    fiphotf.close()

    return fiphotline


def get_fiphot_line_linecache(fiphot, linenum, fiphotlinechars=249):
    """
    This uses linecache's getline function to get the line out of the file
    fiphot.
    """

    return getline(fiphot, linenum)


def collect_aperturephot_lightcurve(hatid,
                                    photindex,
                                    outdir,
                                    skipcollected=True,
                                    fiphotlinefunc=get_fiphot_line,
                                    fiphotlinechars=249):
    """
    This collects the imagesubphot lightcurve of a single object into a .ilc
    file.

    hatid -> the hatid of the object to collect the light-curve for

    photindexfile -> the file containing the master index of which .fiphot,
                      .sourcelist, and .fits contain the lines corresponding to
                      this HATID. this way, we can look these lines up
                      super-fast using the linecache module.

    outdir -> the directory where to the place the collected lightcurve

    skipcollected -> if True, looks for an existing LC for this hatid in
                       outdir. if found, returns the path to that LC instead of
                       actually processing. if this is False, redoes the
                       processing for this LC anyway.

    fiphotlinefunc -> this is the function to use for getting a specific line
                     out of the specified fiphot file.

    The collected LC is similar to the aperturephot LC, but some extra columns
    added by fiphot running on the subtracted frames. columns are:

    00 rjd    Reduced Julian Date (RJD = JD - 2400000.0)
    01 hat    HAT ID of the object
    02 rstfc  Unique frame key ({STID}-{FRAMENUMBER}_{CCDNUM})
    03 xcc    original X coordinate on CCD
    04 ycc    original y coordinate on CCD
    05 bgv    Background value
    06 bge    Background measurement error
    07 fsv    Measured S value
    08 fdv    Measured D value
    09 fkv    Measured K value
    10 im1   Instrumental magnitude in aperture 1
    11 ie1   Instrumental magnitude error for aperture 1
    12 iq1   Instrumental magnitude quality flag for aperture 1 (0/G OK, X bad)
    13 im2   Instrumental magnitude in aperture 2
    14 ie2   Instrumental magnitude error for aperture 2
    15 iq2   Instrumental magnitude quality flag for aperture 2 (0/G OK, X bad)
    16 im3   Instrumental magnitude in aperture 3
    17 ie3   Instrumental magnitude error for aperture 3
    18 iq3   Instrumental magnitude quality flag for aperture 3 (0/G OK, X bad)
    19 rm1    Reduced Mags from magfit in aperture 1
    20 rm2    Reduced Mags from magfit in aperture 2
    21 rm3    Reduced Mags from magfit in aperture 3
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

    if rows and len(rows) > 0:

        # prepare the output file
        outfile = os.path.join(os.path.abspath(outdir), '%s.rlc' % hatid)

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

                # next, get the requested line from phot file
                phot_elem = fiphotlinefunc(
                    os.path.join(photdir, phot),
                    photline,
                    fiphotlinechars=fiphotlinechars
                    ).split()

                # parse these lines and prepare the output
                # rstfc_elems = FRAMEREGEX.findall(os.path.basename(phot))
                # rstfc = '%s-%s_%s' % (rstfc_elems[0])
                out_line = '%s %s\n' % (framerjd, ' '.join(phot_elem))
                outf.write(out_line)

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

    # if the hatid isn't found in the photometry index, then we can't do
    # anything
    else:

        print('ERR! %sZ: object %s is not in the '
              'photometry index, ignoring...' %
              (datetime.utcnow().isoformat(), hatid))

        returnf = None

    # at the end, close the DB and return
    indexdb.close()
    return returnf


def aperturephotlc_collection_worker(task):
    """
    This wraps collect_aperurephot_lightcurve for parallel_collect_lightcurves
    below.

    task[0] -> hatid
    task[1] -> photindex DB name
    task[2] -> outdir
    task[3] -> {skipcollected, fiphotlinefunc, fiphotlinechars}
    """

    try:

        return task[0], collect_aperturephot_lightcurve(task[0],
                                                        task[1],
                                                        task[2],
                                                        **task[3])

    except Exception as e:

        print('ERR! %sZ: failed to get LC for %s, error: %s' %
              (datetime.utcnow().isoformat(), task[0], e ))
        return task[0], None


def parallel_collect_aperturephot_lightcurves(framedir,
                                              outdir,
                                              frameglob='*_5.fits',
                                              photindexdb=None,
                                              photdir=None,
                                              photext='text-fiphot',
                                              maxframes=None,
                                              overwritephotindex=False,
                                              skipcollectedlcs=True,
                                              fiphotlinefunc=get_fiphot_line,
                                              fiphotlinechars=249,
                                              nworkers=16,
                                              maxworkertasks=1000):
    """
    This collects all .fiphot files into lightcurves.
    """

    # first, check if the output directory exists
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # next, check if we have to make a photometry index DB, and launch the
    if not photindexdb:

        photdbf = os.path.join(framedir,'TM-aperturephot-index.sqlite')

        photindexdb = make_photometry_indexdb(framedir,
                                              photdbf,
                                              frameglob=frameglob,
                                              photdir=photdir,
                                              photext=photext,
                                              maxframes=maxframes,
                                              overwrite=overwritephotindex)

    # only proceed if the photometry index DB exists
    if os.path.exists(photindexdb):

        # get the list of distinct HATIDs from the photindexdb
        db = sqlite3.connect(photindexdb)

        cur = db.cursor()
        cur.execute(DISTINCT_HATIDS_CMD)
        rows = cur.fetchall()
        # make sure these are unique
        hatids = [x[0].strip() for x in rows]
        hatids = list(set(hatids))

        db.close()

        # generate the task list
        tasks = [(hatid,
                  photindexdb,
                  outdir,
                  {'skipcollected':skipcollectedlcs,
                   'fiphotlinefunc':fiphotlinefunc,
                   'fiphotlinechars':fiphotlinechars}) for hatid in hatids]

        # now start up the parallel collection
        print('%sZ: %s HATIDs to get LCs for, starting...' %
              (datetime.utcnow().isoformat(), len(hatids), ))
        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        # fire up the pool of workers
        results = pool.map(aperturephotlc_collection_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}

    # if the photometry index DB doesn't exist, nothing we can do
    else:

        print('ERR! %sZ: specified photometry index DB does not exist!' %
              (datetime.utcnow().isoformat(), ))


###################
## EPD FUNCTIONS ##
###################

def epd_diffmags(coeff, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag):
    """
    This calculates the difference in mags after EPD coefficients are
    calculated.

    final EPD mags = median(magseries) + epd_diffmags()
    """

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
             coeff[18]*bgv +
             coeff[19]*bge -
             mag)


def epd_magseries(mag, fsv, fdv, fkv, xcc, ycc, bgv, bge,
                  smooth=21, sigmaclip=3.0):
    """
    Detrends a magnitude series given in mag using accompanying values of S in
    fsv, D in fdv, K in fkv, x coords in xcc, y coords in ycc, background in
    bgv, and background error in bge. smooth is used to set a smoothing
    parameter for the fit function. Does EPD voodoo.
    """

    # find all the finite values of the magnitude
    finiteind = np.isfinite(mag)

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

    # make the linear equation matrix
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
                      bgv[finalind],
                      bge[finalind]]

    # solve the equation epdmatrix * x = smoothedmag
    # return the EPD differential mags if the solution succeeds
    try:

        coeffs, residuals, rank, singulars = lstsq(epdmatrix, smoothedmag)

        if DEBUG:
            print('coeffs = %s, residuals = %s' % (coeffs, residuals))

        return epd_diffmags(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag)

    # if the solution fails, return nothing
    except Exception as e:

        print('%sZ: EPD solution did not converge! Error was: %s' %
              (datetime.utcnow().isoformat(), e))
        return None


def epd_lightcurve(rlcfile,
                   mags=[19,20,21],
                   sdk=[7,8,9],
                   xy=[3,4],
                   backgnd=[5,6],
                   smooth=21,
                   sigmaclip=3.0,
                   rlcext='rlc',
                   outfile=None,
                   minndet=200):
    """
    Runs the EPD process on rlcfile, using columns specified to get the required
    parameters. If outfile is None, the .epdlc will be placeed in the same
    directory as rlcfile.
    """

    # read the lightcurve in
    rlc = np.genfromtxt(rlcfile,
                        usecols=tuple(xy + backgnd + sdk + mags),
                        dtype='f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                        names=['xcc','ycc','bgv','bge','fsv','fdv','fkv',
                               'rm1','rm2','rm3'])

    if len(rlc['xcc']) >= minndet:

        # calculate the EPD differential mags
        epddiffmag1 = epd_magseries(rlc['rm1'],rlc['fsv'],rlc['fdv'],rlc['fkv'],
                                    rlc['xcc'],rlc['ycc'],rlc['bgv'],rlc['bge'],
                                    smooth=smooth, sigmaclip=sigmaclip)
        epddiffmag2 = epd_magseries(rlc['rm2'],rlc['fsv'],rlc['fdv'],rlc['fkv'],
                                    rlc['xcc'],rlc['ycc'],rlc['bgv'],rlc['bge'],
                                    smooth=smooth, sigmaclip=sigmaclip)
        epddiffmag3 = epd_magseries(rlc['rm3'],rlc['fsv'],rlc['fdv'],rlc['fkv'],
                                    rlc['xcc'],rlc['ycc'],rlc['bgv'],rlc['bge'],
                                    smooth=smooth, sigmaclip=sigmaclip)

        # add the EPD diff mags back to the median mag to get the EPD mags
        if epddiffmag1 is not None:
            mag_median = np.median(rlc['rm1'][np.isfinite(rlc['rm1'])])
            epdmag1 = epddiffmag1 + mag_median
        else:
            epdmag1 = np.array([np.nan for x in rlc['rm1']])
            print('%sZ: no EP1 mags available for %s!' %
                  (datetime.utcnow().isoformat(), rlcfile))

        if epddiffmag2 is not None:
            mag_median = np.median(rlc['rm2'][np.isfinite(rlc['rm2'])])
            epdmag2 = epddiffmag2 + mag_median
        else:
            epdmag2 = np.array([np.nan for x in rlc['rm2']])
            print('%sZ: no EP2 mags available for %s!' %
                  (datetime.utcnow().isoformat(), rlcfile))

        if epddiffmag3 is not None:
            mag_median = np.median(rlc['rm3'][np.isfinite(rlc['rm3'])])
            epdmag3 = epddiffmag3 + mag_median
        else:
            epdmag3 = np.array([np.nan for x in rlc['rm3']])
            print('%sZ: no EP3 mags available for %s!' %
                  (datetime.utcnow().isoformat(), rlcfile))

        # now write the EPD LCs out to the outfile
        if not outfile:
            outfile = '%s.epdlc' % re.sub('.%s' % rlcext, '', rlcfile)

        inf = open(rlcfile,'rb')
        inflines = inf.readlines()
        inf.close()
        outf = open(outfile,'wb')

        for line, epd1, epd2, epd3 in zip(inflines, epdmag1, epdmag2, epdmag3):
            outline = '%s %.6f %.6f %.6f\n' % (line.rstrip('\n'), epd1, epd2, epd3)
            outf.write(outline)

        outf.close()
        return outfile

    else:
        print('not running EPD for %s, ndet = %s < min ndet = %s' %
              (rlcfile, len(rlc['xcc']), minndet))
        return None


def parallel_epd_worker(task):
    """
    Function to wrap the epd_lightcurve function for use with mp.Pool.

    task[0] = rlcfile
    task[1] = {'mags', 'sdk', 'xy', 'backgnd',
               'smooth', 'sigmaclip', 'rlcext',
               'minndet'}
    """

    try:
        return task[0], epd_lightcurve(task[0], **task[1])
    except Exception as e:
        print('EPD failed for %s, error was: %s' % (task[0], e))
        return task[0], None


def parallel_run_epd(rlcdir,
                     mags=[19,20,21],
                     sdk=[7,8,9],
                     xy=[3,4],
                     backgnd=[5,6],
                     smooth=21,
                     sigmaclip=3.0,
                     rlcext='rlc',
                     rlcglobprefix='*',
                     outfile=None,
                     nworkers=16,
                     maxworkertasks=1000,
                     minndet=200):
    """
    This runs EPD in parallel on the lightcurves in rlcdir.
    """

    # find all the rlc files in the rlcdir
    rlclist = glob.glob(os.path.join(os.path.abspath(rlcdir), '%s.%s' %
                                     (rlcglobprefix, rlcext)))

    tasks = [(x, {'mags':mags, 'sdk':sdk, 'xy':xy, 'backgnd':backgnd,
                  'smooth':smooth, 'sigmaclip':sigmaclip, 'rlcext':rlcext,
                  'minndet':minndet})
             for x in rlclist]

    # now start up the parallel EPD processes
    print('%sZ: %s HATIDs for EPD, starting...' %
          (datetime.utcnow().isoformat(), len(rlclist), ))
    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(parallel_epd_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}


###################
## TFA FUNCTIONS ##
###################

def choose_tfa_template(statsfile,
                        fovcatalog,
                        epdlcdir,
                        ignoretfamin=False,
                        fovcat_idcol=0,
                        fovcat_xicol=3,
                        fovcat_etacol=4,
                        fovcathasgaiaids=False,
                        fovcat_magcol=9,
                        max_nstars=1000,
                        min_nstars=20,
                        target_nstars=None,
                        brightest_mag=8.5,
                        faintest_mag=12.0,
                        max_rms=0.1,
                        max_sigma_above_rmscurve=4.0,
                        outprefix=None,
                        tfastage1=True,
                        epdlcext='.epdlc'):
    """
    This chooses suitable stars for TFA template purposes. This "template set"
    is a subsample of the stars, and is supposed to represent all the types of
    systematics across the dataset. Kovacs et al (2005) give details.

    statsfile = the file with LC statistics made when running EPD

    fovcatalog = the fovcatalog file, this must have xi and eta coordinates,
                 ras, decs, and magnitudes

    Returns a dict with lists of stars chosen, their stats, and filenames of
    where each star list was written.
    """

    # read in the stats file
    stats = read_stats_file(statsfile, fovcathasgaiaids=fovcathasgaiaids)

    # read in the fovcatalog
    if not fovcathasgaiaids:
        # assume HAT-IDs, HAT-123-4567890, 17 character strings
        fovcat = np.genfromtxt(fovcatalog,
                               usecols=(fovcat_idcol,
                                        fovcat_xicol,
                                        fovcat_etacol,
                                        fovcat_magcol),
                               dtype='U17,f8,f8,f8',
                               names=['objid','xi','eta','mag'])
        staridstr = 'HAT-'
    else:
        # assume GAIA-IDs. From gaia2read, with "GAIA" id option, this is just
        # 19 character integers. The (xi,eta) and mag precision also change.
        fovcat = np.genfromtxt(fovcatalog,
                               usecols=(fovcat_idcol,
                                        fovcat_xicol,
                                        fovcat_etacol,
                                        fovcat_magcol),
                               dtype='U19,f8,f8,f8',
                               names=['objid','xi','eta','mag'])
        staridstr = '' # no pre-identifier for Gaia IDs.

    # figure out the number of stars to use in the initial TFA template
    # number of stars = TFA_TEMPLATE_FRACTION * median ndet

    # 1. ndet >= median_ndet
    # 2. max rms <= 0.1
    # 3. brightest_mag < median_mag < faintest_mag
    # 4. fit rms-mag, then discard anything above max_sigma_above_rmscurve
    # find the objects in the fovcat and stats file that match these
    # conditions, then pick up to 1000 random stars

    outdict = {'statsfile':os.path.abspath(statsfile),
               'fovcat':os.path.abspath(fovcatalog),
               'lcdir':os.path.abspath(epdlcdir),
               'staridstr':staridstr}

    # do this per aperture
    for aperture in [1,2,3]:

        outdict[aperture] = {}

        # first, pick the stars that meet our stats requirements

        epdstr = 'ep%s' % aperture

        objid_col = 'lcobj'
        median_mag_col = 'med_sc_%s' % epdstr
        mad_mag_col = 'mad_sc_%s' % epdstr
        ndet_col = 'ndet_sc_%s' % epdstr

        objectid = stats[objid_col]
        mags_median = stats[median_mag_col]
        mags_mad = stats[mad_mag_col]
        obj_ndet = stats[ndet_col]

        goodind = np.isfinite(mags_median) & np.isfinite(mags_mad)
        objectid = objectid[goodind]
        mags_median = mags_median[goodind]
        mags_mad = mags_mad[goodind]
        obj_ndet = obj_ndet[goodind]

        print('\naperture %s: total good objects = %s' % (aperture,
                                                        len(objectid)))

        median_ndet = np.nanmedian(obj_ndet)
        if not target_nstars:
            TFA_TEMPLATE_FRACTION = 0.1
            target_nstars = TFA_TEMPLATE_FRACTION * median_ndet
        else:
            pass

        print('aperture %s: median ndet = %s' % (aperture, median_ndet))
        print('aperture %s: target TFA template size = %s' %
              (aperture, int(target_nstars)))
        outdict[aperture]['target_tfa_nstars'] = (
            target_nstars
        )

        stars_ndet_condition = obj_ndet >= median_ndet
        print('aperture %s: objects with ndet condition = %s' %
              (aperture, len(objectid[stars_ndet_condition])))

        stars_rms_condition = mags_mad < max_rms
        print('aperture %s: objects with rms condition = %s' %
              (aperture, len(objectid[stars_rms_condition])))

        rmsfit_condition = stars_ndet_condition
        print('aperture %s: objects with rmsfit condition = %s' %
              (aperture, len(objectid[rmsfit_condition])))

        # selection 1: fit a parabola to the median mag - mag MAD relation and
        #              reject all stars with RMS > max_sigma_above_rmscurve
        polyfit_coeffs = np.polyfit(mags_median[rmsfit_condition],
                             mags_mad[rmsfit_condition],
                             2)

        print('aperture %s: rms fit params = %s' % (aperture,polyfit_coeffs))

        # generate the model RMS curve with fit parameters
        model_rms = (polyfit_coeffs[0]*mags_median*mags_median +
                     polyfit_coeffs[1]*mags_median +
                     polyfit_coeffs[2])

        # find objects that lie below the requested rms threshold from this
        # curve
        threshold_condition = (mags_mad/model_rms) < max_sigma_above_rmscurve
        print('aperture %s: objects with threshold condition = %s' %
              (aperture, len(objectid[threshold_condition])))

        final_statistics_ind = (threshold_condition &
                                stars_rms_condition &
                                stars_ndet_condition)

        print('aperture %s: stars with good stats = %s' % (
            aperture,
            len(objectid[final_statistics_ind])
        ))

        good_stats_objects = objectid[final_statistics_ind]

        # selection 2: get the stars with good magnitudes
        mag_condition = ((fovcat['mag'] < faintest_mag) &
                         (fovcat['mag'] > brightest_mag))
        good_mag_objects = fovcat['objid'][mag_condition]
        print('aperture %s: stars with good mags = %s' %
              (aperture,len(good_mag_objects)))

        # finally, intersect these two arrays to find a set of good TFA objects
        good_tfa_objects = np.intersect1d(good_stats_objects,
                                          good_mag_objects)
        print('aperture %s: stars suitable for TFA = %s' %
              (aperture,len(good_tfa_objects)))

        # put this initial list into the outdict
        outdict[aperture]['tfa_suitable_objects'] = good_tfa_objects

        # selection 3: pick the target number of stars for TFA. Note
        # target_nstars can be larger than max_nstars, in which case max_nstars
        # template stars are chosen.
        if target_nstars > max_nstars:
            if len(good_tfa_objects) > max_nstars:
                tfa_stars = nprand.choice(good_tfa_objects, replace=False,
                                          size=max_nstars)
            elif len(good_tfa_objects) > min_nstars:
                tfa_stars = good_tfa_objects
            else:
                print("aperture %s: not enough stars suitable for TFA!" %
                      aperture)
                if not ignoretfamin:
                    tfa_stars = None
                else:
                    tfa_stars = good_tfa_objects
        else:
            if len(good_tfa_objects) > target_nstars:
                tfa_stars = nprand.choice(good_tfa_objects, replace=False,
                                          size=target_nstars)
            elif len(good_tfa_objects) > min_nstars:
                tfa_stars = good_tfa_objects
            else:
                print("aperture %s: not enough stars suitable for TFA!" %
                      aperture)
                if not ignoretfamin:
                    tfa_stars = None
                else:
                    tfa_stars = good_tfa_objects

        # now get these stars IDs, LC fnames, xis, etas, and other things
        # needed for the first stage of TFA (this will choose exactly
        # target_nstars to use as the template for the final stage of TFA)
        if tfa_stars is not None:

            print('aperture %s: %s objects chosen as TFA templates for stage 1' %
                  (aperture,len(tfa_stars)))
            outdict[aperture]['tfa_chosen_objects'] = tfa_stars

            tfa_stars_catmag = []
            tfa_stars_statrms = []
            tfa_stars_statndet = []
            tfa_stars_xi = []
            tfa_stars_eta = []
            tfa_stars_lcfile = []

            print('aperture %s: getting object info...' %
                  (aperture,))

            # get the stats for these objects
            for i, tfaobj in enumerate(tfa_stars):

                # search for the LC file for this object and make sure it exists
                lcfile_searchpath = os.path.join(
                    epdlcdir, '{:s}{:s}'.format(tfaobj,epdlcext)
                )

                if os.path.exists(lcfile_searchpath):

                    tfa_stars_lcfile.append(
                        os.path.abspath(lcfile_searchpath)
                    )
                    tfa_stars_catmag.append(
                        fovcat['mag'][fovcat['objid'] == tfaobj]
                    )
                    tfa_stars_statrms.append(
                        mags_mad[objectid == tfaobj]
                    )
                    tfa_stars_statndet.append(
                        obj_ndet[objectid == tfaobj]
                    )
                    tfa_stars_xi.append(
                        fovcat['xi'][fovcat['objid'] == tfaobj]
                    )
                    tfa_stars_eta.append(
                        fovcat['eta'][fovcat['objid'] == tfaobj]
                    )

                # if it doesn't, then add nans to the file
                else:

                    print('ERR! couldn\'t find a LC for %s' % tfaobj)

                    tfa_stars_lcfile.append(None)
                    tfa_stars_catmag.append(np.nan)
                    tfa_stars_statrms.append(np.nan)
                    tfa_stars_statndet.append(np.nan)
                    tfa_stars_xi.append(np.nan)
                    tfa_stars_eta.append(np.nan)

            outdict[aperture]['tfa_chosen_lcfile'] = tfa_stars_lcfile
            outdict[aperture]['tfa_chosen_mag'] = np.ravel(tfa_stars_catmag)
            outdict[aperture]['tfa_chosen_rms'] = np.ravel(tfa_stars_statrms)
            outdict[aperture]['tfa_chosen_ndet'] = np.ravel(tfa_stars_statndet)
            outdict[aperture]['tfa_chosen_xi'] = np.ravel(tfa_stars_xi)
            outdict[aperture]['tfa_chosen_eta'] = np.ravel(tfa_stars_eta)

        # if no TFA stars could be chosen, return Nones for this aperture
        else:
            outdict[aperture]['tfa_chosen_lcfile'] = None
            outdict[aperture]['tfa_chosen_objects'] = None
            outdict[aperture]['tfa_chosen_mag'] = None
            outdict[aperture]['tfa_chosen_rms'] = None
            outdict[aperture]['tfa_chosen_ndet'] = None


        # make the input file for TFA stage 1 for this aperture
        if not outprefix:
            outfile = os.path.abspath(
                os.path.join(os.getcwd(),
                             'tfa-stage1-input-aperture-%s.txt' % aperture)
            )
        else:
            outfile = os.path.abspath(
                os.path.join(outprefix,
                             'tfa-stage1-input-aperture-%s.txt' % aperture)
            )


        outf = open(outfile,'wb')

        outline = '%s %s %.6f %.6f %i %.6f %.6f\n'

        for objid, lcf, mag, rms, ndet, xi, eta in zip(
                outdict[aperture]['tfa_chosen_objects'],
                outdict[aperture]['tfa_chosen_lcfile'],
                outdict[aperture]['tfa_chosen_mag'],
                outdict[aperture]['tfa_chosen_rms'],
                outdict[aperture]['tfa_chosen_ndet'],
                outdict[aperture]['tfa_chosen_xi'],
                outdict[aperture]['tfa_chosen_eta']
        ):
            outf.write(
                (
                outline % (objid, lcf, mag, rms, ndet, xi, eta)
                ).encode('utf-8')
            )

        outf.close()
        print('aperture %s: wrote object info to %s' %
              (aperture, outfile))
        outdict[aperture]['info_file'] = os.path.abspath(outfile)

        # END OF PER APERTURE STUFF


    # run TFAs stage 1 if we're supposed to do so
    if tfastage1:

        print('\nrunning TFA stage 1...')

        # run TFA stage 1 to pick the good objects
        tfa_stage1 = run_tfa_stage1(outdict)

        for aperture in tfa_stage1:

            outdict[aperture]['stage1_templatelist'] = tfa_stage1[aperture]

            # make the TFA template list file for this aperture
            if outdict[aperture]['stage1_templatelist']:
                templatelistfname = os.path.join(
                    outdict['lcdir'],
                    'aperture-%s-tfa-template.list' % aperture
                )
                outf = open(templatelistfname,'wb')
                for tfaobjid in outdict[aperture]['stage1_templatelist']:

                    templatelc = os.path.join(
                        outdict['lcdir'],
                        tfaobjid + '.epdlc'
                    )

                    if os.path.exists(templatelc):
                        outf.write(
                            ('%s\n' % os.path.abspath(templatelc)
                            ).encode('utf-8')
                        )

                outf.close()
                outdict[aperture]['stage1_tfa_templatefile'] = (
                    templatelistfname
                )
                print('aperture %s: wrote TFA template list to %s' %
                      (aperture, templatelistfname))


    return outdict


def run_tfa_stage1(tfainfo):
    """
    This just runs the TFA in fake mode to generate a list of template
    stars. Uses the tfainfo dict created in choose_tfa_template above.

    Communicates with the tfa program over pipe and then writes its output to a
    tfa input file for the next stage.

       tfa -r <REFFILE> --col-ref-id <REFFILE_IDCOLNUM>
                        --col-ref-x <REFFILE_XCOLNUM>
                        --col-ref-y <REFFILE_YCOLNUM>
                        -n <NTEMPLATES_TO_USE>
                        -T - (for stdout)
                        -i /dev/null (no input file?)
    """

    staridstr = tfainfo['staridstr']
    tfa_stage1_results = {}

    for aperture in [1,2,3]:

        tfacmdstr = ("tfa -r {inputfile} --col-ref-id 1 "
                     "--col-ref-x 6 --col-ref-y 7 "
                     "-n {ntemplates} -T - -i /dev/null")

        # the +30 is to force selection of slightly more than the number
        # of strictly required templates
        tfacmd = tfacmdstr.format(
            inputfile=tfainfo[aperture]['info_file'],
            ntemplates=int(tfainfo[aperture]['target_tfa_nstars'])+30
        )

        print('aperture %s: starting TFA stage 1...' % aperture)

        if DEBUG:
            print(tfacmd)

        tfaproc = subprocess.Popen(shlex.split(tfacmd),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        tfa_stdout, tfa_stderr = tfaproc.communicate()

        # get results if succeeded, log outcome, and return path of outfile.
        # (note: this suppresses errors...)
        if tfaproc.returncode == 0 or tfa_stdout:
            tfaobjects = tfa_stdout.decode('utf-8').split('\n')
            tfaobjects = [x for x in tfaobjects
                          if x.startswith(staridstr) and x != '']
            print('aperture %s: TFA stage 1 completed, %s templates selected' %
                  (aperture, len(tfaobjects)))
            tfa_stage1_results[aperture] = tfaobjects
        else:
            print('aperture %s: TFA stage 1 failed, error was: %s' %
                  (aperture, tfa_stderr))
            tfa_stage1_results[aperture] = None

    return tfa_stage1_results



def run_tfa_singlelc(epdlc,
                     templatefiles,
                     outfile=None,
                     epdlc_jdcol=0,
                     epdlc_magcol=(22,23,24),
                     template_sigclip=5.0,
                     epdlc_sigclip=5.0):
    """
    This runs TFA for all apertures defined in epdlc_magcol for the input
    epdlc file, given an existing TFA template list in templatefile. If outfile
    is None, the output TFA LC will be in the same directory as epdlc but with
    an extension of .tfalc.
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

                print('%sZ: aperture %s TFA failed for %s! Error was: %s' %
                      (datetime.utcnow().isoformat(), magind+1, epdlc, tfa_stderr))

                tfalc_output.append(None)

        return tfalc_output

    else:

        print('ERR! %sZ: no TFA possible for %s! ndet < 2 x n(TFA templates)' %
              (datetime.utcnow().isoformat(), epdlc))
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
                     epdlc_magcol=(22,23,24),
                     template_sigclip=5.0,
                     epdlc_sigclip=5.0,
                     nworkers=16,
                     workerntasks=1000):
    """
    This runs TFA on the EPD lightcurves.
    """

    epdlcfiles = glob.glob(os.path.join(lcdir, epdlc_glob))

    tasks = [(x, templatefiles, {'epdlc_jdcol':epdlc_jdcol,
                                 'epdlc_magcol':epdlc_magcol,
                                 'template_sigclip':template_sigclip,
                                 'epdlc_sigclip':epdlc_sigclip})
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


#############################
## LC STATISTICS FUNCTIONS ##
#############################

# FIXME: break these out into their own module
# 1. RMS vs. MAG plot for EPD and TFA lightcurves
# 2. MAD vs. MAG plot for EPD and TFA lightcurves
# 3. ratios of RMS and MAD vs. MAG for CCD 6,7,8 to that of CCD 5
# 4. binned LC versions of these plots, using 10, 30, and 60 minute binning

# FIXME: get adding median magnitude measurement errors to this as well. This
# will allow for getting the predicted error relation and scintillation noise.
def get_magnitude_measurement_errs(photfile,
                                   frame,
                                   errcols=[],
                                   airmassheaderkey='X'):
    """
    This gets the median mag errs for the object and the airmass.

    Used to calculate the expected noise curve in an RMS plot.

    predicted total noise = sqrt((median mag err)**2 + (scintillation noise)**2)
    """



def get_lc_statistics(lcfile,
                      rmcols=[19,20,21],
                      epcols=[22,23,24],
                      tfcols=[25,26,27],
                      rfcols=None,
                      sigclip=4.0,
                      tfalcrequired=False,
                      epdlcrequired=True,
                      fitslcnottxt=False,
                      istessandmaskedges=False):
    """
    This calculates the following statistics for the magnitude columns in the
    given lcfile.

    mean, median, MAD, stdev.

    Args:
        lcfile: if using text file lightcurves, the lcfile is always the .epdlc
        file (which contains the rlc, and is used to derive the filenames of
        the tfalcs)

        fitslcnottxt (bool): a workaround for the above.

    rfcols are for the flux in aperture 1, 2, 3. used for ISM only
    """

    tf1lc_check = os.path.exists(lcfile.replace('.epdlc','.tfalc.TF1'))
    tf2lc_check = os.path.exists(lcfile.replace('.epdlc','.tfalc.TF2'))
    tf3lc_check = os.path.exists(lcfile.replace('.epdlc','.tfalc.TF3'))

    # otherwise, proceed with stat collection
    if fitslcnottxt:

        hdulist = pyfits.open(lcfile)

        # by default, raw fluxes are not populated. however, they are useful to
        # get a true estimate of the expected shot noise.
        rf1, rf2, rf3 = [], [], []

        if hdulist[0].header['DTR_EPD']:
            rm1 = hdulist[1].data['IRM1']
            rm2 = hdulist[1].data['IRM2']
            rm3 = hdulist[1].data['IRM3']
            if epdlcrequired:
                ep1 = hdulist[1].data['EP1']
                ep2 = hdulist[1].data['EP2']
                ep3 = hdulist[1].data['EP3']
            rf1 = hdulist[1].data['IFL1']
            rf2 = hdulist[1].data['IFL2']
            rf3 = hdulist[1].data['IFL3']
        elif not hdulist[0].header['DTR_EPD'] and not epdlcrequired:
            # a hack. some code has been written to rely on "EPD"
            # statistics. however, if you're skipping EPD, you want to
            # instead rely on IRM statistics. populating the EPD statistics
            # columns with IRM statistics is hacky, but fine.
            rm1 = hdulist[1].data['IRM1']
            rm2 = hdulist[1].data['IRM2']
            rm3 = hdulist[1].data['IRM3']
            ep1 = hdulist[1].data['IRM1']
            ep2 = hdulist[1].data['IRM2']
            ep3 = hdulist[1].data['IRM3']
            rf1 = hdulist[1].data['IFL1']
            rf2 = hdulist[1].data['IFL2']
            rf3 = hdulist[1].data['IFL3']
        else:
            print('expected DTR_EPD to be true in get_lc_statistics')
            raise AssertionError

        if hdulist[0].header['DTR_TFA']:
            tf1 = hdulist[1].data['TFA1']
            tf2 = hdulist[1].data['TFA2']
            tf3 = hdulist[1].data['TFA3']
        elif not hdulist[0].header['DTR_TFA'] and tfalcrequired:
            print(
                '{:s}Z: no TFA for {:s} and TFA is required, skipping...'.
                format(datetime.utcnow().isoformat(), lcfile)
            )
            return None
        else:
            tf1,tf2,tf3 = [], [], []

    else:

        # check if we need TFALCs to proceed
        if tfalcrequired and ((not tf1lc_check) or
                              (not tf2lc_check) or
                              (not tf3lc_check)):

            print('%sZ: no TFA mags available for %s and '
                  'TFALC is required, skipping...' %
                  (datetime.utcnow().isoformat(), lcfile))
            return None

        try:
            # get the reduced magnitude columns
            (rm1, rm2, rm3,
             ep1, ep2, ep3) = np.genfromtxt(lcfile,
                                            usecols=tuple(rmcols + epcols),
                                            unpack=True)
            tf1, tf2, tf3 = np.genfromtxt(
                lcfile.replace('.epdlc','.tfalc'),
                usecols=tfcols,
                unpack=True)
            if rfcols and len(rfcols) == 3:
                rf1, rf2, rf3 = np.genfromtxt(lcfile,usecols=tuple(rfcols),
                                              unpack=True)
            else:
                rf1, rf2, rf3 = [], [], []

        # if we don't have TF columns, cut down to RM and EP only
        except Exception as e:

            print('%sZ: no TFA mags available for %s!' %
                  (datetime.utcnow().isoformat(), lcfile))
            try:
                (rm1, rm2, rm3,
                 ep1, ep2, ep3) = np.genfromtxt(lcfile,
                                                usecols=tuple(rmcols + epcols),
                                                unpack=True)
                tf1, tf2, tf3 = [], [], []
                if rfcols and len(rfcols) == 3:
                    rf1, rf2, rf3 = np.genfromtxt(lcfile,usecols=tuple(rfcols),
                                                  unpack=True)
                else:
                    rf1, rf2, rf3 = [], [], []

            except Exception as e:

                print('%sZ: no EPD mags available for %s!' %
                      (datetime.utcnow().isoformat(), lcfile))

                rm1, rm2, rm3 = np.genfromtxt(lcfile,
                                              usecols=tuple(rmcols),
                                              unpack=True)

                ep1, ep2, ep3, tf1, tf2, tf3 = [], [], [], [], [], []

                if rfcols and len(rfcols) == 3:
                    rf1, rf2, rf3 = np.genfromtxt(lcfile,usecols=tuple(rfcols),
                                                  unpack=True)
                else:
                    rf1, rf2, rf3 = [], [], []

    #
    # optionally, if it's TESS data, mask the orbit edges since they are known
    # to be rampy
    #
    if istessandmaskedges:

        time = hdulist[1].data['TMID_BJD']

        from tessutils import mask_orbit_start_and_end

        orbitgap = 1
        expected_norbits = 2
        orbitpadding = 6/24
        raise_error = False

        _, rm1 = mask_orbit_start_and_end(time, rm1, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)
        _, rm2 = mask_orbit_start_and_end(time, rm2, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)
        _, rm3 = mask_orbit_start_and_end(time, rm3, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)

        _, ep1 = mask_orbit_start_and_end(time, ep1, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)
        _, ep2 = mask_orbit_start_and_end(time, ep2, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)
        _, ep3 = mask_orbit_start_and_end(time, ep3, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)

        _, rf1 = mask_orbit_start_and_end(time, rf1, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)
        _, rf2 = mask_orbit_start_and_end(time, rf2, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)
        _, rf3 = mask_orbit_start_and_end(time, rf3, orbitgap=orbitgap,
                                          expected_norbits=expected_norbits,
                                          orbitpadding=orbitpadding,
                                          raise_error=raise_error)

        if len(tf1) > 0:
            _, tf1 = mask_orbit_start_and_end(time, tf1, orbitgap=orbitgap,
                                              expected_norbits=expected_norbits,
                                              orbitpadding=orbitpadding,
                                              raise_error=raise_error)
        if len(tf2) > 0:
            _, tf2 = mask_orbit_start_and_end(time, tf2, orbitgap=orbitgap,
                                              expected_norbits=expected_norbits,
                                              orbitpadding=orbitpadding,
                                              raise_error=raise_error)
        if len(tf3) > 0:
            _, tf3 = mask_orbit_start_and_end(time, tf3, orbitgap=orbitgap,
                                              expected_norbits=expected_norbits,
                                              orbitpadding=orbitpadding,
                                              raise_error=raise_error)


    ##################################
    # get statistics for each column #
    ##################################

    # fluxes
    # RF1
    if len(rf1) > 4:

        finiteind = np.isfinite(rf1)
        rf1 = rf1[finiteind]
        median_rf1 = np.median(rf1)
        mad_rf1 = np.median(np.fabs(rf1 - median_rf1))
        mean_rf1 = np.mean(rf1)
        stdev_rf1 = np.std(rf1)
        ndet_rf1 = len(rf1)

        if sigclip:
            sigclip_rf1, lo, hi = stats_sigmaclip(rf1,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_rf1 = np.median(sigclip_rf1)
            mad_sigclip_rf1 = np.median(np.fabs(sigclip_rf1 -
                                                median_sigclip_rf1))
            mean_sigclip_rf1 = np.mean(sigclip_rf1)
            stdev_sigclip_rf1 = np.std(sigclip_rf1)
            ndet_sigclip_rf1 = len(sigclip_rf1)

        else:
            median_sigclip_rf1 = np.nan
            mad_sigclip_rf1 = np.nan
            mean_sigclip_rf1 = np.nan
            stdev_sigclip_rf1 = np.nan
            ndet_sigclip_rf1 = np.nan

    else:

        median_rf1, mad_rf1, mean_rf1, stdev_rf1 = np.nan, np.nan, np.nan, np.nan
        ndet_rf1 = np.nan
        median_sigclip_rf1, mad_sigclip_rf1 = np.nan, np.nan
        mean_sigclip_rf1, stdev_sigclip_rf1 = np.nan, np.nan
        ndet_sigclip_rf1 = np.nan

    # RF2
    if len(rf2) > 4:

        finiteind = np.isfinite(rf2)
        rf2 = rf2[finiteind]
        median_rf2 = np.median(rf2)
        mad_rf2 = np.median(np.fabs(rf2 - median_rf2))
        mean_rf2 = np.mean(rf2)
        stdev_rf2 = np.std(rf2)
        ndet_rf2 = len(rf2)

        if sigclip:
            sigclip_rf2, lo, hi = stats_sigmaclip(rf2,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_rf2 = np.median(sigclip_rf2)
            mad_sigclip_rf2 = np.median(np.fabs(sigclip_rf2 -
                                                median_sigclip_rf2))
            mean_sigclip_rf2 = np.mean(sigclip_rf2)
            stdev_sigclip_rf2 = np.std(sigclip_rf2)
            ndet_sigclip_rf2 = len(sigclip_rf2)

        else:
            median_sigclip_rf2 = np.nan
            mad_sigclip_rf2 = np.nan
            mean_sigclip_rf2 = np.nan
            stdev_sigclip_rf2 = np.nan
            ndet_sigclip_rf2 = np.nan

    else:

        median_rf2, mad_rf2, mean_rf2, stdev_rf2 = np.nan, np.nan, np.nan, np.nan
        ndet_rf2 = np.nan
        median_sigclip_rf2, mad_sigclip_rf2 = np.nan, np.nan
        mean_sigclip_rf2, stdev_sigclip_rf2 = np.nan, np.nan
        ndet_sigclip_rf2 = np.nan

    # RF3
    if len(rf3) > 4:

        finiteind = np.isfinite(rf3)
        rf3 = rf3[finiteind]
        median_rf3 = np.median(rf3)
        mad_rf3 = np.median(np.fabs(rf3 - median_rf3))
        mean_rf3 = np.mean(rf3)
        stdev_rf3 = np.std(rf3)
        ndet_rf3 = len(rf3)

        if sigclip:
            sigclip_rf3, lo, hi = stats_sigmaclip(rf3,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_rf3 = np.median(sigclip_rf3)
            mad_sigclip_rf3 = np.median(np.fabs(sigclip_rf3 -
                                                median_sigclip_rf3))
            mean_sigclip_rf3 = np.mean(sigclip_rf3)
            stdev_sigclip_rf3 = np.std(sigclip_rf3)
            ndet_sigclip_rf3 = len(sigclip_rf3)

        else:
            median_sigclip_rf3 = np.nan
            mad_sigclip_rf3 = np.nan
            mean_sigclip_rf3 = np.nan
            stdev_sigclip_rf3 = np.nan
            ndet_sigclip_rf3 = np.nan

    else:

        median_rf3, mad_rf3, mean_rf3, stdev_rf3 = (np.nan, np.nan,
                                                    np.nan, np.nan)
        ndet_rf3 = np.nan
        median_sigclip_rf3, mad_sigclip_rf3 = np.nan, np.nan
        mean_sigclip_rf3, stdev_sigclip_rf3 = np.nan, np.nan
        ndet_sigclip_rf3 = np.nan


    # mags
    # RM1
    if len(rm1) > 4:

        finiteind = np.isfinite(rm1)
        rm1 = rm1[finiteind]
        median_rm1 = np.median(rm1)
        mad_rm1 = np.median(np.fabs(rm1 - median_rm1))
        mean_rm1 = np.mean(rm1)
        stdev_rm1 = np.std(rm1)
        ndet_rm1 = len(rm1)

        if sigclip:
            sigclip_rm1, lo, hi = stats_sigmaclip(rm1,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_rm1 = np.median(sigclip_rm1)
            mad_sigclip_rm1 = np.median(np.fabs(sigclip_rm1 -
                                                median_sigclip_rm1))
            mean_sigclip_rm1 = np.mean(sigclip_rm1)
            stdev_sigclip_rm1 = np.std(sigclip_rm1)
            ndet_sigclip_rm1 = len(sigclip_rm1)

        else:
            median_sigclip_rm1 = np.nan
            mad_sigclip_rm1 = np.nan
            mean_sigclip_rm1 = np.nan
            stdev_sigclip_rm1 = np.nan
            ndet_sigclip_rm1 = np.nan

    else:

        median_rm1, mad_rm1, mean_rm1, stdev_rm1 = np.nan, np.nan, np.nan, np.nan
        ndet_rm1 = np.nan
        median_sigclip_rm1, mad_sigclip_rm1 = np.nan, np.nan
        mean_sigclip_rm1, stdev_sigclip_rm1 = np.nan, np.nan
        ndet_sigclip_rm1 = np.nan

    # RM2
    if len(rm2) > 4:

        finiteind = np.isfinite(rm2)
        rm2 = rm2[finiteind]
        median_rm2 = np.median(rm2)
        mad_rm2 = np.median(np.fabs(rm2 - median_rm2))
        mean_rm2 = np.mean(rm2)
        stdev_rm2 = np.std(rm2)
        ndet_rm2 = len(rm2)

        if sigclip:
            sigclip_rm2, lo, hi = stats_sigmaclip(rm2,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_rm2 = np.median(sigclip_rm2)
            mad_sigclip_rm2 = np.median(np.fabs(sigclip_rm2 -
                                                median_sigclip_rm2))
            mean_sigclip_rm2 = np.mean(sigclip_rm2)
            stdev_sigclip_rm2 = np.std(sigclip_rm2)
            ndet_sigclip_rm2 = len(sigclip_rm2)

        else:
            median_sigclip_rm2 = np.nan
            mad_sigclip_rm2 = np.nan
            mean_sigclip_rm2 = np.nan
            stdev_sigclip_rm2 = np.nan
            ndet_sigclip_rm2 = np.nan

    else:

        median_rm2, mad_rm2, mean_rm2, stdev_rm2 = np.nan, np.nan, np.nan, np.nan
        ndet_rm2 = np.nan
        median_sigclip_rm2, mad_sigclip_rm2 = np.nan, np.nan
        mean_sigclip_rm2, stdev_sigclip_rm2 = np.nan, np.nan
        ndet_sigclip_rm2 = np.nan

    # RM3
    if len(rm3) > 4:

        finiteind = np.isfinite(rm3)
        rm3 = rm3[finiteind]
        median_rm3 = np.median(rm3)
        mad_rm3 = np.median(np.fabs(rm3 - median_rm3))
        mean_rm3 = np.mean(rm3)
        stdev_rm3 = np.std(rm3)
        ndet_rm3 = len(rm3)

        if sigclip:
            sigclip_rm3, lo, hi = stats_sigmaclip(rm3,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_rm3 = np.median(sigclip_rm3)
            mad_sigclip_rm3 = np.median(np.fabs(sigclip_rm3 -
                                                median_sigclip_rm3))
            mean_sigclip_rm3 = np.mean(sigclip_rm3)
            stdev_sigclip_rm3 = np.std(sigclip_rm3)
            ndet_sigclip_rm3 = len(sigclip_rm3)

        else:
            median_sigclip_rm3 = np.nan
            mad_sigclip_rm3 = np.nan
            mean_sigclip_rm3 = np.nan
            stdev_sigclip_rm3 = np.nan
            ndet_sigclip_rm3 = np.nan

    else:

        median_rm3, mad_rm3, mean_rm3, stdev_rm3 = (np.nan, np.nan,
                                                    np.nan, np.nan)
        ndet_rm3 = np.nan
        median_sigclip_rm3, mad_sigclip_rm3 = np.nan, np.nan
        mean_sigclip_rm3, stdev_sigclip_rm3 = np.nan, np.nan
        ndet_sigclip_rm3 = np.nan

    # EP1
    if len(ep1) > 4:

        finiteind = np.isfinite(ep1)
        ep1 = ep1[finiteind]
        median_ep1 = np.median(ep1)
        mad_ep1 = np.median(np.fabs(ep1 - median_ep1))
        mean_ep1 = np.mean(ep1)
        stdev_ep1 = np.std(ep1)
        ndet_ep1 = len(ep1)

        if sigclip:
            sigclip_ep1, lo, hi = stats_sigmaclip(ep1,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_ep1 = np.median(sigclip_ep1)
            mad_sigclip_ep1 = np.median(np.fabs(sigclip_ep1 -
                                                median_sigclip_ep1))
            mean_sigclip_ep1 = np.mean(sigclip_ep1)
            stdev_sigclip_ep1 = np.std(sigclip_ep1)
            ndet_sigclip_ep1 = len(sigclip_ep1)

        else:
            median_sigclip_ep1 = np.nan
            mad_sigclip_ep1 = np.nan
            mean_sigclip_ep1 = np.nan
            stdev_sigclip_ep1 = np.nan
            ndet_sigclip_ep1 = np.nan

    else:

        median_ep1, mad_ep1, mean_ep1, stdev_ep1 = np.nan, np.nan, np.nan, np.nan
        ndet_ep1 = np.nan
        median_sigclip_ep1, mad_sigclip_ep1 = np.nan, np.nan
        mean_sigclip_ep1, stdev_sigclip_ep1 = np.nan, np.nan
        ndet_sigclip_ep1 = np.nan

    # EP2
    if len(ep2) > 4:

        finiteind = np.isfinite(ep2)
        ep2 = ep2[finiteind]
        median_ep2 = np.median(ep2)
        mad_ep2 = np.median(np.fabs(ep2 - median_ep2))
        mean_ep2 = np.mean(ep2)
        stdev_ep2 = np.std(ep2)
        ndet_ep2 = len(ep2)

        if sigclip:
            sigclip_ep2, lo, hi = stats_sigmaclip(ep2,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_ep2 = np.median(sigclip_ep2)
            mad_sigclip_ep2 = np.median(np.fabs(sigclip_ep2 -
                                                median_sigclip_ep2))
            mean_sigclip_ep2 = np.mean(sigclip_ep2)
            stdev_sigclip_ep2 = np.std(sigclip_ep2)
            ndet_sigclip_ep2 = len(sigclip_ep2)

        else:
            median_sigclip_ep2 = np.nan
            mad_sigclip_ep2 = np.nan
            mean_sigclip_ep2 = np.nan
            stdev_sigclip_ep2 = np.nan
            ndet_sigclip_ep2 = np.nan

    else:

        median_ep2, mad_ep2, mean_ep2, stdev_ep2 = np.nan, np.nan, np.nan, np.nan
        ndet_ep2 = np.nan
        median_sigclip_ep2, mad_sigclip_ep2 = np.nan, np.nan
        mean_sigclip_ep2, stdev_sigclip_ep2 = np.nan, np.nan
        ndet_sigclip_ep2 = np.nan

    # EP3
    if len(ep3) > 4:

        finiteind = np.isfinite(ep3)
        ep3 = ep3[finiteind]
        median_ep3 = np.median(ep3)
        mad_ep3 = np.median(np.fabs(ep3 - median_ep3))
        mean_ep3 = np.mean(ep3)
        stdev_ep3 = np.std(ep3)
        ndet_ep3 = len(ep3)

        if sigclip:
            sigclip_ep3, lo, hi = stats_sigmaclip(ep3,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_ep3 = np.median(sigclip_ep3)
            mad_sigclip_ep3 = np.median(np.fabs(sigclip_ep3 -
                                                median_sigclip_ep3))
            mean_sigclip_ep3 = np.mean(sigclip_ep3)
            stdev_sigclip_ep3 = np.std(sigclip_ep3)
            ndet_sigclip_ep3 = len(sigclip_ep3)

        else:
            median_sigclip_ep3 = np.nan
            mad_sigclip_ep3 = np.nan
            mean_sigclip_ep3 = np.nan
            stdev_sigclip_ep3 = np.nan
            ndet_sigclip_ep3 = np.nan

    else:

        median_ep3, mad_ep3, mean_ep3, stdev_ep3 = np.nan, np.nan, np.nan, np.nan
        ndet_ep3 = np.nan
        median_sigclip_ep3, mad_sigclip_ep3 = np.nan, np.nan
        mean_sigclip_ep3, stdev_sigclip_ep3 = np.nan, np.nan
        ndet_sigclip_ep3 = np.nan

    # TF1
    if len(tf1) > 4:

        finiteind = np.isfinite(tf1)
        tf1 = tf1[finiteind]
        median_tf1 = np.median(tf1)
        mad_tf1 = np.median(np.fabs(tf1 - median_tf1))
        mean_tf1 = np.mean(tf1)
        stdev_tf1 = np.std(tf1)
        ndet_tf1 = len(tf1)

        if sigclip:
            sigclip_tf1, lo, hi = stats_sigmaclip(tf1,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_tf1 = np.median(sigclip_tf1)
            mad_sigclip_tf1 = np.median(np.fabs(sigclip_tf1 -
                                                median_sigclip_tf1))
            mean_sigclip_tf1 = np.mean(sigclip_tf1)
            stdev_sigclip_tf1 = np.std(sigclip_tf1)
            ndet_sigclip_tf1 = len(sigclip_tf1)

        else:
            median_sigclip_tf1 = np.nan
            mad_sigclip_tf1 = np.nan
            mean_sigclip_tf1 = np.nan
            stdev_sigclip_tf1 = np.nan
            ndet_sigclip_tf1 = np.nan

    else:

        median_tf1, mad_tf1, mean_tf1, stdev_tf1 = np.nan, np.nan, np.nan, np.nan
        ndet_tf1 = np.nan
        median_sigclip_tf1, mad_sigclip_tf1 = np.nan, np.nan
        mean_sigclip_tf1, stdev_sigclip_tf1 = np.nan, np.nan
        ndet_sigclip_tf1 = np.nan

    # TF2
    if len(tf2) > 4:

        finiteind = np.isfinite(tf2)
        tf2 = tf2[finiteind]
        median_tf2 = np.median(tf2)
        mad_tf2 = np.median(np.fabs(tf2 - median_tf2))
        mean_tf2 = np.mean(tf2)
        stdev_tf2 = np.std(tf2)
        ndet_tf2 = len(tf2)

        if sigclip:
            sigclip_tf2, lo, hi = stats_sigmaclip(tf2,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_tf2 = np.median(sigclip_tf2)
            mad_sigclip_tf2 = np.median(np.fabs(sigclip_tf2 -
                                                median_sigclip_tf2))
            mean_sigclip_tf2 = np.mean(sigclip_tf2)
            stdev_sigclip_tf2 = np.std(sigclip_tf2)
            ndet_sigclip_tf2 = len(sigclip_tf2)

        else:
            median_sigclip_tf2 = np.nan
            mad_sigclip_tf2 = np.nan
            mean_sigclip_tf2 = np.nan
            stdev_sigclip_tf2 = np.nan
            ndet_sigclip_tf2 = np.nan

    else:

        median_tf2, mad_tf2, mean_tf2, stdev_tf2 = np.nan, np.nan, np.nan, np.nan
        ndet_tf2 = np.nan
        median_sigclip_tf2, mad_sigclip_tf2 = np.nan, np.nan
        mean_sigclip_tf2, stdev_sigclip_tf2 = np.nan, np.nan
        ndet_sigclip_tf2 = np.nan

    # TF3
    if len(tf3) > 4:

        finiteind = np.isfinite(tf3)
        tf3 = tf3[finiteind]
        median_tf3 = np.median(tf3)
        mad_tf3 = np.median(np.fabs(tf3 - median_tf3))
        mean_tf3 = np.mean(tf3)
        stdev_tf3 = np.std(tf3)
        ndet_tf3 = len(tf3)

        if sigclip:
            sigclip_tf3, lo, hi = stats_sigmaclip(tf3,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_tf3 = np.median(sigclip_tf3)
            mad_sigclip_tf3 = np.median(np.fabs(sigclip_tf3 -
                                                median_sigclip_tf3))
            mean_sigclip_tf3 = np.mean(sigclip_tf3)
            stdev_sigclip_tf3 = np.std(sigclip_tf3)
            ndet_sigclip_tf3 = len(sigclip_tf3)

        else:
            median_sigclip_tf3 = np.nan
            mad_sigclip_tf3 = np.nan
            mean_sigclip_tf3 = np.nan
            stdev_sigclip_tf3 = np.nan
            ndet_sigclip_tf3 = np.nan

    else:

        median_tf3, mad_tf3, mean_tf3, stdev_tf3 = np.nan, np.nan, np.nan, np.nan
        ndet_tf3 = np.nan
        median_sigclip_tf3, mad_sigclip_tf3 = np.nan, np.nan
        mean_sigclip_tf3, stdev_sigclip_tf3 = np.nan, np.nan
        ndet_sigclip_tf3 = np.nan


    ## COLLECT STATS

    print('%sZ: done with statistics for %s' %
          (datetime.utcnow().isoformat(), lcfile))

    lcobj = os.path.splitext(os.path.basename(lcfile))[0]
    if '_llc' in lcobj:
        # TESS filenaming format: {gaiaid}_llc.fits -- so if we want to
        # retrieve the Gaia-ID, must omit "_llc" here.
        lcobj = lcobj.replace('_llc','')
    if 'hlsp' in lcobj:
        # typical format:
        # hlsp_cdips_tess_ffi_gaiatwo0002917938284837069312-0006_tess_v01_llc.fits
        from parse import search
        res = search('hlsp_cdips_tess_ffi_gaiatwo{}-{}_tess{}', lcobj)
        lcobj = res[0].lstrip('0')

    return {'lcfile':lcfile,
            'lcobj':lcobj,

            # reduced mags aperture 1
            'median_rf1':median_rf1,
            'mad_rf1':mad_rf1,
            'mean_rf1':mean_rf1,
            'stdev_rf1':stdev_rf1,
            'ndet_rf1':ndet_rf1,
            'median_sigclip_rf1':median_sigclip_rf1,
            'mad_sigclip_rf1':mad_sigclip_rf1,
            'mean_sigclip_rf1':mean_sigclip_rf1,
            'stdev_sigclip_rf1':stdev_sigclip_rf1,
            'ndet_sigclip_rf1':ndet_sigclip_rf1,
            # reduced mags aperture 2
            'median_rf2':median_rf2,
            'mad_rf2':mad_rf2,
            'mean_rf2':mean_rf2,
            'stdev_rf2':stdev_rf2,
            'ndet_rf2':ndet_rf2,
            'median_sigclip_rf2':median_sigclip_rf2,
            'mad_sigclip_rf2':mad_sigclip_rf2,
            'mean_sigclip_rf2':mean_sigclip_rf2,
            'stdev_sigclip_rf2':stdev_sigclip_rf2,
            'ndet_sigclip_rf2':ndet_sigclip_rf2,
            # reduced mags aperture 3
            'median_rf3':median_rf3,
            'mad_rf3':mad_rf3,
            'mean_rf3':mean_rf3,
            'stdev_rf3':stdev_rf3,
            'ndet_rf3':ndet_rf3,
            'median_sigclip_rf3':median_sigclip_rf3,
            'mad_sigclip_rf3':mad_sigclip_rf3,
            'mean_sigclip_rf3':mean_sigclip_rf3,
            'stdev_sigclip_rf3':stdev_sigclip_rf3,
            'ndet_sigclip_rf3':ndet_sigclip_rf3,

            # reduced mags aperture 1
            'median_rm1':median_rm1,
            'mad_rm1':mad_rm1,
            'mean_rm1':mean_rm1,
            'stdev_rm1':stdev_rm1,
            'ndet_rm1':ndet_rm1,
            'median_sigclip_rm1':median_sigclip_rm1,
            'mad_sigclip_rm1':mad_sigclip_rm1,
            'mean_sigclip_rm1':mean_sigclip_rm1,
            'stdev_sigclip_rm1':stdev_sigclip_rm1,
            'ndet_sigclip_rm1':ndet_sigclip_rm1,
            # reduced mags aperture 2
            'median_rm2':median_rm2,
            'mad_rm2':mad_rm2,
            'mean_rm2':mean_rm2,
            'stdev_rm2':stdev_rm2,
            'ndet_rm2':ndet_rm2,
            'median_sigclip_rm2':median_sigclip_rm2,
            'mad_sigclip_rm2':mad_sigclip_rm2,
            'mean_sigclip_rm2':mean_sigclip_rm2,
            'stdev_sigclip_rm2':stdev_sigclip_rm2,
            'ndet_sigclip_rm2':ndet_sigclip_rm2,
            # reduced mags aperture 3
            'median_rm3':median_rm3,
            'mad_rm3':mad_rm3,
            'mean_rm3':mean_rm3,
            'stdev_rm3':stdev_rm3,
            'ndet_rm3':ndet_rm3,
            'median_sigclip_rm3':median_sigclip_rm3,
            'mad_sigclip_rm3':mad_sigclip_rm3,
            'mean_sigclip_rm3':mean_sigclip_rm3,
            'stdev_sigclip_rm3':stdev_sigclip_rm3,
            'ndet_sigclip_rm3':ndet_sigclip_rm3,

            # EPD mags aperture 1
            'median_ep1':median_ep1,
            'mad_ep1':mad_ep1,
            'mean_ep1':mean_ep1,
            'stdev_ep1':stdev_ep1,
            'ndet_ep1':ndet_ep1,
            'median_sigclip_ep1':median_sigclip_ep1,
            'mad_sigclip_ep1':mad_sigclip_ep1,
            'mean_sigclip_ep1':mean_sigclip_ep1,
            'stdev_sigclip_ep1':stdev_sigclip_ep1,
            'ndet_sigclip_ep1':ndet_sigclip_ep1,
            # EPD mags aperture 2
            'median_ep2':median_ep2,
            'mad_ep2':mad_ep2,
            'mean_ep2':mean_ep2,
            'stdev_ep2':stdev_ep2,
            'ndet_ep2':ndet_ep2,
            'median_sigclip_ep2':median_sigclip_ep2,
            'mad_sigclip_ep2':mad_sigclip_ep2,
            'mean_sigclip_ep2':mean_sigclip_ep2,
            'stdev_sigclip_ep2':stdev_sigclip_ep2,
            'ndet_sigclip_ep2':ndet_sigclip_ep2,
            # EPD mags aperture 3
            'median_ep3':median_ep3,
            'mad_ep3':mad_ep3,
            'mean_ep3':mean_ep3,
            'stdev_ep3':stdev_ep3,
            'ndet_ep3':ndet_ep3,
            'median_sigclip_ep3':median_sigclip_ep3,
            'mad_sigclip_ep3':mad_sigclip_ep3,
            'mean_sigclip_ep3':mean_sigclip_ep3,
            'stdev_sigclip_ep3':stdev_sigclip_ep3,
            'ndet_sigclip_ep3':ndet_sigclip_ep3,

            # TFA mags aperture 1
            'median_tf1':median_tf1,
            'mad_tf1':mad_tf1,
            'mean_tf1':mean_tf1,
            'stdev_tf1':stdev_tf1,
            'ndet_tf1':ndet_tf1,
            'median_sigclip_tf1':median_sigclip_tf1,
            'mad_sigclip_tf1':mad_sigclip_tf1,
            'mean_sigclip_tf1':mean_sigclip_tf1,
            'stdev_sigclip_tf1':stdev_sigclip_tf1,
            'ndet_sigclip_tf1':ndet_sigclip_tf1,
            # TFA mags aperture 2
            'median_tf2':median_tf2,
            'mad_tf2':mad_tf2,
            'mean_tf2':mean_tf2,
            'stdev_tf2':stdev_tf2,
            'ndet_tf2':ndet_tf2,
            'median_sigclip_tf2':median_sigclip_tf2,
            'mad_sigclip_tf2':mad_sigclip_tf2,
            'mean_sigclip_tf2':mean_sigclip_tf2,
            'stdev_sigclip_tf2':stdev_sigclip_tf2,
            'ndet_sigclip_tf2':ndet_sigclip_tf2,
            # TFA mags aperture 3
            'median_tf3':median_tf3,
            'mad_tf3':mad_tf3,
            'mean_tf3':mean_tf3,
            'stdev_tf3':stdev_tf3,
            'ndet_tf3':ndet_tf3,
            'median_sigclip_tf3':median_sigclip_tf3,
            'mad_sigclip_tf3':mad_sigclip_tf3,
            'mean_sigclip_tf3':mean_sigclip_tf3,
            'stdev_sigclip_tf3':stdev_sigclip_tf3,
            'ndet_sigclip_tf3':ndet_sigclip_tf3}



def lc_statistics_worker(task):
    """
    This is a worker that runs the function above in a parallel worker pool.
    """

    try:
        return get_lc_statistics(task[0], **task[1])
    except Exception as e:
        print('LC STATS SOMETHING WENT WRONG! task was {}, exception was {}'.
              format( repr(task), repr(e))
        )
        return None



def parallel_lc_statistics(lcdir,
                           lcglob,
                           fovcatalog,
                           fovcathasgaiaids=False,
                           tfalcrequired=False,
                           fitslcnottxt=False,
                           fovcatcols=(0,9), # objectid, magcol to use
                           fovcatmaglabel='r',
                           outfile=None,
                           nworkers=16,
                           workerntasks=500,
                           rmcols=[19,20,21],
                           epcols=[22,23,24],
                           tfcols=[25,26,27],
                           rfcols=None,
                           correctioncoeffs=None,
                           sigclip=4.0,
                           epdlcrequired=True,
                           istessandmaskedges=False):
    """
    This calculates statistics on all lc files in lcdir.

    Args:
        lcdir (str): directory containing lightcurves

        lcglob (str): glob to epd lcs, inside lcdir. E.g., '*.epdlc'. These
        contain the rlc, and are used to derive the filenames of the tfalcs.

        fovcatalog (str): path to the REFORMED fov catalog, which gets the
        catalog magnitude corresponding to canonical magnitude for any star.

        fovcathasgaiaids (bool): if the reformed FOV catalog has Gaia ids, set
        this to be true. The default is to assume HAT-IDs, which have different
        string lengths & and are read differently.

        fitslcnottxt (bool): if True, I/O will attempt to read
        INSTRUMENTAL/EPD/TFA magnitudes from a FITS-format lightcurve, not a
        text-file lightcurve. By default, false.

        epdlcrequired (bool): a variety of aperturephot.py tools assume that if
        you are creating statistics, you have run EPD. This isn't necessarily
        true (you may wish to get statistics on the instrumental raw
        magnitudes). If you set this to False, the statistics file will,
        hackily, populate the "EPD statistics" with IRM values.

    Output:

        Puts the results in text file outfile.
        outfile contains the following columns:

            object, ndet,
            median RM[1-3], MAD RM[1-3], mean RM[1-3], stdev RM[1-3],
            median EP[1-3], MAD EP[1-3], mean EP[1-3], stdev EP[1-3],
            median TF[1-3], MAD TF[1-3], mean TF[1-3], stdev TF[1-3]

        if a value is missing, it will be np.nan.

    Notes:
        For ISM, consider using correctioncoeffs as well. These are c1, c2
        resulting from a fit to the catalogmag-flux relation using the
        expression:

        catrmag = -2.5 * log10(flux/c1) + c2

        where the fit is done in the bright limit (8.0 < r < 12.0). this
        corrects for too-faint catalog mags because of crowding and blending.

        correctioncoeffs is like:
            [[ap1_c1,ap1_c2],[ap2_c1,ap2_c2],[ap3_c1,ap3_c2]]
    """

    lcfiles = glob.glob(os.path.join(lcdir, lcglob))

    tasks = [[x, {'rmcols':rmcols,
                  'epcols':epcols,
                  'tfcols':tfcols,
                  'rfcols':rfcols,
                  'sigclip':sigclip,
                  'tfalcrequired':tfalcrequired,
                  'fitslcnottxt':fitslcnottxt,
                  'epdlcrequired':epdlcrequired,
                  'istessandmaskedges':istessandmaskedges}] for x in lcfiles]

    pool = mp.Pool(nworkers,maxtasksperchild=workerntasks)
    results = pool.map(lc_statistics_worker, tasks)
    pool.close()
    pool.join()

    print('%sZ: done. %s lightcurves processed.' %
          (datetime.utcnow().isoformat(), len(lcfiles)))

    if not outfile:
        outfile = os.path.join(lcdir, 'lightcurve-statistics.txt')

    outf = open(outfile,'wb')

    outlineformat = (
        '%s %.3f  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.3f %.3f %.3f\n'
        )

    outheader = '# total objects: %s, sigmaclip used: %s\n' % (
        len(lcfiles), sigclip)

    outf.write(outheader.encode('utf-8'))

    outcolumnkey = (
        '# columns are:\n'
        '# 0,1: object, catalog mag %s\n'
        '# 2,3,4,5,6: median RM1, MAD RM1, mean RM1, stdev RM1, ndet RM1\n'
        '# 7,8,9,10,11: sigma-clipped median RM1, MAD RM1, mean RM1, '
        'stdev RM1, ndet RM1\n'
        '# 12,13,14,15,16: median RM2, MAD RM2, mean RM2, stdev RM2, ndet RM2\n'
        '# 17,18,19,20,21: sigma-clipped median RM2, MAD RM2, mean RM2, '
        'stdev RM2, ndet RM2\n'
        '# 22,23,24,25,26: median RM3, MAD RM3, mean RM3, stdev RM3, ndet RM3\n'
        '# 27,28,29,30,31: sigma-clipped median RM3, MAD RM3, mean RM3, '
        'stdev RM3, ndet RM3\n'
        '# 32,33,34,35,36: median EP1, MAD EP1, mean EP1, stdev EP1, ndet EP1\n'
        '# 37,38,39,40,41: sigma-clipped median EP1, MAD EP1, mean EP1, '
        'stdev EP1, ndet EP1\n'
        '# 42,43,44,45,46: median EP2, MAD EP2, mean EP2, stdev EP2, ndet EP2\n'
        '# 47,48,49,50,51: sigma-clipped median EP2, MAD EP2, mean EP2, '
        'stdev EP2, ndet EP2\n'
        '# 52,53,54,55,56: median EP3, MAD EP3, mean EP3, stdev EP3, ndet EP3\n'
        '# 57,58,59,60,61: sigma-clipped median EP3, MAD EP3, mean EP3, '
        'stdev EP3, ndet EP3\n'
        '# 62,63,64,65,66: median TF1, MAD TF1, mean TF1, stdev TF1, ndet TF1\n'
        '# 67,68,69,70,71: sigma-clipped median TF1, MAD TF1, mean TF1, '
        'stdev TF1, ndet TF1\n'
        '# 72,73,74,75,76: median TF2, MAD TF2, mean TF2, stdev TF2, ndet TF2\n'
        '# 77,78,79,80,81: sigma-clipped median TF2, MAD TF2, mean TF2, '
        'stdev TF2, ndet TF2\n'
        '# 82,83,84,85,86: median TF3, MAD TF3, mean TF3, stdev TF3, ndet TF3\n'
        '# 87,88,89,90,91: sigma-clipped median TF3, MAD TF3, mean TF3, '
        'stdev TF3, ndet TF3\n'
        '# 92,93,94,95,96: median RF1, MAD RF1, mean RF1, stdev RF1, ndet RF1\n'
        '# 97,98,99,100,101: sigma-clipped median RF1, MAD RF1, mean RF1, '
        'stdev RF1, ndet RF1\n'
        '# 102,103,104,105,106: median RF2, MAD RF2, mean RF2, stdev RF2, ndet '
        'RF2\n'
        '# 107,108,109,110,111: sigma-clipped median RF2, MAD RF2, mean RF2, '
        'stdev RF2, ndet RF2\n'
        '# 112,113,114,115,116: median RF3, MAD RF3, mean RF3, stdev RF3, '
        'ndet RF3\n'
        '# 117,118,119,120,121: sigma-clipped median RF3, MAD RF3, mean RF3, '
        'stdev RF3, ndet RF3\n'
        '# 122, 123, 124: corrected cat mag AP1, corrected cat mag AP1, '
        'corrected cat mag AP3\n'
        ) % fovcatmaglabel
    outf.write(outcolumnkey.encode('utf-8'))

    # open the fovcatalog and read in the column magnitudes and hatids
    if not fovcathasgaiaids:
        # assume HAT-IDs, HAT-123-4567890, 17 character strings
        fovcat = np.genfromtxt(fovcatalog,
                               usecols=fovcatcols,
                               dtype='U17,f8',
                               names=['objid','mag'])
    else:
        # assume GAIA-IDs. From gaia2read, with "GAIA" id option, this is just
        # 19 character integers.
        fovcat = np.genfromtxt(fovcatalog,
                               usecols=fovcatcols,
                               dtype='U19,f8',
                               names=['objid','mag'])

    # Using a dictionary leads to ~ 300x speedup
    fovdict = dict(fovcat)

    for stat in results:
        if stat is not None:
            # find the catalog mag for this object
            if stat['lcobj'] in fovdict:
                catmag = fovdict[stat['lcobj']]
            else:
                print('no catalog mag for %s, using median TF3 mag' %
                      stat['lcobj'])
                catmag = stat['median_tf3']
            if pd.isnull(catmag):
                print('no median_tf3 mag for %s, using median EP3 mag' %
                      stat['lcobj'])
                catmag = stat['median_ep3']
            if pd.isnull(catmag):
                print('WRN! no catalog, TF3 or EP3 mag for {:s}. using nan'.
                      format(stat['lcobj']))
                catmag = np.array([np.nan])

            # calculate the corrected mags if present
            if (correctioncoeffs and len(correctioncoeffs) == 3 and
                rfcols and len(rfcols) == 3):

                ap1_c1, ap1_c2 = correctioncoeffs[0]
                ap2_c1, ap2_c2 = correctioncoeffs[1]
                ap3_c1, ap3_c2 = correctioncoeffs[2]

                corrmag_ap1 = -2.5*np.log10(stat['median_rf1']/ap1_c1) + ap1_c2
                corrmag_ap2 = -2.5*np.log10(stat['median_rf2']/ap2_c1) + ap2_c2
                corrmag_ap3 = -2.5*np.log10(stat['median_rf3']/ap3_c1) + ap3_c2

            else:

                corrmag_ap1 = catmag
                corrmag_ap2 = catmag
                corrmag_ap3 = catmag


            outline = outlineformat % (
                stat['lcobj'],
                np.asscalar(catmag),

                stat['median_rm1'],
                stat['mad_rm1'],
                stat['mean_rm1'],
                stat['stdev_rm1'],
                stat['ndet_rm1'],
                stat['median_sigclip_rm1'],
                stat['mad_sigclip_rm1'],
                stat['mean_sigclip_rm1'],
                stat['stdev_sigclip_rm1'],
                stat['ndet_sigclip_rm1'],

                stat['median_rm2'],
                stat['mad_rm2'],
                stat['mean_rm2'],
                stat['stdev_rm2'],
                stat['ndet_rm2'],
                stat['median_sigclip_rm2'],
                stat['mad_sigclip_rm2'],
                stat['mean_sigclip_rm2'],
                stat['stdev_sigclip_rm2'],
                stat['ndet_sigclip_rm2'],

                stat['median_rm3'],
                stat['mad_rm3'],
                stat['mean_rm3'],
                stat['stdev_rm3'],
                stat['ndet_rm3'],
                stat['median_sigclip_rm3'],
                stat['mad_sigclip_rm3'],
                stat['mean_sigclip_rm3'],
                stat['stdev_sigclip_rm3'],
                stat['ndet_sigclip_rm3'],

                stat['median_ep1'],
                stat['mad_ep1'],
                stat['mean_ep1'],
                stat['stdev_ep1'],
                stat['ndet_ep1'],
                stat['median_sigclip_ep1'],
                stat['mad_sigclip_ep1'],
                stat['mean_sigclip_ep1'],
                stat['stdev_sigclip_ep1'],
                stat['ndet_sigclip_ep1'],

                stat['median_ep2'],
                stat['mad_ep2'],
                stat['mean_ep2'],
                stat['stdev_ep2'],
                stat['ndet_ep2'],
                stat['median_sigclip_ep2'],
                stat['mad_sigclip_ep2'],
                stat['mean_sigclip_ep2'],
                stat['stdev_sigclip_ep2'],
                stat['ndet_sigclip_ep2'],

                stat['median_ep3'],
                stat['mad_ep3'],
                stat['mean_ep3'],
                stat['stdev_ep3'],
                stat['ndet_ep3'],
                stat['median_sigclip_ep3'],
                stat['mad_sigclip_ep3'],
                stat['mean_sigclip_ep3'],
                stat['stdev_sigclip_ep3'],
                stat['ndet_sigclip_ep3'],

                stat['median_tf1'],
                stat['mad_tf1'],
                stat['mean_tf1'],
                stat['stdev_tf1'],
                stat['ndet_tf1'],
                stat['median_sigclip_tf1'],
                stat['mad_sigclip_tf1'],
                stat['mean_sigclip_tf1'],
                stat['stdev_sigclip_tf1'],
                stat['ndet_sigclip_tf1'],

                stat['median_tf2'],
                stat['mad_tf2'],
                stat['mean_tf2'],
                stat['stdev_tf2'],
                stat['ndet_tf2'],
                stat['median_sigclip_tf2'],
                stat['mad_sigclip_tf2'],
                stat['mean_sigclip_tf2'],
                stat['stdev_sigclip_tf2'],
                stat['ndet_sigclip_tf2'],

                stat['median_tf3'],
                stat['mad_tf3'],
                stat['mean_tf3'],
                stat['stdev_tf3'],
                stat['ndet_tf3'],
                stat['median_sigclip_tf3'],
                stat['mad_sigclip_tf3'],
                stat['mean_sigclip_tf3'],
                stat['stdev_sigclip_tf3'],
                stat['ndet_sigclip_tf3'],

                stat['median_rf1'],
                stat['mad_rf1'],
                stat['mean_rf1'],
                stat['stdev_rf1'],
                stat['ndet_rf1'],
                stat['median_sigclip_rf1'],
                stat['mad_sigclip_rf1'],
                stat['mean_sigclip_rf1'],
                stat['stdev_sigclip_rf1'],
                stat['ndet_sigclip_rf1'],

                stat['median_rf2'],
                stat['mad_rf2'],
                stat['mean_rf2'],
                stat['stdev_rf2'],
                stat['ndet_rf2'],
                stat['median_sigclip_rf2'],
                stat['mad_sigclip_rf2'],
                stat['mean_sigclip_rf2'],
                stat['stdev_sigclip_rf2'],
                stat['ndet_sigclip_rf2'],

                stat['median_rf3'],
                stat['mad_rf3'],
                stat['mean_rf3'],
                stat['stdev_rf3'],
                stat['ndet_rf3'],
                stat['median_sigclip_rf3'],
                stat['mad_sigclip_rf3'],
                stat['mean_sigclip_rf3'],
                stat['stdev_sigclip_rf3'],
                stat['ndet_sigclip_rf3'],

                corrmag_ap1,
                corrmag_ap2,
                corrmag_ap3,

                )
            outf.write(outline.encode('utf-8'))

    outf.close()

    print('%sZ: wrote statistics to file %s' %
          (datetime.utcnow().isoformat(), outfile))

    return results


def read_stats_file(statsfile, fovcathasgaiaids=False):
    """
    Reads the stats file into a numpy recarray.
    """

    if fovcathasgaiaids:
        idstrlength = 19
    else:
        idstrlength = 17

    # open the statfile and read all the columns
    stats = np.genfromtxt(
        statsfile,
        dtype=(
            'U{:d},f8,'
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RM1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RM2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RM3
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP3
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF3
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RF1
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RF2
            'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # RF3
            'f8,f8,f8'.format(idstrlength)    # corrmags
        ),
        names=[
            'lcobj','cat_mag',
            'med_rm1','mad_rm1','mean_rm1','stdev_rm1','ndet_rm1',
            'med_sc_rm1','mad_sc_rm1','mean_sc_rm1','stdev_sc_rm1',
            'ndet_sc_rm1',
            'med_rm2','mad_rm2','mean_rm2','stdev_rm2','ndet_rm2',
            'med_sc_rm2','mad_sc_rm2','mean_sc_rm2','stdev_sc_rm2',
            'ndet_sc_rm2',
            'med_rm3','mad_rm3','mean_rm3','stdev_rm3','ndet_rm3',
            'med_sc_rm3','mad_sc_rm3','mean_sc_rm3','stdev_sc_rm3',
            'ndet_sc_rm3',
            'med_ep1','mad_ep1','mean_ep1','stdev_ep1','ndet_ep1',
            'med_sc_ep1','mad_sc_ep1','mean_sc_ep1','stdev_sc_ep1',
            'ndet_sc_ep1',
            'med_ep2','mad_ep2','mean_ep2','stdev_ep2','ndet_ep2',
            'med_sc_ep2','mad_sc_ep2','mean_sc_ep2','stdev_sc_ep2',
            'ndet_sc_ep2',
            'med_ep3','mad_ep3','mean_ep3','stdev_ep3','ndet_ep3',
            'med_sc_ep3','mad_sc_ep3','mean_sc_ep3','stdev_sc_ep3',
            'ndet_sc_ep3',
            'med_tf1','mad_tf1','mean_tf1','stdev_tf1','ndet_tf1',
            'med_sc_tf1','mad_sc_tf1','mean_sc_tf1','stdev_sc_tf1',
            'ndet_sc_tf1',
            'med_tf2','mad_tf2','mean_tf2','stdev_tf2','ndet_tf2',
            'med_sc_tf2','mad_sc_tf2','mean_sc_tf2','stdev_sc_tf2',
            'ndet_sc_tf2',
            'med_tf3','mad_tf3','mean_tf3','stdev_tf3','ndet_tf3',
            'med_sc_tf3','mad_sc_tf3','mean_sc_tf3','stdev_sc_tf3',
            'ndet_sc_tf3',
            'med_rf1','mad_rf1','mean_rf1','stdev_rf1','ndet_rf1',
            'med_sc_rf1','mad_sc_rf1','mean_sc_rf1','stdev_sc_rf1',
            'ndet_sc_rf1',
            'med_rf2','mad_rf2','mean_rf2','stdev_rf2','ndet_rf2',
            'med_sc_rf2','mad_sc_rf2','mean_sc_rf2','stdev_sc_rf2',
            'ndet_sc_rf2',
            'med_rf3','mad_rf3','mean_rf3','stdev_rf3','ndet_rf3',
            'med_sc_rf3','mad_sc_rf3','mean_sc_rf3','stdev_sc_rf3',
            'ndet_sc_rf3',
            'corr_mag_ap1','corr_mag_ap2','corr_mag_ap3',
        ]
    )

    return stats


##########################
## LC BINNING FUNCTIONS ##
##########################

def time_bin_lightcurve(lcprefix,
                        lcexts=('epdlc',
                                'tfalc.TF1','tfalc.TF2','tfalc.TF3'),
                        jdcol=0,
                        lcmagcols=([22,23,24],[25,],[25,],[25,]),
                        binsize=540,
                        outfile=None):
    """
    This bins a lightcurve in time using the binsize given. binsize is in
    seconds.

    Needs only the lcprefix; figures out the fnames and cols using the lcexts
    and lcmagcols kwargs.

    For ISM LCs, use:

    lcmagcols=([27,28,29],[30,],[30,],[30,])

    For gzipped TFA LCs, use:

    lcexts = ('epdlc','tfalc.TF1.gz','tfalc.TF2.gz','tfalc.TF3.gz')
    """

    collected_binned_mags = {}

    for ext, magcolspec in zip(lcexts, lcmagcols):

        lcfname = '%s.%s' % (lcprefix, ext)

        if not os.path.exists(lcfname):
            print('LC: %s does not exist! skipping...' % lcfname)
            continue

        lcmagcols = ['AP%s' % x for x in range(len(magcolspec))]

        collected_binned_mags[ext] = {x:{} for x in lcmagcols}

        # extract the JD and magnitude columns
        if lcfname.endswith('.gz'):
            lcfd = gzip.open(lcfname)
        else:
            lcfd = open(lcfname)

        lcdata = np.genfromtxt(lcfd,
                               usecols=tuple([jdcol] + magcolspec),
                               names=['rjd'] + lcmagcols)

        lcfd.close()

        if lcdata['rjd'].shape and len(lcdata['rjd']) >= 100:

            # convert binsize in seconds to JD units
            binsizejd = binsize/(86400.0)
            nbins = int(np.ceil((np.max(lcdata['rjd']) -
                             np.min(lcdata['rjd']))/binsizejd) + 1)

            minjd = np.min(lcdata['rjd'])
            jdbins = [(minjd + x*binsizejd) for x in range(nbins)]

            # make a KD-tree on the JDs so we can do fast distance calculations
            rjd_coords = np.array([[x,1.0] for x in lcdata['rjd']])
            jdtree = kdtree(rjd_coords)
            binned_timeseries_indices = []

            for jd in jdbins:
                # find all bin indices close to within binsizejd of this point
                # using the kdtree query. we use the p-norm = 1 (I think this
                # means straight-up pairwise distance? FIXME: check this)
                bin_indices = jdtree.query_ball_point(np.array([jd,1.0]),
                                                      binsizejd/2.0, p=1.0)

                # if the bin_indices have already been collected, then we're
                # done with this bin, move to the next one. if they haven't,
                # then this is the start of a new bin.
                if (bin_indices not in binned_timeseries_indices and
                    len(bin_indices)) > 0:
                    binned_timeseries_indices.append(bin_indices)


            # convert to ndarrays
            binned_timeseries_indices = [np.array(x) for x in
                                         binned_timeseries_indices]

            collected_binned_mags[ext]['jdbins'] = binned_timeseries_indices
            collected_binned_mags[ext]['nbins'] = len(binned_timeseries_indices)

            # collect the JDs
            binned_jd = np.array([np.median(lcdata['rjd'][x])
                                  for x in binned_timeseries_indices])
            collected_binned_mags[ext]['RJD'] = binned_jd

            # now collect the mags
            for magcol in lcmagcols:

                print('%s.%s: binning aperture %s mags...' %
                      (lcprefix,ext,magcol))

                mags = lcdata[magcol]
                collected_binned_mags[ext][magcol] = (
                    np.array([np.median(lcdata[magcol][x])
                              for x in binned_timeseries_indices])
                    )


    collected_binned_mags['binsize'] = binsize

    # write everything to a file. This is a pickled file because we might have
    # different row numbers for each column and I don't feel like handling this
    # right now.
    if not outfile:

        outfile = lcprefix + '.binned-%ssec-lc.pkl' % binsize
        outf = open(outfile,'wb')
        pickle.dump(collected_binned_mags, outf, pickle.HIGHEST_PROTOCOL)
        outf.close()

    return outfile


def serial_bin_lightcurves(lcdir,
                           epdlc_glob='*.epdlc',
                           binsizes=[540,3600],
                           lcexts=('epdlc',
                                   'tfalc.TF1','tfalc.TF2','tfalc.TF3'),
                           jdcol=0,
                           lcmagcols=([22,23,24],[25,],[25,],[25,]),
                           nworkers=16,
                           workerntasks=500):

    epdlcfiles = glob.glob(os.path.join(lcdir, epdlc_glob))

    tasks = [(os.path.splitext(x)[0],
              binsizes,
              {'lcexts':lcexts,
               'jdcol':jdcol,
               'lcmagcols':lcmagcols})
             for x in epdlcfiles]

    print('%sZ: %s objects to process, starting serial binning...' %
          (datetime.utcnow().isoformat(), len(epdlcfiles)))

    for epdlc in epdlcfiles:
        lcprefix = os.path.splitext(epdlc)[0]
        for binsize in binsizes:
            binlc = time_bin_lightcurve(lcprefix,binsize=binsize)


def parallel_lcbinning_worker(task):
    """
    This calls time_bin_lightcurve above with all binsizes specified in
    task[1]. task[0] contains the lcprefix, and task[2] is a dict containing
    kwargs, which are expanded.
    """

    try:
        results = [time_bin_lightcurve(task[0],binsize=x,**task[2])
                   for x in task[1]]
    except Exception as e:
        print('something went wrong with %s!' % task)
        results = None

    return results


def parallel_bin_lightcurves(lcdir,
                             epdlc_glob='*.epdlc',
                             binsizes=[540,3600],
                             lcexts=('epdlc',
                                     'tfalc.TF1','tfalc.TF2','tfalc.TF3'),
                             jdcol=0,
                             lcmagcols=([22,23,24],[25,],[25,],[25,]),
                             nworkers=16,
                             workerntasks=1000):
    """
    This bins light curves in time.

    For ISM LCs, use:

    lcmagcols=([27,28,29],[30,],[30,],[30,])

    For gzipped TFA LCs, use:

    lcexts = ('epdlc','tfalc.TF1.gz','tfalc.TF2.gz','tfalc.TF3.gz')
    """

    epdlcfiles = glob.glob(os.path.join(lcdir, epdlc_glob))

    tasks = [(os.path.splitext(x)[0],
              binsizes,
              {'lcexts':lcexts,
               'jdcol':jdcol,
               'lcmagcols':lcmagcols})
             for x in epdlcfiles]

    print('%sZ: %s objects to process, starting parallel binning...' %
          (datetime.utcnow().isoformat(), len(epdlcfiles)))

    pool = mp.Pool(nworkers, maxtasksperchild=workerntasks)
    results = pool.map(parallel_lcbinning_worker, tasks)
    pool.close()
    pool.join()

    print('%sZ: done. %s LCs processed.' %
          (datetime.utcnow().isoformat(), len(epdlcfiles)))



def read_binned_lc(binnedlc):
    """
    This reads back the binnedlc pkl file to a dictionary.
    """
    with open(binnedlc,'rb') as lcf:
        lcdict = pickle.load(lcf)
    return lcdict



def get_binnedlc_statistics(lcfile,
                            sigclip=4.0):
    """
    This collects stats for the binned LCs in the lcfile.
    """

    # read in the lcfile
    binnedlc = read_binned_lc(lcfile)

    # read each lckey and get its statistics
    stats_dict = {}

    # get the EPDLC first
    ep1 = binnedlc['epdlc']['AP0']
    ep2 = binnedlc['epdlc']['AP1']
    ep3 = binnedlc['epdlc']['AP2']

    # then get the TFALC
    try:
        tf1 = binnedlc['tfalc.TF1']['AP0']
        tf2 = binnedlc['tfalc.TF2']['AP0']
        tf3 = binnedlc['tfalc.TF3']['AP0']

    except Exception as e:
        tf1 = binnedlc['tfalc.TF1.gz']['AP0']
        tf2 = binnedlc['tfalc.TF2.gz']['AP0']
        tf3 = binnedlc['tfalc.TF3.gz']['AP0']


    # get stats for each column
    # EP1
    if len(ep1) > 4:

        finiteind = np.isfinite(ep1)
        ep1 = ep1[finiteind]
        median_ep1 = np.median(ep1)
        mad_ep1 = np.median(np.fabs(ep1 - median_ep1))
        mean_ep1 = np.mean(ep1)
        stdev_ep1 = np.std(ep1)
        ndet_ep1 = len(ep1)

        if sigclip:
            sigclip_ep1, lo, hi = stats_sigmaclip(ep1,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_ep1 = np.median(sigclip_ep1)
            mad_sigclip_ep1 = np.median(np.fabs(sigclip_ep1 -
                                                median_sigclip_ep1))
            mean_sigclip_ep1 = np.mean(sigclip_ep1)
            stdev_sigclip_ep1 = np.std(sigclip_ep1)
            ndet_sigclip_ep1 = len(sigclip_ep1)

        else:
            median_sigclip_ep1 = np.nan
            mad_sigclip_ep1 = np.nan
            mean_sigclip_ep1 = np.nan
            stdev_sigclip_ep1 = np.nan
            ndet_sigclip_ep1 = np.nan

    else:

        median_ep1, mad_ep1, mean_ep1, stdev_ep1 = np.nan, np.nan, np.nan, np.nan
        ndet_ep1 = np.nan
        median_sigclip_ep1, mad_sigclip_ep1 = np.nan, np.nan
        mean_sigclip_ep1, stdev_sigclip_ep1 = np.nan, np.nan
        ndet_sigclip_ep1 = np.nan

    # EP2
    if len(ep2) > 4:

        finiteind = np.isfinite(ep2)
        ep2 = ep2[finiteind]
        median_ep2 = np.median(ep2)
        mad_ep2 = np.median(np.fabs(ep2 - median_ep2))
        mean_ep2 = np.mean(ep2)
        stdev_ep2 = np.std(ep2)
        ndet_ep2 = len(ep2)

        if sigclip:
            sigclip_ep2, lo, hi = stats_sigmaclip(ep2,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_ep2 = np.median(sigclip_ep2)
            mad_sigclip_ep2 = np.median(np.fabs(sigclip_ep2 -
                                                median_sigclip_ep2))
            mean_sigclip_ep2 = np.mean(sigclip_ep2)
            stdev_sigclip_ep2 = np.std(sigclip_ep2)
            ndet_sigclip_ep2 = len(sigclip_ep2)

        else:
            median_sigclip_ep2 = np.nan
            mad_sigclip_ep2 = np.nan
            mean_sigclip_ep2 = np.nan
            stdev_sigclip_ep2 = np.nan
            ndet_sigclip_ep2 = np.nan

    else:

        median_ep2, mad_ep2, mean_ep2, stdev_ep2 = np.nan, np.nan, np.nan, np.nan
        ndet_ep2 = np.nan
        median_sigclip_ep2, mad_sigclip_ep2 = np.nan, np.nan
        mean_sigclip_ep2, stdev_sigclip_ep2 = np.nan, np.nan
        ndet_sigclip_ep2 = np.nan

    # EP3
    if len(ep3) > 4:

        finiteind = np.isfinite(ep3)
        ep3 = ep3[finiteind]
        median_ep3 = np.median(ep3)
        mad_ep3 = np.median(np.fabs(ep3 - median_ep3))
        mean_ep3 = np.mean(ep3)
        stdev_ep3 = np.std(ep3)
        ndet_ep3 = len(ep3)

        if sigclip:
            sigclip_ep3, lo, hi = stats_sigmaclip(ep3,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_ep3 = np.median(sigclip_ep3)
            mad_sigclip_ep3 = np.median(np.fabs(sigclip_ep3 -
                                                median_sigclip_ep3))
            mean_sigclip_ep3 = np.mean(sigclip_ep3)
            stdev_sigclip_ep3 = np.std(sigclip_ep3)
            ndet_sigclip_ep3 = len(sigclip_ep3)

        else:
            median_sigclip_ep3 = np.nan
            mad_sigclip_ep3 = np.nan
            mean_sigclip_ep3 = np.nan
            stdev_sigclip_ep3 = np.nan
            ndet_sigclip_ep3 = np.nan

    else:

        median_ep3, mad_ep3, mean_ep3, stdev_ep3 = np.nan, np.nan, np.nan, np.nan
        ndet_ep3 = np.nan
        median_sigclip_ep3, mad_sigclip_ep3 = np.nan, np.nan
        mean_sigclip_ep3, stdev_sigclip_ep3 = np.nan, np.nan
        ndet_sigclip_ep3 = np.nan

    # TF1
    if len(tf1) > 4:

        finiteind = np.isfinite(tf1)
        tf1 = tf1[finiteind]
        median_tf1 = np.median(tf1)
        mad_tf1 = np.median(np.fabs(tf1 - median_tf1))
        mean_tf1 = np.mean(tf1)
        stdev_tf1 = np.std(tf1)
        ndet_tf1 = len(tf1)

        if sigclip:
            sigclip_tf1, lo, hi = stats_sigmaclip(tf1,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_tf1 = np.median(sigclip_tf1)
            mad_sigclip_tf1 = np.median(np.fabs(sigclip_tf1 -
                                                median_sigclip_tf1))
            mean_sigclip_tf1 = np.mean(sigclip_tf1)
            stdev_sigclip_tf1 = np.std(sigclip_tf1)
            ndet_sigclip_tf1 = len(sigclip_tf1)

        else:
            median_sigclip_tf1 = np.nan
            mad_sigclip_tf1 = np.nan
            mean_sigclip_tf1 = np.nan
            stdev_sigclip_tf1 = np.nan
            ndet_sigclip_tf1 = np.nan

    else:

        median_tf1, mad_tf1, mean_tf1, stdev_tf1 = np.nan, np.nan, np.nan, np.nan
        ndet_tf1 = np.nan
        median_sigclip_tf1, mad_sigclip_tf1 = np.nan, np.nan
        mean_sigclip_tf1, stdev_sigclip_tf1 = np.nan, np.nan
        ndet_sigclip_tf1 = np.nan

    # TF2
    if len(tf2) > 4:

        finiteind = np.isfinite(tf2)
        tf2 = tf2[finiteind]
        median_tf2 = np.median(tf2)
        mad_tf2 = np.median(np.fabs(tf2 - median_tf2))
        mean_tf2 = np.mean(tf2)
        stdev_tf2 = np.std(tf2)
        ndet_tf2 = len(tf2)

        if sigclip:
            sigclip_tf2, lo, hi = stats_sigmaclip(tf2,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_tf2 = np.median(sigclip_tf2)
            mad_sigclip_tf2 = np.median(np.fabs(sigclip_tf2 -
                                                median_sigclip_tf2))
            mean_sigclip_tf2 = np.mean(sigclip_tf2)
            stdev_sigclip_tf2 = np.std(sigclip_tf2)
            ndet_sigclip_tf2 = len(sigclip_tf2)

        else:
            median_sigclip_tf2 = np.nan
            mad_sigclip_tf2 = np.nan
            mean_sigclip_tf2 = np.nan
            stdev_sigclip_tf2 = np.nan
            ndet_sigclip_tf2 = np.nan

    else:

        median_tf2, mad_tf2, mean_tf2, stdev_tf2 = np.nan, np.nan, np.nan, np.nan
        ndet_tf2 = np.nan
        median_sigclip_tf2, mad_sigclip_tf2 = np.nan, np.nan
        mean_sigclip_tf2, stdev_sigclip_tf2 = np.nan, np.nan
        ndet_sigclip_tf2 = np.nan

    # TF3
    if len(tf3) > 4:

        finiteind = np.isfinite(tf3)
        tf3 = tf3[finiteind]
        median_tf3 = np.median(tf3)
        mad_tf3 = np.median(np.fabs(tf3 - median_tf3))
        mean_tf3 = np.mean(tf3)
        stdev_tf3 = np.std(tf3)
        ndet_tf3 = len(tf3)

        if sigclip:
            sigclip_tf3, lo, hi = stats_sigmaclip(tf3,
                                                  low=sigclip,
                                                  high=sigclip)
            median_sigclip_tf3 = np.median(sigclip_tf3)
            mad_sigclip_tf3 = np.median(np.fabs(sigclip_tf3 -
                                                median_sigclip_tf3))
            mean_sigclip_tf3 = np.mean(sigclip_tf3)
            stdev_sigclip_tf3 = np.std(sigclip_tf3)
            ndet_sigclip_tf3 = len(sigclip_tf3)

        else:
            median_sigclip_tf3 = np.nan
            mad_sigclip_tf3 = np.nan
            mean_sigclip_tf3 = np.nan
            stdev_sigclip_tf3 = np.nan
            ndet_sigclip_tf3 = np.nan

    else:

        median_tf3, mad_tf3, mean_tf3, stdev_tf3 = np.nan, np.nan, np.nan, np.nan
        ndet_tf3 = np.nan
        median_sigclip_tf3, mad_sigclip_tf3 = np.nan, np.nan
        mean_sigclip_tf3, stdev_sigclip_tf3 = np.nan, np.nan
        ndet_sigclip_tf3 = np.nan

    ## COLLECT STATS

    print('%sZ: collected binned lc statistics for %s' %
          (datetime.utcnow().isoformat(), lcfile))

    return {'lcfile':lcfile,
            'lcobj':os.path.splitext(os.path.basename(lcfile))[0],
            # EPD mags aperture 1
            'median_ep1':median_ep1,
            'mad_ep1':mad_ep1,
            'mean_ep1':mean_ep1,
            'stdev_ep1':stdev_ep1,
            'ndet_ep1':ndet_ep1,
            'median_sigclip_ep1':median_sigclip_ep1,
            'mad_sigclip_ep1':mad_sigclip_ep1,
            'mean_sigclip_ep1':mean_sigclip_ep1,
            'stdev_sigclip_ep1':stdev_sigclip_ep1,
            'ndet_sigclip_ep1':ndet_sigclip_ep1,
            # EPD mags aperture 2
            'median_ep2':median_ep2,
            'mad_ep2':mad_ep2,
            'mean_ep2':mean_ep2,
            'stdev_ep2':stdev_ep2,
            'ndet_ep2':ndet_ep2,
            'median_sigclip_ep2':median_sigclip_ep2,
            'mad_sigclip_ep2':mad_sigclip_ep2,
            'mean_sigclip_ep2':mean_sigclip_ep2,
            'stdev_sigclip_ep2':stdev_sigclip_ep2,
            'ndet_sigclip_ep2':ndet_sigclip_ep2,
            # EPD mags aperture 3
            'median_ep3':median_ep3,
            'mad_ep3':mad_ep3,
            'mean_ep3':mean_ep3,
            'stdev_ep3':stdev_ep3,
            'ndet_ep3':ndet_ep3,
            'median_sigclip_ep3':median_sigclip_ep3,
            'mad_sigclip_ep3':mad_sigclip_ep3,
            'mean_sigclip_ep3':mean_sigclip_ep3,
            'stdev_sigclip_ep3':stdev_sigclip_ep3,
            'ndet_sigclip_ep3':ndet_sigclip_ep3,
            # TFA mags aperture 1
            'median_tf1':median_tf1,
            'mad_tf1':mad_tf1,
            'mean_tf1':mean_tf1,
            'stdev_tf1':stdev_tf1,
            'ndet_tf1':ndet_tf1,
            'median_sigclip_tf1':median_sigclip_tf1,
            'mad_sigclip_tf1':mad_sigclip_tf1,
            'mean_sigclip_tf1':mean_sigclip_tf1,
            'stdev_sigclip_tf1':stdev_sigclip_tf1,
            'ndet_sigclip_tf1':ndet_sigclip_tf1,
            # TFA mags aperture 2
            'median_tf2':median_tf2,
            'mad_tf2':mad_tf2,
            'mean_tf2':mean_tf2,
            'stdev_tf2':stdev_tf2,
            'ndet_tf2':ndet_tf2,
            'median_sigclip_tf2':median_sigclip_tf2,
            'mad_sigclip_tf2':mad_sigclip_tf2,
            'mean_sigclip_tf2':mean_sigclip_tf2,
            'stdev_sigclip_tf2':stdev_sigclip_tf2,
            'ndet_sigclip_tf2':ndet_sigclip_tf2,
            # TFA mags aperture 3
            'median_tf3':median_tf3,
            'mad_tf3':mad_tf3,
            'mean_tf3':mean_tf3,
            'stdev_tf3':stdev_tf3,
            'ndet_tf3':ndet_tf3,
            'median_sigclip_tf3':median_sigclip_tf3,
            'mad_sigclip_tf3':mad_sigclip_tf3,
            'mean_sigclip_tf3':mean_sigclip_tf3,
            'stdev_sigclip_tf3':stdev_sigclip_tf3,
            'ndet_sigclip_tf3':ndet_sigclip_tf3}


def binnedlc_statistics_worker(task):
    """
    This is a worker that runs the function above in a parallel worker pool.
    """

    try:
        return get_binnedlc_statistics(task[0], **task[1])
    except Exception as e:
        print('SOMETHING WENT WRONG! task was %s' % task)
        return None



def parallel_binnedlc_statistics(lcdir,
                                 lcglob,
                                 fovcatalog,
                                 fovcathasgaiaids=False,
                                 fovcatcols=(0,9), # objectid, magcol to use
                                 fovcatmaglabel='r',
                                 corrmagsource=None,
                                 corrmag_idcol=0,
                                 corrmag_magcols=[122,123,124],
                                 outfile=None,
                                 nworkers=16,
                                 workerntasks=500,
                                 sigclip=4.0):
    """
    This calculates statistics on all binned lc files in lcdir.

    Puts the results in text file outfile.

    If corrmagsource is not None, will add in corrected mag cols using the file
    specified, corrmag_idcol for the objectid column, and corrmag_magcols for
    the magnitude columns.

    outfile contains the following columns:

    object, ndet,
    median EP[1-3], MAD EP[1-3], mean EP[1-3], stdev EP[1-3],
    median TF[1-3], MAD TF[1-3], mean TF[1-3], stdev TF[1-3]

    if a value is missing, it will be np.nan.
    """

    lcfiles = glob.glob(os.path.join(lcdir, lcglob))

    tasks = [[x, {'sigclip':sigclip}] for x in lcfiles]

    pool = mp.Pool(nworkers,maxtasksperchild=workerntasks)
    results = pool.map(binnedlc_statistics_worker, tasks)
    pool.close()
    pool.join()

    print('%sZ: done. %s lightcurves processed.' %
          (datetime.utcnow().isoformat(), len(lcfiles)))

    if not outfile:
        outfile = os.path.join(lcdir, 'binned-lightcurve-statistics.txt')

    outf = open(outfile,'wb')

    if corrmagsource and os.path.exists(corrmagsource):

        print('getting corrected mags from %s' % corrmagsource)

        if corrmagsource.endswith('.gz'):
            corrmags = gzip.open(corrmagsource)
        else:
            corrmags = open(corrmagsource)

        corrmaginfo = np.genfromtxt(corrmags,
                                    usecols=tuple([corrmag_idcol] +
                                                  corrmag_magcols),
                                    names=('hatid','ap1','ap2','ap3'),
                                    dtype='S20,f8,f8,f8')

    else:
        corrmaginfo = None

    outlineformat = (
        '%s %.3f  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.6f %.6f %.6f %.6f %s %.6f %.6f %.6f %.6f %s  '
        '%.3f %.3f %.3f\n'
        )

    outcolumnkey = (
        '# columns are:\n'
        '# 0,1: object, catalog mag %s\n'
        '# 2,3,4,5,6: median EP1, MAD EP1, mean EP1, stdev EP1, ndet EP1\n'
        '# 7,8,9,10,11: sigma-clipped median EP1, MAD EP1, mean EP1, '
        'stdev EP1, ndet EP1\n'
        '# 12,13,14,15,16: median EP2, MAD EP2, mean EP2, stdev EP2, '
        'ndet EP2\n'
        '# 17,18,19,20,21: sigma-clipped median EP2, MAD EP2, mean EP2, '
        'stdev EP2, ndet EP2\n'
        '# 22,23,24,25,26: median EP3, MAD EP3, mean EP3, stdev EP3, '
        'ndet EP3\n'
        '# 27,28,29,30,31: sigma-clipped median EP3, MAD EP3, mean EP3, '
        'stdev EP3, ndet EP3\n'
        '# 32,33,34,35,36: median TF1, MAD TF1, mean TF1, stdev TF1, '
        'ndet TF1\n'
        '# 37,38,39,40,41: sigma-clipped median TF1, MAD TF1, mean TF1, '
        'stdev TF1, ndet TF1\n'
        '# 42,43,44,45,46: median TF2, MAD TF2, mean TF2, stdev TF2, '
        'ndet TF2\n'
        '# 47,48,49,50,51: sigma-clipped median TF2, MAD TF2, mean TF2, '
        'stdev TF2, ndet TF2\n'
        '# 52,53,54,55,56: median TF3, MAD TF3, mean TF3, stdev TF3, '
        'ndet TF3\n'
        '# 57,58,59,60,61: sigma-clipped median TF3, MAD TF3, mean TF3, '
        'stdev TF3, ndet TF3\n'
        '# 62, 63, 64: corrected catalog mags AP1, AP2, AP3\n'
        ) % fovcatmaglabel


    # write the header to the file
    outheader = '# total objects: %s, sigmaclip used: %s\n' % (len(lcfiles),
                                                               sigclip)
    outf.write(outheader.encode('utf-8'))
    outf.write(outcolumnkey.encode('utf-8'))

    # open the fovcatalog and read in the column magnitudes and hatids
    if not fovcathasgaiaids:
        # assume HAT-IDs, HAT-123-4567890, 17 character strings
        fovcat = np.genfromtxt(fovcatalog,
                               usecols=fovcatcols,
                               dtype='S17,f8',
                               names=['objid','mag'])
    else:
        # assume GAIA-IDs. From gaia2read, with "GAIA" id option, this is just
        # 19 character integers.
        fovcat = np.genfromtxt(fovcatalog,
                               usecols=fovcatcols,
                               dtype='S19,f8',
                               names=['objid','mag'])
    fovdict = dict(fovcat)

    for stat in results:
        if stat is not None:
            # find the catalog mag for this object
            if stat['lcobj'] in fovdict:
                catmag = fovdict[stat['lcobj']]
            else:
                print('no catalog mag for %s, using median TF3 mag' %
                      stat['lcobj'])
                catmag = stat['median_tf3']

            # find the corrected mag for this source if possible
            if corrmaginfo is not None:
                try:
                    corrmag_ap1 = corrmaginfo['ap1'][
                        np.where(corrmaginfo['hatid'] == stat['lcobj'][:15])
                    ]
                    if not corrmag_ap1[0]:
                        print('no corrected AP1 mag for %s, using med TF1 mag'
                              % (stat['lcobj']))
                        corrmag_ap1 = stat['median_tf1']

                    corrmag_ap2 = corrmaginfo['ap2'][
                        np.where(corrmaginfo['hatid'] == stat['lcobj'][:15])
                    ]
                    if not corrmag_ap2[0]:
                        print('no corrected AP2 mag for %s, using med TF2 mag'
                              % (stat['lcobj']))
                        corrmag_ap2 = stat['median_tf2']

                    corrmag_ap3 = corrmaginfo['ap3'][
                        np.where(corrmaginfo['hatid'] == stat['lcobj'][:15])
                    ]
                    if not corrmag_ap3[0]:
                        print('no corrected AP3 mag for %s, using med TF3 mag'
                              % (stat['lcobj']))
                        corrmag_ap3 = stat['median_tf3']

                except Exception as e:
                    print('no corrected mags for %s, using median TF mags' %
                          stat['lcobj'])
                    corrmag_ap1 = stat['median_tf1']
                    corrmag_ap2 = stat['median_tf2']
                    corrmag_ap3 = stat['median_tf3']

            else:
                corrmag_ap1 = catmag
                corrmag_ap2 = catmag
                corrmag_ap3 = catmag


            outlinelist = [
                stat['lcobj'],
                catmag,

                stat['median_ep1'],
                stat['mad_ep1'],
                stat['mean_ep1'],
                stat['stdev_ep1'],
                stat['ndet_ep1'],
                stat['median_sigclip_ep1'],
                stat['mad_sigclip_ep1'],
                stat['mean_sigclip_ep1'],
                stat['stdev_sigclip_ep1'],
                stat['ndet_sigclip_ep1'],

                stat['median_ep2'],
                stat['mad_ep2'],
                stat['mean_ep2'],
                stat['stdev_ep2'],
                stat['ndet_ep2'],
                stat['median_sigclip_ep2'],
                stat['mad_sigclip_ep2'],
                stat['mean_sigclip_ep2'],
                stat['stdev_sigclip_ep2'],
                stat['ndet_sigclip_ep2'],

                stat['median_ep3'],
                stat['mad_ep3'],
                stat['mean_ep3'],
                stat['stdev_ep3'],
                stat['ndet_ep3'],
                stat['median_sigclip_ep3'],
                stat['mad_sigclip_ep3'],
                stat['mean_sigclip_ep3'],
                stat['stdev_sigclip_ep3'],
                stat['ndet_sigclip_ep3'],

                stat['median_tf1'],
                stat['mad_tf1'],
                stat['mean_tf1'],
                stat['stdev_tf1'],
                stat['ndet_tf1'],
                stat['median_sigclip_tf1'],
                stat['mad_sigclip_tf1'],
                stat['mean_sigclip_tf1'],
                stat['stdev_sigclip_tf1'],
                stat['ndet_sigclip_tf1'],

                stat['median_tf2'],
                stat['mad_tf2'],
                stat['mean_tf2'],
                stat['stdev_tf2'],
                stat['ndet_tf2'],
                stat['median_sigclip_tf2'],
                stat['mad_sigclip_tf2'],
                stat['mean_sigclip_tf2'],
                stat['stdev_sigclip_tf2'],
                stat['ndet_sigclip_tf2'],

                stat['median_tf3'],
                stat['mad_tf3'],
                stat['mean_tf3'],
                stat['stdev_tf3'],
                stat['ndet_tf3'],
                stat['median_sigclip_tf3'],
                stat['mad_sigclip_tf3'],
                stat['mean_sigclip_tf3'],
                stat['stdev_sigclip_tf3'],
                stat['ndet_sigclip_tf3'],

                corrmag_ap1,
                corrmag_ap2,
                corrmag_ap3
            ]

            # write the output line
            outline = outlineformat % tuple(outlinelist)
            outf.write(outline.encode('utf-8'))

    outf.close()

    print('%sZ: wrote statistics to file %s' %
          (datetime.utcnow().isoformat(), outfile))

    return results



def read_binnedlc_stats_file(statsfile):
    """
    Reads the stats file into a numpy recarray.
    """

    # open the statfile and read all the columns
    stats = np.genfromtxt(
        statsfile,
        dtype=('S17,f8,'
               'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP1
               'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP2
               'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # EP3
               'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF1
               'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF2
               'f8,f8,f8,f8,i8,f8,f8,f8,f8,i8,'  # TF3
               'f8,f8,f8'),                      # corrected mags
        names=['lcobj','cat_mag',
               'med_ep1','mad_ep1','mean_ep1','stdev_ep1','ndet_ep1',
               'med_sc_ep1','mad_sc_ep1','mean_sc_ep1','stdev_sc_ep1',
               'ndet_sc_ep1',
               'med_ep2','mad_ep2','mean_ep2','stdev_ep2','ndet_ep2',
               'med_sc_ep2','mad_sc_ep2','mean_sc_ep2','stdev_sc_ep2',
               'ndet_sc_ep2',
               'med_ep3','mad_ep3','mean_ep3','stdev_ep3','ndet_ep3',
               'med_sc_ep3','mad_sc_ep3','mean_sc_ep3','stdev_sc_ep3',
               'ndet_sc_ep3',
               'med_tf1','mad_tf1','mean_tf1','stdev_tf1','ndet_tf1',
               'med_sc_tf1','mad_sc_tf1','mean_sc_tf1','stdev_sc_tf1',
               'ndet_sc_tf1',
               'med_tf2','mad_tf2','mean_tf2','stdev_tf2','ndet_tf2',
               'med_sc_tf2','mad_sc_tf2','mean_sc_tf2','stdev_sc_tf2',
               'ndet_sc_tf2',
               'med_tf3','mad_tf3','mean_tf3','stdev_tf3','ndet_tf3',
               'med_sc_tf3','mad_sc_tf3','mean_sc_tf3','stdev_sc_tf3',
               'ndet_sc_tf3',
               'corr_mag_ap1','corr_mag_ap2','corr_mag_ap3']
        )

    return stats



########################
## PLOTTING FUNCTIONS ##
########################

# lists all plots to make plus columns to use and metadata
MAD_STATS_PLOTS = {
    'median-rm1-vs-mad-rm1':{
        'xcol':'med_rm1',
        'ycol':'mad_rm1',
        'title':'RM1 median mag vs. RM1 median abs. dev.',
        'xlabel':'RM1 median magnitude',
        'ylabel':'RM1 median abs. dev.',
        'binned':False
    },
    'median-rm1-vs-mad-rm1-sigclipped':{
        'xcol':'med_sc_rm1',
        'ycol':'mad_sc_rm1',
        'title':'RM1 median mag vs. RM1 median abs. dev. (sigclip LCs)',
        'xlabel':'RM1 median magnitude',
        'ylabel':'RM1 median abs. dev.',
        'binned':False
    },
    'median-rm2-vs-mad-rm2':{
        'xcol':'med_rm2',
        'ycol':'mad_rm2',
        'title':'RM2 median mag vs. RM2 median abs. dev.',
        'xlabel':'RM2 median magnitude',
        'ylabel':'RM2 median abs. dev.',
        'binned':False
    },
    'median-rm2-vs-mad-rm2-sigclipped':{
        'xcol':'med_sc_rm2',
        'ycol':'mad_sc_rm2',
        'title':'RM2 median mag vs. RM2 median abs. dev. (sigclip LCs)',
        'xlabel':'RM2 median magnitude',
        'ylabel':'RM2 median abs. dev.',
        'binned':False
    },
    'median-rm3-vs-mad-rm3':{
        'xcol':'med_rm3',
        'ycol':'mad_rm3',
        'title':'RM3 median mag vs. RM3 median abs. dev.',
        'xlabel':'RM3 median magnitude',
        'ylabel':'RM3 median abs. dev.',
        'binned':False
    },
    'median-rm3-vs-mad-rm3-sigclipped':{
        'xcol':'med_sc_rm3',
        'ycol':'mad_sc_rm3',
        'title':'RM3 median mag vs. RM3 median abs. dev. (sigclip LCs)',
        'xlabel':'RM3 median magnitude',
        'ylabel':'RM3 median abs. dev.',
        'binned':False
    },
    'median-EP1-vs-mad-EP1':{
        'xcol':'med_ep1',
        'ycol':'mad_ep1',
        'title':'EP1 median mag vs. EP1 median abs. dev.',
        'xlabel':'EP1 median magnitude',
        'ylabel':'EP1 median abs. dev.',
        'binned':True
    },
    'median-EP1-vs-mad-EP1-sigclipped':{
        'xcol':'med_sc_ep1',
        'ycol':'mad_sc_ep1',
        'title':'EP1 median mag vs. EP1 median abs. dev. (sigclip LCs)',
        'xlabel':'EP1 median magnitude',
        'ylabel':'EP1 median abs. dev.',
        'binned':True
    },
    'median-EP2-vs-mad-EP2':{
        'xcol':'med_ep2',
        'ycol':'mad_ep2',
        'title':'EP2 median mag vs. EP2 median abs. dev.',
        'xlabel':'EP2 median magnitude',
        'ylabel':'EP2 median abs. dev.',
        'binned':True
    },
    'median-EP2-vs-mad-EP2-sigclipped':{
        'xcol':'med_sc_ep2',
        'ycol':'mad_sc_ep2',
        'title':'EP2 median mag vs. EP2 median abs. dev. (sigclip LCs)',
        'xlabel':'EP2 median magnitude',
        'ylabel':'EP2 median abs. dev.',
        'binned':True
    },
    'median-EP3-vs-mad-EP3':{
        'xcol':'med_ep3',
        'ycol':'mad_ep3',
        'title':'EP3 median mag vs. EP3 median abs. dev.',
        'xlabel':'EP3 median magnitude',
        'ylabel':'EP3 median abs. dev.',
        'binned':True
    },
    'median-EP3-vs-mad-EP3-sigclipped':{
        'xcol':'med_sc_ep3',
        'ycol':'mad_sc_ep3',
        'title':'EP3 median mag vs. EP3 median abs. dev. (sigclip LCs)',
        'xlabel':'EP3 median magnitude',
        'ylabel':'EP3 median abs. dev.',
        'binned':True
    },
    'median-TF1-vs-mad-TF1':{
        'xcol':'med_tf1',
        'ycol':'mad_tf1',
        'title':'TF1 median mag vs. TF1 median abs. dev.',
        'xlabel':'TF1 median magnitude',
        'ylabel':'TF1 median abs. dev.',
        'binned':True
    },
    'median-TF1-vs-mad-TF1-sigclipped':{
        'xcol':'med_sc_tf1',
        'ycol':'mad_sc_tf1',
        'title':'TF1 median mag vs. TF1 median abs. dev. (sigclip LCs)',
        'xlabel':'TF1 median magnitude',
        'ylabel':'TF1 median abs. dev.',
        'binned':True
    },
    'median-TF2-vs-mad-TF2':{
        'xcol':'med_tf2',
        'ycol':'mad_tf2',
        'title':'TF2 median mag vs. TF2 median abs. dev.',
        'xlabel':'TF2 median magnitude',
        'ylabel':'TF2 median abs. dev.',
        'binned':True
    },
    'median-TF2-vs-mad-TF2-sigclipped':{
        'xcol':'med_sc_tf2',
        'ycol':'mad_sc_tf2',
        'title':'TF2 median mag vs. TF2 median abs. dev. (sigclip LCs)',
        'xlabel':'TF2 median magnitude',
        'ylabel':'TF2 median abs. dev.',
        'binned':True
    },
    'median-TF3-vs-mad-TF3':{
        'xcol':'med_tf3',
        'ycol':'mad_tf3',
        'title':'TF3 median mag vs. TF3 median abs. dev.',
        'xlabel':'TF3 median magnitude',
        'ylabel':'TF3 median abs. dev.',
        'binned':True
    },
    'median-TF3-vs-mad-TF3-sigclipped':{
        'xcol':'med_sc_tf3',
        'ycol':'mad_sc_tf3',
        'title':'TF3 median mag vs. TF3 median abs. dev. (sigclip LCs)',
        'xlabel':'TF3 median magnitude',
        'ylabel':'TF3 median abs. dev.',
        'binned':True
    },
    'catalog-r-mag-vs-mad-TF1':{
        'xcol':'cat_mag',
        'ycol':'mad_tf1',
        'title':'catalog SDSS r mag vs. TF1 median abs. dev.',
        'xlabel':'catalog SDSS r mag',
        'ylabel':'TF1 median abs. dev.',
        'binned':True
    },
    'catalog-r-mag-vs-mad-TF2':{
        'xcol':'cat_mag',
        'ycol':'mad_tf2',
        'title':'catalog SDSS r mag vs. TF2 median abs. dev.',
        'xlabel':'catalog SDSS r mag',
        'ylabel':'TF2 median abs. dev.',
        'binned':True
    },
    'catalog-r-mag-vs-mad-TF3':{
        'xcol':'cat_mag',
        'ycol':'mad_tf3',
        'title':'catalog SDSS r mag vs. TF3 median abs. dev.',
        'xlabel':'catalog SDSS r mag',
        'ylabel':'TF3 median abs. dev.',
        'binned':True
    },
    'corr-r-ap1-mag-vs-mad-bestap':{
        'xcol':'corr_mag_ap1',
        'ycol':('mad_tf1','mad_tf2','mad_tf3'),
        'title':'catalog SDSS r mag vs. TFA bestap MAD',
        'xlabel':'catalog SDSS r mag',
        'ylabel':'TFA best aperture median abs. dev.',
        'binned':True
    },
    'corr-r-ap2-mag-vs-mad-bestap':{
        'xcol':'corr_mag_ap2',
        'ycol':('mad_tf1','mad_tf2','mad_tf3'),
        'title':'catalog SDSS r mag vs. TFA bestap MAD',
        'xlabel':'catalog SDSS r mag',
        'ylabel':'TFA best aperture median abs. dev.',
        'binned':True
    },
    'corr-r-ap3-mag-vs-mad-bestap':{
        'xcol':'corr_mag_ap3',
        'ycol':('mad_tf1','mad_tf2','mad_tf3'),
        'title':'catalog SDSS r mag vs. TFA bestap MAD',
        'xlabel':'catalog SDSS r mag',
        'ylabel':'TFA best aperture median abs. dev.',
        'binned':True
    },
    'catalog-r-mag-vs-mad-bestap':{
        'xcol':'cat_mag',
        'ycol':('mad_tf1','mad_tf2','mad_tf3'),
        'title':'catalog SDSS r mag vs. TFA bestap MAD',
        'xlabel':'catalog SDSS r mag',
        'ylabel':'TFA best aperture median abs. dev.',
        'binned':True
    },
}


RMS_STATS_PLOTS = {
    'median-rm1-vs-rms-rm1':{
        'xcol':'med_rm1',
        'ycol':'stdev_rm1',
        'title':'RM1 median mag vs. RM1 std dev (RMS)',
        'xlabel':'RM1 median magnitude',
        'ylabel':'RM1 std dev (RMS)',
        'binned':False
    },
    'median-rm2-vs-rms-rm2':{
        'xcol':'med_rm2',
        'ycol':'stdev_rm2',
        'title':'RM2 median mag vs. RM2 std dev (RMS)',
        'xlabel':'RM2 median magnitude',
        'ylabel':'RM2 std dev (RMS)',
        'binned':False
    },
    'median-rm3-vs-rms-rm3':{
        'xcol':'med_rm3',
        'ycol':'stdev_rm3',
        'title':'RM3 median mag vs. RM3 std dev (RMS)',
        'xlabel':'RM3 median magnitude',
        'ylabel':'RM3 std dev (RMS)',
        'binned':False
    },
    'median-EP1-vs-rms-EP1':{
        'xcol':'med_ep1',
        'ycol':'stdev_ep1',
        'title':'EP1 median mag vs. EP1 std dev (RMS)',
        'xlabel':'EP1 median magnitude',
        'ylabel':'EP1 std dev (RMS)',
        'binned':True
    },
    'median-EP2-vs-rms-EP2':{
        'xcol':'med_ep2',
        'ycol':'stdev_ep2',
        'title':'EP2 median mag vs. EP2 std dev (RMS)',
        'xlabel':'EP2 median magnitude',
        'ylabel':'EP2 std dev (RMS)',
        'binned':True
    },
    'median-EP3-vs-rms-EP3':{
        'xcol':'med_ep3',
        'ycol':'stdev_ep3',
        'title':'EP3 median mag vs. EP3 std dev (RMS)',
        'xlabel':'EP3 median magnitude',
        'ylabel':'EP3 std dev (RMS)',
        'binned':True
    },
    'median-TF1-vs-rms-TF1':{
        'xcol':'med_tf1',
        'ycol':'stdev_tf1',
        'title':'TF1 median mag vs. TF1 std dev (RMS)',
        'xlabel':'TF1 median magnitude',
        'ylabel':'TF1 std dev (RMS)',
        'binned':True
    },
    'median-TF2-vs-rms-TF2':{
        'xcol':'med_tf2',
        'ycol':'stdev_tf2',
        'title':'TF2 median mag vs. TF2 std dev (RMS)',
        'xlabel':'TF2 median magnitude',
        'ylabel':'TF2 std dev (RMS)',
        'binned':True
    },
    'median-TF3-vs-rms-TF3':{
        'xcol':'med_tf3',
        'ycol':'stdev_tf3',
        'title':'TF3 median mag vs. TF3 std dev (RMS)',
        'xlabel':'TF3 median magnitude',
        'ylabel':'TF3 std dev (RMS)',
        'binned':True
    },
    'catalog-r-mag-vs-rms-TF1':{
        'xcol':'cat_mag',
        'ycol':'stdev_tf1',
        'title':'catalog Gaia Rp mag vs. TF1 std dev (RMS)',
        'xlabel':'catalog Gaia Rp mag',
        'ylabel':'TF1 std dev (RMS)',
        'binned':True
    },
    'catalog-r-mag-vs-rms-TF2':{
        'xcol':'cat_mag',
        'ycol':'stdev_tf2',
        'title':'catalog Gaia Rp mag vs. TF2 std dev (RMS)',
        'xlabel':'catalog Gaia Rp mag',
        'ylabel':'TF2 std dev (RMS)',
        'binned':True
    },
    'catalog-r-mag-vs-rms-TF3':{
        'xcol':'cat_mag',
        'ycol':'stdev_tf3',
        'title':'catalog Gaia Rp mag vs. TF3 std dev (RMS)',
        'xlabel':'catalog Gaia Rp mag',
        'ylabel':'TF3 std dev (RMS)',
        'binned':True
    },
    'corr-r-ap1-mag-vs-rms-bestap':{
        'xcol':'corr_mag_ap1',
        'ycol':('stdev_tf1','stdev_tf2','stdev_tf3'),
        'title':'catalog Gaia Rp mag vs. TFA bestap rms',
        'xlabel':'catalog Gaia Rp mag',
        'ylabel':'TFA best aperture std. dev. (RMS)',
        'binned':True
    },
    'corr-r-ap2-mag-vs-rms-bestap':{
        'xcol':'corr_mag_ap2',
        'ycol':('stdev_tf1','stdev_tf2','stdev_tf3'),
        'title':'catalog Gaia Rp mag vs. TFA bestap rms',
        'xlabel':'catalog Gaia Rp mag',
        'ylabel':'TFA best aperture std. dev. (RMS)',
        'binned':True
    },
    'corr-r-ap3-mag-vs-rms-bestap':{
        'xcol':'corr_mag_ap3',
        'ycol':('stdev_tf1','stdev_tf2','stdev_tf3'),
        'title':'catalog Gaia Rp mag vs. TFA bestap rms',
        'xlabel':'catalog Gaia Rp mag',
        'ylabel':'TFA best aperture std. dev. (RMS)',
        'binned':True
    },
    'catalog-r-mag-vs-rms-bestap':{
        'xcol':'cat_mag',
        'ycol':('stdev_tf1','stdev_tf2','stdev_tf3'),
        'title':'catalog Gaia Rp mag vs. TFA bestap rms',
        'xlabel':'catalog Gaia Rp mag',
        'ylabel':'TFA best aperture std. dev. (RMS)',
        'binned':True
    },
}



def plot_stats_file(statsfile, outdir, outprefix,
                    binned=False,
                    logy=False,
                    logx=False,
                    correctmagsafter=None,
                    rangex=(5.9,14.1),
                    observatory='hatpi',
                    fovcathasgaiaids=False,
                    yaxisval='MAD'):
    """
    This plots MAD vs magnitude for RAW, EPD, TFA for all apertures.

    args:

        statsfile (str): path to statsfile

        outdir (str): path to directory that writes plots

        outprefix (str): string to prefix on plots

    kwargs:

        yaxisval (str): "MAD" or "RMS".

        binned (bool, or int): whether the passed statsfile is binned, or not.
        If an integer is passed, it is assumed to be the bin cadence in
        seconds. This is then used to overplot model precision curves.

        correctmagsafter (float): use the corrected mags for all objects with
        mags > than this value. This is used for crowded fields where the
        catalog photometry may not be as precise, so will get incorrect mags
        for fainter stars.
    """

    plt.close('all')

    # read the stats file
    if binned:
        stats = read_binnedlc_stats_file(statsfile)
    else:
        stats = read_stats_file(statsfile, fovcathasgaiaids=fovcathasgaiaids)

    if yaxisval=='MAD':
        STATS_PLOTS = MAD_STATS_PLOTS
    elif yaxisval=='RMS':
        STATS_PLOTS = RMS_STATS_PLOTS
    else:
        raise ValueError('yaxisval must be MAD or RMS')

    for plot in STATS_PLOTS:

        try:

            if binned and not STATS_PLOTS[plot]['binned']:
                print('not making %s for binned LCs' % plot)
                continue

            if logy:
                logypostfix = '-logy'
            else:
                logypostfix = ''
            if logx:
                logxpostfix = '-logx'
            else:
                logxpostfix = ''

            outfile = os.path.join(outdir, '%s-%s%s%s.png' % (outprefix,
                                                              plot,
                                                              logxpostfix,
                                                              logypostfix))

            # if this is a bestap plot, then do special processing
            if 'bestap' in plot:

                # get the magnitudes
                xcol = stats[STATS_PLOTS[plot]['xcol']]

                # if we're supposed to correct the magnitudes
                if correctmagsafter and 'corr_mag_ap1' in stats.dtype.names:

                    # replace the bad mags with the corrected mags
                    tocorrectindex = xcol > correctmagsafter
                    xcol[tocorrectindex] = stats['corr_mag_ap1'][tocorrectindex]

                ycol1 = stats[STATS_PLOTS[plot]['ycol'][0]]
                ycol2 = stats[STATS_PLOTS[plot]['ycol'][1]]
                ycol3 = stats[STATS_PLOTS[plot]['ycol'][2]]

                # remove any spurious zeros (caused by TFA?)
                ycol1[ycol1 == 0.0] = np.nan
                ycol2[ycol2 == 0.0] = np.nan
                ycol3[ycol3 == 0.0] = np.nan

                # make a column stack of the MAD arrays
                bestapstack = np.column_stack((ycol1, ycol2, ycol3))

                # get the min of each row of the stack, this is the best MAD
                ycol = np.nanmin(bestapstack,axis=1)

                xlabel, ylabel = (STATS_PLOTS[plot]['xlabel'],
                                  STATS_PLOTS[plot]['ylabel'])
                title = '%s - %s - %s' % (outprefix,
                                          len(xcol),
                                          STATS_PLOTS[plot]['title'])

            # otherwise, make the same old boring plots
            else:

                xcol, ycol = (stats[STATS_PLOTS[plot]['xcol']],
                              stats[STATS_PLOTS[plot]['ycol']])

                # if we're supposed to correct the magnitudes
                if correctmagsafter and 'corr_mag_ap1' in stats.dtype.names:

                    # replace the bad mags with the corrected mags
                    tocorrectindex = xcol > correctmagsafter
                    xcol[tocorrectindex] = stats['corr_mag_ap1'][tocorrectindex]

                xlabel, ylabel = (STATS_PLOTS[plot]['xlabel'],
                                  STATS_PLOTS[plot]['ylabel'])
                title = '%s - %s - %s' % (outprefix,
                                          len(xcol),
                                          STATS_PLOTS[plot]['title'])

            # make the plot
            if logy:
                plt.scatter(xcol, ycol,
                            s=1,
                            marker='.')
                plt.yscale('log',basey=10.0)
            elif logx:
                plt.scatter(xcol, ycol,
                            s=1,
                            marker='.')
                plt.xscale('log',basex=10.0)
            elif logx and logy:
                plt.scatter(xcol, ycol,
                            s=1,
                            marker='.')
                plt.xscale('log',basex=10.0)
                plt.yscale('log',basey=10.0)
            else:
                plt.scatter(xcol, ycol,
                            s=1,
                            marker='.')

            # put the labels on the plot
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xlim(rangex)

            if binned:
                if observatory=='hatpi':
                    plt.ylim((0.0001,1.0))
                    plt.hlines([0.001,0.002,0.003],
                               xmin=rangex[0],xmax=rangex[1],colors='b')
                    plt.hlines([0.0005],
                               xmin=rangex[0],xmax=rangex[1],colors='r')
                elif observatory=='tess' and yaxisval=='RMS':
                    plt.ylim((0.00001,1.0))

                    # overplot toy tess noise model
                    Tmag = np.linspace(6, 13, num=200)
                    lnA = 3.29685004771
                    B = 0.8500214657
                    C = -0.2850416324
                    D = 0.039590832137
                    E = -0.00223080159
                    F = 4.73508403525e-5
                    ln_sigma_1hr = lnA + B*Tmag + C*Tmag**2 + D*Tmag**3 + \
                                   E*Tmag**4 + F*Tmag**5
                    sigma_1hr = np.exp(ln_sigma_1hr)

                    if isinstance(binned,int):
                        bincadence_hr = binned/3600

                    sigma_binned = sigma_1hr * np.sqrt(1/bincadence_hr)
                    plt.plot(Tmag, sigma_binned/1e6, 'k-', zorder=3, lw=2)

            else:
                if observatory=='hatpi':
                    plt.ylim((0.0009,1.0))

                    # make the horizontal lines for 10, 5, 1 mmag
                    plt.hlines([0.001, 0.002, 0.003, 0.004, 0.005],
                               xmin=rangex[0],xmax=rangex[1],colors='b')

                elif observatory=='tess' and yaxisval=='RMS':
                    plt.ylim((0.00009,1.0))

                    # overplot tess noise model
                    Tmag = np.linspace(6, 13, num=200)
                    lnA = 3.29685004771
                    B = 0.8500214657
                    C = -0.2850416324
                    D = 0.039590832137
                    E = -0.00223080159
                    F = 4.73508403525e-5
                    ln_sigma_1hr = (
                        lnA + B*Tmag + C*Tmag**2 + D*Tmag**3 +
                        E*Tmag**4 + F*Tmag**5
                    )
                    sigma_1hr = np.exp(ln_sigma_1hr)
                    sigma_30min = sigma_1hr * np.sqrt(2)

                    plt.plot(Tmag, sigma_30min/1e6, 'k-', zorder=3, lw=2,
                             label='toy model $\sigma_{\mathrm{30\,min}}$')
                    plt.legend(loc='best', fontsize='xx-small')

            # put the grid on the plot
            plt.gca().grid(color='#a9a9a9',
                           alpha=0.9,
                           zorder=0,
                           linewidth=1.0,
                           linestyle=':')

            plt.savefig(outfile)
            plt.close('all')

            print('%sZ: made %s plot: %s' %
                  (datetime.utcnow().isoformat(), title, outfile))

        except Exception as e:

            print('%sZ: plot failed! Error was: %s' %
                  (datetime.utcnow().isoformat(), e))
            raise


def plot_magrms_comparison(reference_stats_file,
                           comparison_stats_file,
                           ref_name, comp_name,
                           outfile,
                           ref_col='mad_tf3',
                           comp_col='mad_tf3',
                           logy=False, logx=False,
                           rangex=(5.9,14.1)):
    """
    This makes magrms comparison plots using two lightcurve stats files.
    """

    ref_stats = read_stats_file(reference_stats_file)
    comp_stats = read_stats_file(comparison_stats_file)

    ref_objects = ref_stats['lcobj']
    comp_objects = comp_stats['lcobj']

    common_objects = np.intersect1d(ref_objects, comp_objects)

    print('common objects = %s' % len(common_objects))

    if len(common_objects) > 0:

        # put together the data for the common objects
        ref_mag = np.ravel([ref_stats['cat_mag'][ref_stats['lcobj'] == x]
                            for x in common_objects])
        comp_mag = np.ravel([comp_stats['cat_mag'][comp_stats['lcobj'] == x]
                             for x in common_objects])

        if ref_col == 'mad_tfbestap':

            ref_tf1_compcol = np.ravel(
                [ref_stats['mad_tf1'][ref_stats['lcobj'] == x]
                 for x in common_objects]
            )
            ref_tf2_compcol = np.ravel(
                [ref_stats['mad_tf2'][ref_stats['lcobj'] == x]
                 for x in common_objects]
            )
            ref_tf3_compcol = np.ravel(
                [ref_stats['mad_tf3'][ref_stats['lcobj'] == x]
                 for x in common_objects]
            )

            ref_tf_compcolstack = np.column_stack((ref_tf1_compcol,
                                                   ref_tf2_compcol,
                                                   ref_tf3_compcol))

            # get the min of each row of the stack, this is the best MAD
            ref_compcol = np.amin(ref_tf_compcolstack,axis=1)

        else:

            ref_compcol = np.ravel([ref_stats[ref_col][ref_stats['lcobj'] == x]
                                    for x in common_objects])

        if comp_col == 'mad_tfbestap':

            comp_tf1_compcol = np.ravel(
                [comp_stats['mad_tf1'][comp_stats['lcobj'] == x]
                 for x in common_objects]
            )
            comp_tf2_compcol = np.ravel(
                [comp_stats['mad_tf2'][comp_stats['lcobj'] == x]
                 for x in common_objects]
            )
            comp_tf3_compcol = np.ravel(
                [comp_stats['mad_tf3'][comp_stats['lcobj'] == x]
                 for x in common_objects]
            )

            comp_tf_compcolstack = np.column_stack((comp_tf1_compcol,
                                                    comp_tf2_compcol,
                                                    comp_tf3_compcol))

            # get the min of each row of the stack, this is the best MAD
            comp_compcol = np.amin(comp_tf_compcolstack,axis=1)

        else:

            comp_compcol = np.ravel(
                [comp_stats[comp_col][comp_stats['lcobj'] == x]
                 for x in common_objects]
            )

        # get the ratios
        compcol_ratios = (
            np.array(ref_compcol)/np.array(comp_compcol)
            )

        nonzero_ind = np.where(compcol_ratios > 0.0)
        xcol = (np.array(ref_mag))[nonzero_ind[0]]
        ycol = compcol_ratios[nonzero_ind[0]]

        xlabel = 'catalog SDSS r mag'
        ylabel = 'TFA MAD %s/%s' % (ref_name, comp_name)

        title = 'TFA MAD ratio - %s/%s' % (ref_name, comp_name)

        # make the plot
        if logy:
            plt.scatter(xcol, ycol,
                        s=1,
                        marker='.')
            plt.yscale('log')
        elif logx:
            plt.scatter(xcol, ycol,
                        s=1,
                        marker='.')
            plt.xscale('log')
        elif logx and logy:
            plt.scatter(xcol, ycol,
                        s=1,
                        marker='.')
            plt.xscale('log')
            plt.yscale('log')
        else:
            plt.scatter(xcol, ycol,
                        s=1,
                        marker='.')


        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(rangex)
        plt.ylim(-0.5,4)

        # make the horizontal lines for 10, 5, 1 mmag
        plt.hlines([-2.0,-1.5,0.5,1,1.5,2.0],
                   xmin=rangex[0],xmax=rangex[1],colors='r')

        # put the grid on the plot
        plt.gca().grid(color='#a9a9a9',
                       alpha=0.9,
                       zorder=0,
                       linewidth=1.0,
                       linestyle=':')

        plt.savefig(outfile)
        plt.close()

    else:

        print('no common objects to use for comparison!')




def rollup_plots(projid,
                 projdir,
                 ccdlist=[5,6,7,8],
                 phottype='ism'):
    """
    This makes the mag-RMS plots and the comparison plots for all of the ccds.
    """


    # make the rms plots
    for ccd in ccdlist:

        statfile = os.path.join(projdir,'ccd%s-tfa-lcstats.txt' % ccd)

        if os.path.exists(statfile):
            rmsplotprefix = 'projid{projid}-ccd{ccd}-{phottype}'.format(
                projid=projid,
                ccd=ccd,
                phottype=phottype
            )
            plot_stats_file(statfile,'.',rmsplotprefix,logy=True)
        else:
            print('statsfile for CCD%s not present, skipping...' % ccd)

    # make the comparison plots
    for ccd in ccdlist:

        thisindex = ccdlist.index(ccd)
        otherccds = ccdlist[thisindex+1:]

        thisstatfile = os.path.join(projdir,'ccd%s-tfa-lcstats.txt' % ccd)

        if os.path.exists(thisstatfile):

            for otherccd in otherccds:

                otherstatfile = os.path.join(projdir,
                                             'ccd%s-tfa-lcstats.txt' %
                                             otherccd)

                if os.path.exists(otherstatfile):

                    print(
                        'comparison plot between ccd %s and ccd %s' % (ccd,
                                                                       otherccd)
                    )

                    tf3outfile = (
                        'projid%s-ccd%s-ccd%s-comparison.png' % (projid,
                                                                 ccd,
                                                                 otherccd)
                    )
                    tf3reftitle = 'projid%s-ccd%s-tf3' % (projid, ccd)
                    tf3comptitle = 'projid%s-ccd%s-tf3' % (projid, otherccd)

                    bestapoutfile = (
                        'projid%s-ccd%s-ccd%s-bestap-comparison.png' % (
                            projid,
                            ccd,
                            otherccd
                        )
                    )
                    bestapreftitle = 'projid%s-ccd%s-bestap' % (projid, ccd)
                    bestapcomptitle = 'projid%s-ccd%s-bestap' % (projid,
                                                                 otherccd)

                    # TF3 plot first
                    plot_magrms_comparison(thisstatfile,
                                           otherstatfile,
                                           tf3reftitle,
                                           tf3comptitle,
                                           tf3outfile,
                                           ref_col='mad_tf3',comp_col='mad_tf3')
                    # bestap plot next
                    plot_magrms_comparison(thisstatfile,
                                           otherstatfile,
                                           bestapreftitle,
                                           bestapcomptitle,
                                           bestapoutfile,
                                           ref_col='mad_tfbestap',
                                           comp_col='mad_tfbestap')
