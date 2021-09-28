#!/usr/bin/env python

"""
autoimagesub.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 09/2016

This contains functions to run image subtraction photometry continuously.
Namely,

Image & Database management:

    fitslist_frameinfo: parallelizes get_frame_info
        get_frame_info: gets info from frame for selecting photref candidates.

    parallel_frames_to_database
        calibrated_frame_to_database
        calframe_to_db_worker
        arefshifted_frame_to_database
        arefshifted_frame_to_db_worker
        dbupdate_calibratedframe
        convsubtracted_frame_to_database

Astrometric references:

    dbgen_get_astromref
    generate_astromref
    dbget_astromref: from postgresql database
    get_astromref: from sqlite database
    frames_astromref_worker
    framelist_make_xtrnsfits

Photometric references:

    generate_photref_candidates_from_xtrns
    amend_candidate_photrefs
    generate_combined_photref
    get_combined_photref
    insert_phots_into_database

Image subtraction:

    xtrnsfits_convsub_worker
    parallel_xtrnsfits_convsub
    convsubfits_staticphot_worker: photometry on subtracted frames
    parallel_convsubfits_staticphot
"""

#############
## IMPORTS ##
#############

import os
import os.path
import glob
import multiprocessing as mp

try:
    import subprocess32 as subprocess
    from subprocess32 import check_output
except:
    import subprocess
    from subprocess import check_output

import shlex
from datetime import datetime
import re
import json
import shutil
import random
from functools import partial

try:
    import cPickle as pickle
except:
    import pickle

import sqlite3
import time
from hashlib import md5, sha256
import gzip
from traceback import format_exc
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import tempfile

import numpy as np, pandas as pd
from numpy import nan
import psycopg2 as pg
from psycopg2.extras import Json

from astropy import wcs

try:
    from framecalib import make_frame_movie
except ModuleNotFoundError:
    print('WRN! not importing framecalib; was not found.')
import aperturephot as ap
import imagesubphot as ism
from imageutils import get_header_keyword, get_header_keyword_list, \
    fits_to_full_jpeg, check_frame_warping, frame_radecbox_to_jpeg, \
    fitscoords_to_jpeg

import shared_variables as sv

# get fiphot binary reader
try:
    from HATpipepy.Common.BinPhot import read_fiphot
    HAVEBINPHOT = True
except:
    print("can't import binary fiphot reading functions from "
          "HATpipe, binary fiphot files will be unreadable!")
    HAVEBINPHOT = False


############
## CONFIG ##
############

DEBUG = True

# used to get the station ID, frame number, and CCD number from a FITS filename
FRAMEREGEX = re.compile(r'(\d{1})\-(\d{6}\w{0,1})_(\d{1})')
# used to get the station ID, frame number, subframe, and CCD number
FRAMESUBREGEX = re.compile(r'(\d{1})\-(\d{6})(\w{0,1})_(\d{1})')
# this defines the field string and CCDs
FIELD_REGEX = re.compile('^G(\d{2})(\d{2})([\+\-]\d{2})(\d{2})_(\w{3})$')
FIELD_CCDS = [5,6,7,8]
# this is to recognize a HATID
HATIDREGEX = re.compile(r'^HAT\-\d{3}\-\d{7}$')

# defines where the reference frames go
REFBASEDIR = sv.REFBASEDIR
REFINFO = sv.REFINFO
# define where the frameinfo cache is
FRAMEINFOCACHEDIR = sv.FRAMEINFOCACHEDIR
# these define the field catalog location and properties
FIELDCAT_DIR = sv.FIELDCAT_DIR
# these define the light curve directory
LCBASEDIR = sv.LCPATH

# these define the postgres database credentials
PGPASSFILE = sv.PGPASSFILE
PGUSER = sv.PGUSER
PGDATABASE = sv.PGDATABASE
PGHOST = sv.PGHOST

if os.path.exists(PGPASSFILE):
    with open(PGPASSFILE) as infd:
        pgpass_contents = infd.readlines()
        pgpass_contents = [x.split(':') for x in pgpass_contents if not
                           x.startswith('#')]
        PGPASSWORD = [x[-1] for x in pgpass_contents
                      if (x[0] == PGHOST and
                          x[2] == PGDATABASE and
                          x[3] == PGUSER)]
        PGPASSWORD = PGPASSWORD[0].strip('\n')
else:
    print('ERR! %sZ: could not find postgres database credientials at %s' %
          (datetime.utcnow().isoformat(), PGPASSFILE))


###############
## UTILITIES ##
###############

def smartcast(castee, caster, subval=None):
    """
    This just tries to apply the caster function to castee.

    Returns None on failure.
    """

    try:
        return caster(castee)
    except Exception as e:
        if caster is float or caster is int:
            return nan
        elif caster is str:
            return ''
        else:
            return subval



def fits_fieldprojectidccd_worker(frame):
    """
    This is a worker for the two functions below.
    """

    try:

        # first, figure out the input frame's projid, field, and ccd
        frameelems = get_header_keyword_list(frame,
                                             ['object',
                                              'projid'])
        felems = FRAMEREGEX.findall(
            os.path.basename(frame)
        )
        field, ccd, projectid = (frameelems['object'],
                                 felems[0][2],
                                 frameelems['projid'])

        return frame, (field, projectid, int(ccd))

    except Exception as e:

        print('ERR! %sZ: could get info from frame %s, error was: %s' %
              (datetime.utcnow().isoformat(), frame, e))
        return frame, None



def find_original_fits_in_database(field, projectid, ccd,
                                   enforceok=True,
                                   database=None):
    """
    This finds the FITS matching the field-projectid-ccd combo in the DB.
    """




def find_original_fits_fieldprojectidccd(dirlist,
                                         field,
                                         projectid,
                                         ccd,
                                         fglob='?-???????_?.fits',
                                         nworkers=8,
                                         maxworkertasks=1000):
    """This searches in dirlist for all original FITS files matching the specified
    projectid, field, and ccd combination.

    Returns a flat list of matching FITS, and list of all fits + their info.
    """

    # first, go through the directories and get all the original FITS files
    print('%sZ: finding frames matching %s...' %
          (datetime.utcnow().isoformat(), fglob))
    fitslist = []
    for fdir in dirlist:
        fitslist.extend(glob.glob(os.path.join(fdir, fglob)))
    fitslist = sorted(fitslist)

    # next, run through all these files and get the info needed
    print('%sZ: %s frames found, getting info...' %
          (datetime.utcnow().isoformat(), len(fitslist)))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = fitslist

    # fire up the pool of workers
    results = pool.map(fits_fieldprojectidccd_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    # now filter the results based on the requested field, projectid, and ccd
    matchingframes = []

    for elem in results:

        if (elem[1] and
            elem[1][0] == field and
            elem[1][1] == projectid and
            elem[1][2] == ccd):
            matchingframes.append(elem[0])

    print('%sZ: %s frames with field = %s, projectid = %s, and ccd = %s' %
          (datetime.utcnow().isoformat(),
           len(matchingframes),
           field, projectid, ccd))

    return matchingframes, results



def find_arefshifted_fits_fieldprojectidccd(dirlist,
                                            field,
                                            projectid,
                                            ccd,
                                            fglob='?-???????_?-xtrns.fits',
                                            nworkers=8,
                                            maxworkertasks=1000):
    """This searches for all astromref-shifted FITS files matching the
    specified projectid, field, and ccd combination.

    Returns a flat list of matching FITS, and list of all fits + their info.
    """

    return find_original_fits_fieldprojectidccd(dirlist,
                                                field,
                                                projectid,
                                                ccd,
                                                fglob=fglob,
                                                nworkers=nworkers,
                                                maxworkertasks=maxworkertasks)



def find_subtracted_fits_fieldprojectidccd(
        dirlist,
        field,
        projectid,
        ccd,
        subtracttype='reverse',
        photreftype='oneframe',
        kernelspec='b/4;i/4;d=4/4',
        nworkers=8,
        maxworkertasks=1000):
    """This searches for all subtracted FITS files matching the specified
    projectid, field, and ccd combination.

    Returns a flat list of matching FITS, and list of all fits + their info.
    """

    photrefbit = (
        'rsub' if subtracttype == 'reverse' else 'nsub'
    )
    # generate the convsubfits hash
    convsubhash = ism.get_convsubfits_hash(
        photreftype,
        subtracttype,
        kernelspec
    )


    fglob= '%s-%s-?-???????_?-xtrns.fits' % (photrefbit, convsubhash)

    return find_original_fits_fieldprojectidccd(dirlist,
                                                field,
                                                projectid,
                                                ccd,
                                                fglob=fglob,
                                                nworkers=nworkers,
                                                maxworkertasks=maxworkertasks)

########################
## GETTING FRAME INFO ##
########################

def get_frame_info(frame):
    """
    This gets the needed info from a frame for selecting photref candidates.
    """

    try:

        # find the frame's fistar and fiphot files
        fitsdir = os.path.dirname(frame)
        fitsbase = os.path.splitext(os.path.basename(frame))[0]

        # handle gz/fz files
        if '.fits' in fitsbase:
            fitsbase = fitsbase.replace('.fits','')

        # if the xtrns files are passed in, make sure we look at the
        # right fistar and fiphot files
        if '-xtrns' in fitsbase:
            fitsbase = fitsbase.rstrip('-xtrns')

        photpath = os.path.join(
            fitsdir,
            fitsbase + '.fiphot'
            )
        srclistpath = os.path.join(
            fitsdir,
            fitsbase + '.fistar'
            )

        if not (os.path.exists(frame) and
                os.path.exists(photpath) and
                os.path.exists(srclistpath)):

            print('ERR! %sZ: frame %s has missing fiphot/fistar, '
                  'so no phot info...' %
                  (datetime.utcnow().isoformat(), frame))
            return frame, None



        # 1. get the data from FITS header
        headerdata = get_header_keyword_list(
            frame,
            ['Z','MOONDIST','MOONELEV','MOONPH','HA']
        )

        # 2. get the data from the fiphot file

        # decide if the phot file is binary or not. read the first 600
        # bytes and look for the '--binary-output' text
        with open(photpath,'rb') as photf:
            header = photf.read(1000)

        if '--binary-output' in header.decode('utf-8') and HAVEBINPHOT:

            photdata_f = read_fiphot(photpath)
            photdata = {
                'mag':np.array(photdata_f['per aperture'][2]['mag']),
                'err':np.array(photdata_f['per aperture'][2]['mag err']),
                'flag':np.array(
                    photdata_f['per aperture'][2]['status flag']
                    )
                }
            del photdata_f

        elif '--binary-output' in header.decode('utf-8') and not HAVEBINPHOT:

            print('WRN! %sZ: %s is a binary phot file, '
                  'but no binary phot reader is present, skipping...' %
                  (datetime.utcnow().isoformat(), photpath))
            return frame, None

        else:

            # read in the phot file
            photdata = np.genfromtxt(
                photpath,
                usecols=(12,13,14),
                dtype='f8,f8,U5',
                names=['mag','err','flag']
                )
            photdata = np.atleast_1d(photdata)

        # 3. get the data from the fistar file
        srcdata = np.genfromtxt(srclistpath,
                                usecols=(3,5,6),
                                dtype='f8,f8,f8',
                                names=['background',
                                       'svalue',
                                       'dvalue'])
        srcdata = np.atleast_1d(srcdata)

        # process fiphot data
        if '--binary-output' in header.decode('utf-8'):
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


        # now consolidate all the data
        frameinfo = {
            'zenithdist':(headerdata['Z']
                          if 'Z' in headerdata else np.nan),
            'moondist':(headerdata['MOONDIST']
                        if 'MOONDIST' in headerdata else np.nan),
            'moonelev':(headerdata['MOONELEV']
                        if 'MOONELEV' in headerdata else np.nan),
            'moonphase':(headerdata['MOONPH']
                         if 'MOONPH' in headerdata else np.nan),
            'hourangle':(headerdata['HA']
                         if 'HA' in headerdata else np.nan),
            'ngoodobjects':ngood,
            'medmagerr':median_magerr,
            'magerrmad':medabsdev_mag,
            'medsrcbgv':np.nanmedian(srcdata['background']),
            'stdsrcbgv':np.nanstd(srcdata['background']),
            'medsval':np.nanmedian(srcdata['svalue']),
            'meddval':np.nanmedian(srcdata['dvalue']),
        }

        return frame, frameinfo

    except Exception as e:

        print('ERR! %sZ: could not get info from frame %s, error was: %s' %
              (datetime.utcnow().isoformat(), frame, e))
        return frame, None



def fitslist_frameinfo(fitslist,
                       forcecollectinfo=False,
                       nworkers=8,
                       maxworkertasks=1000):
    """
    This runs a parallel get_frame_info job.
    """

    # check if we have it in the frameinfo cache, if so, use it. if not, redo
    # the info collection, and then write it back to the cache.
    cachefile = os.path.join(FRAMEINFOCACHEDIR,
                             ('TM-frameinfo-%s.pkl.gz' %
                              md5(repr(fitslist).encode('utf-8')).hexdigest()))

    if os.path.exists(cachefile) and not forcecollectinfo:

        with gzip.open(cachefile,'rb') as infd:
            frameinfo = pickle.load(infd)

        print('%sZ: frameinfo found in cache file: %s' %
              (datetime.utcnow().isoformat(), cachefile))

        return frameinfo

    # if the cache doesn't exist, we'll run the frameinfo procedure and write
    # the results back to the cache
    else:

        print('%sZ: getting frameinfo for %s frames...' %
              (datetime.utcnow().isoformat(), len(fitslist)))

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        tasks = fitslist

        # fire up the pool of workers
        results = pool.map(get_frame_info, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        # now turn everything into ndarrays
        frames = np.array([x[0] for x in results])
        zenithdist = np.array([(x[1]['zenithdist']
                                if x[1] else np.nan) for x in results])
        moondist = np.array([(x[1]['moondist']
                                if x[1] else np.nan) for x in results])
        moonelev = np.array([(x[1]['moonelev']
                                if x[1] else np.nan) for x in results])
        moonphase = np.array([(x[1]['moonphase']
                                if x[1] else np.nan) for x in results])
        hourangle = np.array([(x[1]['hourangle']
                                if x[1] else np.nan) for x in results])
        ngoodobjects = np.array([(x[1]['ngoodobjects']
                                if x[1] else np.nan) for x in results])
        medmagerr = np.array([(x[1]['medmagerr']
                                if x[1] else np.nan) for x in results])
        magerrmad = np.array([(x[1]['magerrmad']
                                if x[1] else np.nan) for x in results])
        medsrcbgv = np.array([(x[1]['medsrcbgv']
                                if x[1] else np.nan) for x in results])
        stdsrcbgv = np.array([(x[1]['stdsrcbgv']
                                if x[1] else np.nan) for x in results])
        medsval = np.array([(x[1]['medsval']
                                if x[1] else np.nan) for x in results])
        meddval = np.array([(x[1]['meddval']
                                if x[1] else np.nan) for x in results])


        outdict = {'frames':frames,
                   'zenithdist':zenithdist,
                   'moondist':moondist,
                   'moonelev':moonelev,
                   'moonphase':moonphase,
                   'hourangle':hourangle,
                   'ngoodobjects':ngoodobjects,
                   'medmagerr':medmagerr,
                   'magerrmad':magerrmad,
                   'medsrcbgv':medsrcbgv,
                   'stdsrcbgv':stdsrcbgv,
                   'medsval':medsval,
                   'meddval':meddval}

        with gzip.open(cachefile,'wb') as outfd:
            pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)

        print('%sZ: wrote frameinfo to cache file: %s' %
              (datetime.utcnow().isoformat(), cachefile))

        return outdict



######################################
## PUTTING FRAMES INTO THE DATABASE ##
######################################

def calibrated_frame_to_database(fitsfile,
                                 observatory='hatpi',
                                 overwrite=False,
                                 badframetag='badframes',
                                 nonwcsframes_are_ok=False,
                                 custom_projid=False,
                                 database=None):
    """
    This puts a fully calibrated FITS into the database.

    Requires that the frame have gone through the full
    cron_transfer_calibrate.autocal_fitsdir function process, so that it has an
    extracted source list (fistar), astrometry attempted on it (wcs), and basic
    aperture photometry carried out (fiphot).

    Searches the same directory as the FITS for fistar, fiphot, and wcs
    files. Should be used with a "find {fitsdir} -type f -name '?-*_?.fits'"
    command to generate the list of the FITS files, so that it can work on bad
    frames that have been moved to the {badframedir} subdirectory of {fitsdir}.

    Args:

        fitsfile (str): path to fits file

    Kwargs:

        observatory (str): hatpi or tess

        overwrite (bool): whether to overwrite sql rows

        badframetag (str): string for subdirectory where the badframes are

        nonwcsframes_are_ok (bool): if False, then frames without wcs will be
        marked as badframes.  If a frame doesn't have an accompanying wcs or
        fiphot, it will be tagged with frameisok = false in the database as
        well.

        custom_projid (str): Custom project ID

        database: if passing pg.connect() instance
    """

    # open database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        cursor = database.cursor()
        closedb = True

    try:

        # get paths to frame's fistar, wcs, and fiphot files
        fits = os.path.abspath(fitsfile)
        prospective_fistar = fits.replace('.fits','.fistar')
        prospective_fiphot = fits.replace('.fits','.fiphot')
        prospective_wcs = fits.replace('.fits','.wcs')

        if observatory == 'hatpi':
            header_list = ['PROJID','OBJECT','JD','EXPTIME',
                           'STVER','MTID','MTVER',
                           'CMID','CMVER','TELID','TELVER','FOV',
                           'BIASVER','DARKVER','FLATVER','MNTSTATE','PTVER',
                           'FILID','IMAGETYP','RA','DEC',
                           'MOONDIST','MOONELEV','MOONPH',
                           'HA','Z','MGENSTAT','WIND','HUMIDITY','SKYTDIFF','AMBTEMP',
                           'DEWPT','FOCUS', 'COMMENT']

        elif observatory == 'tess':
            header_list = ['PROJID', 'CAMERA', 'CCD', 'TSTART', 'TSTOP',
                           'CRVAL1', 'CRVAL2', 'RA_NOM', 'DEC_NOM', 'ROLL_NOM',
                           'DQUALITY', 'CCDTEMP']

        else:
            raise ValueError('observatory must be tess or hatpi')

        # get header keywords from header_list
        headerdata = get_header_keyword_list(fitsfile, header_list)

        # get the frame's photometry info (useful for selecting refs)
        photfits, photinfo = get_frame_info(fitsfile)

        # check if frames are bad
        if nonwcsframes_are_ok:
            badframecheck = (
                os.path.exists(prospective_fistar) and
                os.path.exists(prospective_fiphot) and
                (badframetag not in os.path.abspath(fitsfile))
            )

        else:
            badframecheck = (
                os.path.exists(prospective_fistar) and
                os.path.exists(prospective_wcs) and
                os.path.exists(prospective_fiphot) and
                (badframetag not in os.path.abspath(fitsfile))
            )

        # if this is a bad frame set stuff accordingly
        if badframecheck:
            frameisok = True
            fits = os.path.abspath(fitsfile)
            fistar = os.path.abspath(prospective_fistar)
            if os.path.exists(prospective_wcs):
                wcs = os.path.abspath(prospective_wcs)
            else:
                wcs = None
            fiphot = os.path.abspath(prospective_fiphot)

        else:
            frameisok = False
            fits = os.path.abspath(fitsfile)
            if os.path.exists(prospective_fistar):
                fistar = os.path.abspath(prospective_fistar)
            else:
                fistar = None
            if os.path.exists(prospective_wcs):
                wcs = os.path.abspath(prospective_wcs)
            else:
                wcs = None
            if os.path.exists(prospective_fiphot):
                fiphot = os.path.abspath(prospective_fiphot)
            else:
                fiphot = None

        # Convert fits COMMENT (many lines!) to a dictionary that merges all the
        # comments. The mega-comment string will be
        # formatted like this:
        # "'Weather.WTextern=0'|'Weather.WTweb=0'|'Weather.WTparan=3'|"
        # For HATPI, these COMMENTS include info on camera groups, dome status,
        # mount status, filter, observe info, station IDs, weather, bias, flats
        # and darks applied.

        fitsheader = headerdata

        if observatory=='hatpi':

            # replace whitespace by underscores because json binary does not
            # respect whitespace.
            bigstr = re.sub('\\n=\ ','|',
                            repr(headerdata['COMMENT'])).replace(' ','_')

            # make the dict that parses the comment lines
            # keep only the comment lines in the format "*=*"
            commentdict = {}
            for commentline in bigstr.split("'|'"):
                if len(commentline.split('='))==2:
                    commentdict[commentline.split('=')[0]] = commentline.split('=')[1]

            fitsheader['COMMENT'] = commentdict

        fitsheader = {(k if not pd.isnull(v) else k):
                      (v if not pd.isnull(v) else 'NaN')
                      for k,v in fitsheader.items()}

        if custom_projid is not None and isinstance(custom_projid,int):

            # if forced, overwrite PROJID from the fits header with whatever is
            # passed. this is relevant for PSQL bookkeeping situations in which
            # you are using the same calibrated frames across different
            # projids (to save disk space).

            fitsheader['PROJID'] = custom_projid

        fitsheaderjson = Json(fitsheader)

        if photinfo:
            photinfo = {(k if not pd.isnull(v) else k):
                        (v if not pd.isnull(v) else 'NaN')
                         for k,v in photinfo.items()}
            photinfojson = Json(photinfo)
        else:
            photinfojson = None

        if overwrite:
            query = ("insert into calibratedframes ("
                     "fits, fistar, fiphot, wcs, fitsheader, photinfo, "
                     "frameisok) values ("
                     "%s, %s, %s, %s, %s, %s, "
                     "%s) on conflict "
                     "(fits) do update "
                     "set fistar = %s, fiphot = %s, wcs = %s, "
                     "fitsheader = %s, photinfo = %s, frameisok = %s, "
                     "entryts = current_timestamp")
            params = (fits, fistar, fiphot, wcs, fitsheaderjson, photinfojson,
                      frameisok, fistar, fiphot, wcs,
                      fitsheaderjson, photinfojson, frameisok)
        else:
            # one row per fits image. if the fits image already has a row, do
            # nothing.
            query = ("insert into calibratedframes ("
                     "fits, fistar, fiphot, wcs, fitsheader, photinfo, "
                     "frameisok) values ("
                     "%s, %s, %s, %s, %s, %s, "
                     "%s) on conflict (fits) do nothing" )
            params = (fits, fistar, fiphot, wcs, fitsheaderjson, photinfojson,
                     frameisok)

        if DEBUG:
            print('query: {:s}\nparams: {:s}'.format(repr(query),repr(params)))

        # execute the query and commit
        cursor.execute(query, params)
        database.commit()

        message = (
            'inserted {:s} into calibratedframes (or not if conflict) -- OK'.
            format(fits)
        )
        print('%sZ: %s' %
              (datetime.utcnow().isoformat(), message) )
        returnval = (fits, True)

    # catch the overwrite = False scenario
    except pg.IntegrityError as e:

        database.rollback()

        message = ('failed to insert %s '
                   'into DB because it exists already '
                   'and overwrite = False'
                   % fitsfile)
        print('EXC! %sZ: %s\n%s' %
               (datetime.utcnow().isoformat(), message, format_exc()) )
        returnval = (fitsfile, False)


    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to insert %s into DB' % fitsfile
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (fitsfile, False)

    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def calframe_to_db_worker(task):
    """
    This wraps calibrated_frames_to_database for the parallel driver below.
    """

    fitsfile, kwargs = task
    return calibrated_frame_to_database(fitsfile, **kwargs)



def arefshifted_frame_to_database(
        fitsfile,
        overwrite=False,
        badframetag='badframes',
        nonwcsframes_are_ok=False,
        custom_projid=None,
        database=None):
    """
    This puts a shifted-to-astromref xtrns FITS into the DB.

    fitsfile: of the shifted image, for example:

        /nfs/phtess1/ar1/HATPI/HP0/RED/projid12-G577-ccd8-sdssr/1-439340f_8-xtrns.fits

    Associates it with the framekey of the original FITS. Adds info on shift
    success, (optionally) the shifted-frame's x and y direction gradients
    calculated, and its full path. Adds info about the itrans file used as
    well.

    DONE:
        * given a fitsfile, will put it into arefshiftedframes

    TODO:
        * include "warp check" flagging
        * include "badframe" tagging

    """

    if nonwcsframes_are_ok:
        raise NotImplementedError

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    # get the values: arefshiftedframe, astomref, origframekey, itrans,
    # shiftisok, didwarpcheck
    arefshiftedframe = fitsfile

    frameelems = get_header_keyword_list(arefshiftedframe, ['object',
                                                            'projid'])

    felems = FRAMEREGEX.findall(os.path.basename(arefshiftedframe))

    if felems and felems[0]:

        ccd = felems[0][2]
        frameinfo = {'field':frameelems['object'],
                     'ccd':ccd,
                     'projectid':frameelems['projid']}

        if custom_projid is not None:
            frameinfo['projectid'] = custom_projid

        # find this frame's associated active astromref
        astromref = dbget_astromref(frameinfo['projectid'], frameinfo['field'],
                                    frameinfo['ccd'])

        #  areffistar = astromref['framepath'].replace('.fits', '.fistar')

    # get frame key for the original (calibrated) fits frame:
    originalfits = re.sub('-xtrns.fits', '.fits', arefshiftedframe)
    query = "select framekey from calibratedframes where fits = '{:s}'".format(
            originalfits)

    cursor.execute(query)
    row = cursor.fetchone()
    if row is None:
        print(query)

    origframekey = row[0] # keeps long type

    if cursor.fetchone():
        raise Exception('there should be only a single entry for each unique fits')

    itrans = re.sub('-xtrns.fits', '.itrans', fitsfile)
    shiftisok = True if os.path.exists(fitsfile) else False
    didwarpcheck = False # NOTE: may need to change if warp check is necessary

    if didwarpcheck:
        raise NotImplementedError
        warpcheckmargin = _
        warpcheckthresh = _
        warpinfopickle = _

    # start work here
    try:

        # put together the query and execute it, inserting the object into the
        # database and overwriting if told to do so
        if overwrite:
            if didwarpcheck:
                query = ("insert into arefshiftedframes ("
                         "arefshiftedframe, "
                         "origframekey, "
                         "astromref, "
                         "itrans, "
                         "shiftisok, "
                         "didwarpcheck, "
                         "warpcheckmargin, "
                         "warpcheckthresh, "
                         "warpinfopickle "
                         ") values ("
                         "%s, %s, %s, %s, %s, "
                         "%s, %s, %s, %s "
                         ") on conflict ( "
                         " arefshiftedframe, astromref "
                         ") do update set "
                         "origframekey = %s, "
                         "astromref = %s, "
                         "shiftisok = %s, didwarpcheck = %s, "
                         "warpcheckmargin = %s, warpcheckthresh = %s, "
                         "warpinfopickle = %s, "
                         "entryts = current_timestamp")
                params = (arefshiftedframe, origframekey,
                          astromref['framepath'],
                          itrans, shiftisok,
                          didwarpcheck,
                          warpcheckmargin, warpcheckthresh,
                          warpinfopickle,
                          origframekey, astromref['framepath'],
                          shiftisok, didwarpcheck,
                          warpcheckmargin, warpcheckthresh,
                          warpinfopickle
                         )

            else:
                query = ("insert into arefshiftedframes ("
                         "arefshiftedframe, "
                         "origframekey, "
                         "astromref, "
                         "itrans, "
                         "shiftisok, "
                         "didwarpcheck "
                         ") values ("
                         "%s, %s, %s, %s, %s, "
                         "%s "
                         ") on conflict ("
                         " arefshiftedframe, astromref "
                         ") do update set "
                         "origframekey = %s, "
                         "itrans = %s, "
                         "shiftisok = %s, didwarpcheck = %s, "
                         "entryts = current_timestamp"
                         )
                params = (arefshiftedframe,
                          origframekey,
                          astromref['framepath'],
                          itrans,
                          shiftisok,
                          didwarpcheck,
                          origframekey, itrans,
                          shiftisok, didwarpcheck
                         )


        else:
            if didwarpcheck:
                query = ("insert into arefshiftedframes ("
                         "arefshiftedframe, "
                         "origframekey, "
                         "astromref, "
                         "itrans, "
                         "shiftisok, "
                         "didwarpcheck, "
                         "warpcheckmargin, "
                         "warpcheckthresh, "
                         "warpinfopickle "
                         ") values ("
                         "%s, %s, %s, %s, %s, "
                         "%s, %s, %s, %s "
                         ") ")
                params = (arefshiftedframe,
                          origframekey,
                          astromref['framepath'],
                          itrans,
                          shiftisok,
                          didwarpcheck,
                          warpcheckmargin,
                          warpcheckthresh,
                          warpinfopickle
                         )

            else:
                query = ("insert into arefshiftedframes ("
                         "arefshiftedframe, "
                         "origframekey, "
                         "astromref, "
                         "itrans, "
                         "shiftisok, "
                         "didwarpcheck "
                         ") values ("
                         "%s, %s, %s, %s, %s, "
                         "%s "
                         ") ")
                params = (arefshiftedframe,
                          origframekey,
                          astromref['framepath'],
                          itrans,
                          shiftisok,
                          didwarpcheck
                         )

        # execute the query and commit
        cursor.execute(query, params)
        database.commit()

        message = 'inserted %s into DB' % arefshiftedframe
        print('%sZ: %s' %
              (datetime.utcnow().isoformat(), message) )

        returnval = (fitsfile, True)

    # catch the overwrite = False scenario
    except pg.IntegrityError as e:

        database.rollback()

        message = ('failed to insert %s '
                   'into DB because it exists already '
                   'and overwrite = False'
                   % fitsfile)
        print('EXC! %sZ: %s\n%s' %
               (datetime.utcnow().isoformat(), message, format_exc()) )
        returnval = (fitsfile, False)


    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to insert %s into DB' % fitsfile
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (fitsfile, False)

    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def arefshifted_frame_to_db_worker(task):
    """
    This wraps arefshifted_frame_to_database for the parallel driver below.
    """

    fitsfile, kwargs = task
    return arefshifted_frame_to_database(fitsfile, **kwargs)



def parallel_frames_to_database(fitsbasedir,
                                frametype,
                                observatory='hatpi',
                                fitsglob='1-???????_?.fits',
                                overwrite=False,
                                badframetag='badframes',
                                nonwcsframes_are_ok=False,
                                nworkers=16,
                                maxworkertasks=1000,
                                custom_projid=None):
    """
    This runs a DB ingest on all FITS located in fitsbasedir and subdirs.  Runs
    a 'find' subprocess to find all the FITS to process.  If the frames have
    already been injested, based on their fits file paths they will not be
    changed, unless overwrite=True.

    Args:
        fitsbasedir: base directory with fits files. For example,
            `/nfs/phtess1/ar1/HATPI/HP0/RED/projid12-G577-ccd8-sdssr/`

        frametype: 'calibratedframes' and 'arefshifted_frames' are the currently
            implemented options, corresponding to putting calibrated frames,
            and astrometrically shifted frames, into the database.

        fitsglob: if arefshifted frames, it's actually '1-???????_?-xtrns.fits'

        custom_projid: specify a custom project ID (useful for running experiments)

    """
    # find all the FITS files
    print('%sZ: finding all FITS frames matching %s starting in %s' %
          (datetime.utcnow().isoformat(), fitsglob, fitsbasedir))

    fitslist = np.sort(glob.glob(os.path.join(fitsbasedir, fitsglob)))
    if not (frametype=='calibratedframes'
            or frametype=='arefshifted_frames'):
        raise NotImplementedError

    # choose the frame to database worker
    # generate the task list
    if frametype == 'calibratedframes':
        frame_to_db_worker = calframe_to_db_worker
        tasks = [(x, {'observatory':observatory,
                      'overwrite':overwrite,
                      'nonwcsframes_are_ok':nonwcsframes_are_ok,
                      'badframetag':badframetag,
                      'custom_projid':custom_projid}) for x in fitslist]
    elif frametype == 'arefshifted_frames':
        frame_to_db_worker = arefshifted_frame_to_db_worker
        tasks = [(x, {'overwrite':overwrite,
                      'nonwcsframes_are_ok':nonwcsframes_are_ok,
                      'badframetag':badframetag,
                      'custom_projid':custom_projid}) for x in fitslist]

    print('%sZ: %s files to send to db' %
          (datetime.utcnow().isoformat(), len(tasks)))

    if len(tasks) > 0:

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
        results = pool.map(frame_to_db_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}

    else:

        print('ERR! %sZ: none of the files specified exist, bailing out...' %
              (datetime.utcnow().isoformat(),))
        return



def dbupdate_calibratedframe(fitspath,
                             column, newval,
                             database=None):
    """
    This updates a column of the calibratedframes table for a frame.
    """

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    # start work here
    try:

        query = ("update calibratedframes "
                 "set {column} = %s, entryts = current_timestamp "
                 "where fits = %s").format(column=column)
        params = (newval, fitspath)
        cursor.execute(query, params)
        database.commit()

        return (fitspath, {column:newval})

    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to update %s in DB' % fitspath
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (fitspath, False)

        # TEMPORARY
        # raise


    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval




def convsubtracted_frame_to_database(
        fitsfile,
        overwrite=False,
        badframetag='badframes',
        database=None):
    """
    This puts a convolved and subtracted FITS into the DB.

    Associates it with the framekey of the original FITS and the framekey of the
    aref-shifted frame.  Adds info on subtraction success, its full path. Adds
    info about the kernel, and photref used, and type of photref used for
    subtraction.
    """

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    # start work here
    try:

        query = ("")
        params = ()

    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to update %s in DB' % fitspath
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (fitspath, False)

        # TEMPORARY
        # raise


    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



##################################
## ASTROMETRIC REFERENCE FRAMES ##
##################################

def dbgen_get_astromref(fieldinfo, observatory='hatpi', makeactive=True,
                        overwrite=False, refdir=REFBASEDIR, database=None):

    """
    This gets all the frame info from the DB and finds a good astromref.

    Args:

        fieldinfo: dict with keys that point to values for projectid,
        field, and ccd (if HATPI), or camera and ccd (if TESS).
    """

    if not os.path.exists(refdir):
        os.mkdir(refdir)

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    try:

        # get all the frame info for the requested projectid-field-ccd
        if observatory == 'hatpi':

            # e.g., projectid = 12, field = 'G1830-2230_577', ccd = 8
            projectid = fieldinfo['projectid']
            field = fieldinfo['field']
            ccd = fieldinfo['ccd']
            # "camera" is 0 for HATPI. (would be set by postgres db otherwise)
            camera = 0

            query = ("select framekey, fits, fistar, fiphot, "
                     "photinfo->'medsval', "
                     "photinfo->'meddval', photinfo->'medsrcbgv', "
                     "photinfo->'ngoodobjects' "
                     "from calibratedframes where "
                     "(fitsheader->'PROJID' = %s) and "
                     "(replace(fitsheader->'OBJECT'#>>'{}','-','') = %s) and "
                     "(fitsheader->'COMMENT'->>'CamGroups.CGncam' = %s) and "
                     "(frameisok = true)")

            # NOTE: the replace line is needed because of insane json rules that
            # say you are not allowed "-" characters. Best I could come up with.
            # This approach to ccd number extraction is also silly. Why is
            # it not already in the fits header?

            params = (str(projectid), field.replace('-',''), str(ccd))

        elif observatory == 'tess':

            # e.g., camera=1, ccd=1, field='ete6'
            camera = fieldinfo['camera']
            ccd = fieldinfo['ccd']
            field = fieldinfo['field']
            projectid = fieldinfo['projectid']

            query = ("select framekey, fits, fistar, fiphot, "
                     "photinfo->'medsval', "
                     "photinfo->'meddval', photinfo->'medsrcbgv', "
                     "photinfo->'ngoodobjects' "
                     "from calibratedframes where "
                     "(fitsheader->'PROJID' = %s) and "
                     "(fitsheader->'CAMERA' = %s) and "
                     "(fitsheader->'CCD' = %s) and "
                     "(frameisok = true)")

            params = (str(projectid), str(camera), str(ccd))

            if DEBUG:
                print('query: {:s}\nparams: {:s}'.format(
                    repr(query),repr(params)))

        else:
            raise ValueError('observatory must be tess or hatpi')

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # if we're successful
        if rows and len(rows) > 0:

            # get the frame info

            framekey = np.array([x[0] for x in rows])
            fits = np.array([x[1] for x in rows])
            fistar = np.array([x[2] for x in rows])
            fiphot = np.array([x[3] for x in rows])

            mfs = np.array([x[4] for x in rows], dtype=np.float)
            mfd = np.array([x[5] for x in rows], dtype=np.float)
            mbg = np.array([x[6] for x in rows], dtype=np.float)
            ngo = np.array([x[7] for x in rows], dtype=np.float)

            print('%sZ: total frames to process: %s' %
                  (datetime.utcnow().isoformat(), len(fits)))

            #
            # now, find the best astrometric reference frame
            #

            # get the best S --> largest S at the top
            median_sval_ind = np.argsort(mfs)[::-1]

            # here, we want the values closest to zero to be at the top
            median_dval_ind = np.argsort(np.fabs(mfd))

            # want the smallest background
            median_background_ind = np.argsort(mbg)

            # and the most number of detections
            good_detections_ind = np.argsort(ngo)[::-1]

            # get the top 500 of each index
            median_sval_ind = median_sval_ind[:500]
            median_dval_ind = median_dval_ind[:500]
            median_background_ind = median_background_ind[:500]
            good_detections_ind = good_detections_ind[:500]

            # now intersect all of these arrays to find the best candidates for
            # the astrometric reference frame

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

            # if all selectors produced a result, use that one
            if len(best_frame_ind) > 0:

                selectedreference = fits[best_frame_ind[0]]

                print('%sZ: selected best astrometric reference frame is %s' %
                      (datetime.utcnow().isoformat(), selectedreference))

                framejpg = fits_to_full_jpeg(
                    selectedreference,
                    out_fname=os.path.join(
                            os.path.dirname(selectedreference),
                        ('JPEG-ASTROMREF-%s.jpg' %
                         os.path.basename(re.sub(sv.FITS_TAIL,'',selectedreference)))
                    )
                )

                arefinfo = {
                    'fits':selectedreference,
                    'jpg':framejpg,
                    'fistar':fistar[best_frame_ind[0]],
                    'fiphot':fiphot[best_frame_ind[0]],
                    'framekey':framekey[best_frame_ind[0]],
                    'sval':mfs[best_frame_ind[0]],
                    'dval':mfd[best_frame_ind[0]],
                    'bgv':mbg[best_frame_ind[0]],
                    'ndet':ngo[best_frame_ind[0]],
                    'comment':'astromref chosen using sval, dval, bgval, ndet'
                }

            # otherwise, fall back to frames with the best values of S, D, and a
            # large number of detections
            elif len(sdndet_ind) > 0:

                selectedreference = fits[sdndet_ind[0]]

                print('WRN! %sZ: selected best astrometric reference frame '
                      '(using S, D, and ndet only) is %s' %
                      (datetime.utcnow().isoformat(), selectedreference))


                framejpg = fits_to_full_jpeg(
                    selectedreference,
                    out_fname=os.path.join(
                            os.path.dirname(selectedreference),
                        ('JPEG-ASTROMREF-%s.jpg' %
                         os.path.basename(re.sub(sv.FITS_TAIL,'',selectedreference)))
                    )
                )

                arefinfo = {
                    'fits':selectedreference,
                    'jpg':framejpg,
                    'fistar':fistar[sdndet_ind[0]],
                    'fiphot':fiphot[sdndet_ind[0]],
                    'framekey':framekey[sdndet_ind[0]],
                    'sval':mfs[sdndet_ind[0]],
                    'dval':mfd[sdndet_ind[0]],
                    'bgv':mbg[sdndet_ind[0]],
                    'ndet':ngo[sdndet_ind[0]],
                    'comment':'astromref chosen using sval, dval, ndet'
                }

            # otherwise, fall back to the frames with best values of S and D
            elif len(sd_ind) > 0:

                selectedreference = fits[sd_ind[0]]

                print('WRN! %sZ: selected best astrometric reference frame '
                      '(using S and D only) is %s' %
                      (datetime.utcnow().isoformat(), selectedreference))

                framejpg = fits_to_full_jpeg(
                    selectedreference,
                    out_fname=os.path.join(
                            os.path.dirname(selectedreference),
                        ('JPEG-ASTROMREF-%s.jpg' %
                         os.path.basename(re.sub(sv.FITS_TAIL,'',selectedreference)))
                    )
                )

                arefinfo = {
                    'fits':selectedreference,
                    'jpg':framejpg,
                    'fistar':fistar[sd_ind[0]],
                    'fiphot':fiphot[sd_ind[0]],
                    'framekey':framekey[sd_ind[0]],
                    'sval':mfs[sd_ind[0]],
                    'dval':mfd[sd_ind[0]],
                    'bgv':mbg[sd_ind[0]],
                    'ndet':ngo[sd_ind[0]],
                    'comment':'astromref chosen using sval, dval'
                }

            # if that also fails, fall back to the best S value frame
            elif len(median_sval_ind) > 0:

                selectedreference = fits[median_sval_ind[0]]

                print('WRN! %sZ: selected best astrometric reference frame '
                      '(using S only) is %s' %
                      (datetime.utcnow().isoformat(), selectedreference))

                framejpg = fits_to_full_jpeg(
                    selectedreference,
                    out_fname=os.path.join(
                            os.path.dirname(selectedreference),
                        ('JPEG-ASTROMREF-%s.jpg' %
                         os.path.basename(re.sub(sv.FITS_TAIL,'',selectedreference)))
                    )
                )

                arefinfo = {
                    'fits':selectedreference,
                    'jpg':framejpg,
                    'fistar':fistar[median_sval_ind[0]],
                    'fiphot':fiphot[median_sval_ind[0]],
                    'framekey':framekey[median_sval_ind[0]],
                    'sval':mfs[median_sval_ind[0]],
                    'dval':mfd[median_sval_ind[0]],
                    'bgv':mbg[median_sval_ind[0]],
                    'ndet':ngo[median_sval_ind[0]],
                    'comment':'astromref chosen using sval'
                }

            # if everything fails, do nothing
            else:

                print('ERR! %sZ: could not select a '
                      'good astrometric reference frame! all tests failed!' %
                      (datetime.utcnow().isoformat(), ))
                arefinfo = None

            # update the astromrefs table in the database if we found an
            # astromref frame, and copy it to the reference-frames directory
            # with the appropriate filename
            if arefinfo:

                # now that we have the astromref frame, copy it over to the
                # system-wide reference-images directory along with its JPEG
                # snapshot
                if observatory=='hatpi':
                    areftargetfits = (
                        'proj{projectid}-{field}-'
                        'ccd{ccd}-astromref-{origfname}.fits').format(
                            projectid=projectid,
                            field=field,
                            ccd=ccd,
                            origfname=os.path.splitext(
                                os.path.basename(arefinfo['fits'])
                            )[0]
                        )

                elif observatory=='tess':
                    areftargetfits = (
                        'proj{projectid}-'
                        'camera{camera}-ccd{ccd}-'
                        'astromref-{origfname}.fits').format(
                            projectid=projectid,
                            camera=camera,
                            ccd=ccd,
                            origfname=os.path.splitext(
                                os.path.basename(arefinfo['fits'])
                            )[0]
                        )

                areftargetjpeg = areftargetfits.replace('.fits','.jpg')
                areftargetfistar = areftargetfits.replace('.fits','.fistar')
                areftargetfiphot = areftargetfits.replace('.fits','.fiphot')

                # copy the frame, jpeg, fistar, and wcs to the reference-frames dir
                shutil.copy(arefinfo['fits'],os.path.join(refdir,
                                                          areftargetfits))
                shutil.copy(arefinfo['jpg'],os.path.join(refdir,
                                                         areftargetjpeg))
                shutil.copy(arefinfo['fistar'],
                            os.path.join(refdir, areftargetfistar))
                shutil.copy(arefinfo['fiphot'],
                            os.path.join(refdir, areftargetfiphot))
                shutil.copy(arefinfo['fits'].replace('.fits','.wcs'),
                            os.path.join(refdir,
                                         areftargetfits.replace('.fits','.wcs'))
                           )

                # now, update the astomrefs table in the database
                if overwrite and (
                    observatory=='tess' or observatory=='hatpi'
                ):

                    query = (
                        "insert into astromrefs ("
                        "projectid, field, camera, ccd, isactive, "
                        "unixtime, "
                        "framepath, jpegpath, "
                        "sval, dval, bgv, ndet, "
                        "comment"
                        ") values ("
                        "%s, %s, %s, %s, %s, "
                        "%s, "
                        "%s, %s, "
                        "%s, %s, %s, %s, "
                        "%s"
                        ") on conflict on constraint astromrefs_pkey "
                        "do update set "
                        "unixtime = %s, "
                        "framepath = %s, jpegpath = %s, "
                        "sval = %s, dval = %s, bgv = %s, ndet = %s, "
                        "comment = %s"
                    )
                    params = (
                        str(projectid), field, camera, ccd, int(makeactive),
                        time.time(),
                        os.path.join(refdir, areftargetfits),
                        os.path.join(refdir, areftargetjpeg),
                        arefinfo['sval'], arefinfo['dval'],
                        arefinfo['bgv'],arefinfo['ndet'], arefinfo['comment'],
                        time.time(),
                        os.path.join(refdir, areftargetfits),
                        os.path.join(refdir, areftargetjpeg),
                        arefinfo['sval'], arefinfo['dval'],
                        arefinfo['bgv'],arefinfo['ndet'], arefinfo['comment']
                    )

                elif not overwrite and (
                    observatory=='tess' or observatory=='hatpi'
                ):

                    query = (
                        "insert into astromrefs ("
                        "projectid, field, camera, ccd, isactive, "
                        "unixtime, "
                        "framepath, jpegpath, "
                        "sval, dval, bgv, ndet, "
                        "comment"
                        ") values ("
                        "%s, %s, %s, %s, %s, "
                        "%s, "
                        "%s, %s, "
                        "%s, %s, %s, %s, "
                        "%s"
                        ") "
                        "on conflict on constraint astromrefs_pkey do nothing"
                    )
                    params = (
                        int(projectid), str(field), int(camera), int(ccd), int(makeactive),
                        time.time(),
                        os.path.join(refdir, areftargetfits),
                        os.path.join(refdir, areftargetjpeg),
                        float(arefinfo['sval']), float(arefinfo['dval']),
                        float(arefinfo['bgv']), int(arefinfo['ndet']),
                        str(arefinfo['comment'])
                    )

                # execute the query to insert the astromref into the DB
                cursor.execute(query, params)
                database.commit()

                returnval = arefinfo

            # if we failed to find an astromref, do nothing
            else:

                returnval = None

        # if we didn't find any frames for this projectid-field-ccd combo,
        # complain and drop out
        else:

            print('ERR! %sZ: could not select a '
                  'good astrometric reference frame for %s!'
                  ' no frames exist' %
                  (datetime.utcnow().isoformat(), repr(fieldinfo)))
            print(params)
            print(cursor.query)
            returnval = None

    # catch the overwrite = False scenario
    except pg.IntegrityError as e:

        database.rollback()

        message = ('failed to insert astromref '
                   'into DB because it exists already '
                   'and overwrite = False')
        print('EXC! %sZ: %s\n%s' % (datetime.utcnow().isoformat(), message,
                                    format_exc()) )
        returnval = None

    except Exception as e:

        database.rollback()

        message = 'failed to get astromref for %s' % repr(fieldinfo)
        print('EXC! %sZ: %s\nexception was: %s' %
              (datetime.utcnow().isoformat(), message, format_exc()) )
        returnval = None

    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def generate_astromref(fitsfiles,
                       makeactive=True,
                       field=None,
                       ccd=None,
                       projectid=None,
                       refdir=REFBASEDIR,
                       refinfo=REFINFO,
                       overrideref=None):

    """This chooses an astrometry reference frame from the frames in fitfiles.

    writes the frame to refdir.

    ref frames have the following filename pattern:

    proj{projectid}-ccd{ccd}-{field}-astromref-{origfname}.fits

    if field, ccd, or projectid are None, these values are taken from the FITS
    file headers.

    updates the refinfo database.

    if overrideref is passed (a string path to a manually chosen reference
    frame), overrides the automated frame selection called in
    ism.select_astromref_frame.
    """

    goodfits = [x for x in fitsfiles if os.path.exists(x)]

    if not goodfits:
        print('ERR! %sZ: no good FITS files found in input list' %
              (datetime.utcnow().isoformat(),))
        return

    # check if database exists. if not, make it.
    if not os.path.exists(refinfo):

        db = sqlite3.connect(refinfo)
        cur = db.cursor()

        query = open(sv.TREXBASE + 'imagesub-refinfo.sql', 'r').read()

        cur.executescript(query)

        db.commit()
        cur.close()
        db.close()

        print('%sZ: initialized %s database' %
              (datetime.utcnow().isoformat(), refinfo))

    # find the astromref
    astromref = ism.select_astromref_frame(
        fitsfiles,
        '1-*.fits',
        overrideref=overrideref
    )

    # if an astromref was successfully found, then add its info to the DB
    if astromref:

        if field and ccd and projectid:

            frameinfo = {'field':field,
                         'ccd':ccd,
                         'projectid':projectid}

        else:

            # get the frame info
            frameelems = get_header_keyword_list(astromref['astromref'],
                                                 ['object',
                                                  'projid'])

            felems = FRAMEREGEX.findall(
                os.path.basename(astromref['astromref'])
            )

            if felems and felems[0]:

                ccd = felems[0][2]
                frameinfo = {'field':frameelems['object'],
                             'ccd':ccd,
                             'projectid':frameelems['projid']}

            else:

                print('ERR! %sZ: could not figure out CCD for astromref: %s' %
                      (datetime.utcnow().isoformat(), astromref['astromref']))
                return

            # now that we have the astromref frame, copy it over to the
            # system-wide reference-images directory along with its JPEG
            # snapshot
            areftargetfits = ('proj{projectid}-{field}-'
                              'ccd{ccd}-astromref-{origfname}.fits').format(
                                  projectid=frameinfo['projectid'],
                                  field=frameinfo['field'],
                                  ccd=frameinfo['ccd'],
                                  origfname=os.path.splitext(
                                      os.path.basename(astromref['astromref'])
                                  )[0]
                               )
            areftargetjpeg = areftargetfits.replace('.fits','.jpg')
            areftargetfistar = areftargetfits.replace('.fits','.fistar')

            # copy the frame, jpeg, and fistar to the reference-frames dir
            shutil.copy(astromref['astromref'],os.path.join(REFBASEDIR,
                                                            areftargetfits))
            shutil.copy(astromref['framejpg'],os.path.join(REFBASEDIR,
                                                            areftargetjpeg))
            shutil.copy(astromref['astromref'].replace('.fits','.fistar'),
                        os.path.join(REFBASEDIR, areftargetfistar))

            # now, put together the information and write to the refinfo sqlite

            query = ("insert into astromrefs "
                     "(field, projectid, ccd, isactive, unixtime, "
                     "framepath, jpegpath, sval, dval, bgv, ndet, "
                     "comment) values "
                     "(?, ?, ?, ?, ?, "
                     "?, ?, ?, ?, ?, ?, "
                     "?)")
            params = (frameinfo['field'],
                      frameinfo['projectid'],
                      frameinfo['ccd'],
                      1 if makeactive else 0,
                      time.time(),

                      os.path.join(REFBASEDIR,areftargetfits),
                      os.path.join(REFBASEDIR,areftargetjpeg),
                      astromref['sval'],
                      astromref['dval'],
                      astromref['bgv'],
                      astromref['ndet'],

                      (astromref['comment'] +
                       '; original: %s' % astromref['astromref']))

            db = sqlite3.connect(
                refinfo,
                detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
            )
            cur = db.cursor()

            try:

                astromref.update(frameinfo)
                cur.execute(query, params)
                db.commit()

                print('%sZ: using astromref %s for '
                      'field %s, ccd %s, project id %s, database updated.' %
                      (datetime.utcnow().isoformat(),
                       astromref['astromref'],
                       astromref['field'],
                       astromref['ccd'],
                       astromref['projectid']))

                returnval = astromref

            except Exception as e:

                print('ERR! %sZ: could not update refinfo DB! error was: %s' %
                      (datetime.utcnow().isoformat(), e))

                returnval = None
                db.rollback()

            db.close()

    # otherwise, do nothing
    else:

        print('ERR! %sZ: could not find an astromref frame' %
              (datetime.utcnow().isoformat(),))
        returnval = None


    return returnval



def dbget_astromref(projectid, field, ccd, database=None, camera=0):
    """
    This finds the reference frame using the PG database.
    """

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    # start work here
    try:

        query = (
            "select projectid, field, camera, ccd, unixtime, "
            "framepath, jpegpath, "
            "sval, dval, bgv, ndet, comment "
            "from astromrefs where "
            "projectid = %s and field = %s and camera = %s and ccd = %s and isactive = 1"
        )
        params = (str(projectid), field, camera, ccd)

        cursor.execute(query, params)
        row = cursor.fetchone()

        if row and len(row) > 0:

            astromref = {
                x:y for (x,y) in zip(('projectid','field','camera','ccd','unixtime',
                                      'framepath',
                                      'jpegpath','sval','dval',
                                      'bgv','ndet','comment'), row)
            }

            returnval = astromref

        else:
            returnval = None

    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to get astromref for %s, %s, %s from DB' % (projectid,
                                                                      field,
                                                                      ccd)
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = None
        raise


    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def get_astromref(projectid, field, ccd, refinfo=REFINFO):
    """
    This finds the reference frame for the field, projectid, and ccd
    combination using the TM-refinfo.sqlite database.
    """

    db = sqlite3.connect(
        refinfo,
        detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
    )
    cur = db.cursor()

    query = ('select field, projectid, ccd, unixtime, '
             'framepath, jpegpath, sval, dval, bgv, ndet, comment '
             'from astromrefs where '
             'projectid = ? and field = ? and ccd = ? and '
             'isactive = 1')
    params = (projectid, field, ccd)

    try:

        cur.execute(query, params)
        rows = cur.fetchone()

        astromref = {x:y for (x,y) in zip(('field','projectid','ccd',
                                           'unixtime','framepath','jpegpath',
                                           'sval','dval','bgv',
                                           'ndet','comment'),rows)}

        returnval = astromref

    except Exception as e:

        print('ERR! %sZ: could not get astromref info '
              'from DB! error was: %s' %
              (datetime.utcnow().isoformat(), e))
        returnval = None

    db.close()

    return returnval



def frames_astromref_worker(task):
    """
    This is the parallel worker for frames_to_astromref.

    task[0] = fits file
    task[1] = outdir
    task[2] = refinfo
    task[3] = warpcheck
    task[4] = warpcheck kwargs {'threshold', 'margins'}
    task[5] = observatory ('tess', or 'hatpi')
    task[6] = fieldinfo

        fieldinfo is a dict like
            {'ccd': 1, 'camera': 1, 'field':
            'ete6_field0', 'projectid': 42},
        for TESS. For HATPI, it can be None (since REGEXes were hard-coded to
        deal with HATPI naming scheme). If you were to pass it, it would look
        like:
            {'ccd': 8, 'camera': 42, 'field': 'G1830-2230_577', 'projectid': 12}
    """

    try:

        (frame, outdir, refinfo,
         warpcheck, warpcheckkwargs, observatory, fieldinfo ) = task

        # figure out this frame's field, ccd, and projectid
        if observatory=='hatpi':
            if fieldinfo is None:
                frameelems = get_header_keyword_list(frame, ['object', 'projid'])
                felems = FRAMEREGEX.findall(os.path.basename(frame))
                fieldinfo = {'field': frameelems['object'],
                             'ccd': felems[0][2],
                             'projectid': frameelems['projid'],
                             'camera': 0}
        elif observatory=='tess':
            pass
        else:
            raise NotImplementedError

        framefistar = frame.replace('.fits','.fistar')

        if os.path.exists(framefistar):
            # find this frame's associated active astromref
            framearef = dbget_astromref(fieldinfo['projectid'],
                                        fieldinfo['field'],
                                        fieldinfo['ccd'],
                                        camera=fieldinfo['camera'])
            areffistar = framearef['framepath'].replace('.fits','.fistar')
            print(areffistar)

            # calculate the shift and write the itrans back to the frame's
            # directory
            shifted_fistar, shifted_itrans = ism.astromref_shift_worker(
                (framefistar, areffistar, outdir)
            )

            # if the shift calculation is successful, shift the image itself
            if shifted_itrans and os.path.exists(shifted_itrans):

                frame_to_shift, shifted_frame = ism.frame_to_astromref_worker(
                    (frame, None, None)
                )

                if shifted_frame and os.path.exists(shifted_frame):

                    # check if the frame has warped too much after the shift,
                    # these frames look like they're folding into/out of the
                    # z-direction. we need to throw these away.
                    if warpcheck:

                        notwarped, warpinfo = check_frame_warping(
                            shifted_frame,
                            **warpcheckkwargs
                        )

                        # if image is OK, return it
                        if notwarped:

                            print('%sZ: SHIFT OK %s -> %s' %
                                  (datetime.utcnow().isoformat(),
                                   frame, shifted_frame))

                            return frame, shifted_frame

                        # otherwise, move it to the badframes subdir and mark it
                        # as warped
                        else:

                            badframesdir = os.path.join(os.path.dirname(frame),
                                                        'badframes')
                            if not os.path.exists(badframesdir):
                                os.mkdir(badframesdir)

                            # find all the components of this frame and move
                            # them to the badframes subdir
                            badframeglob = glob.glob(
                                os.path.join(
                                    os.path.dirname(shifted_frame),
                                    '*%s*.*' % (
                                        os.path.splitext(
                                            os.path.basename(frame)
                                        )[0]
                                    )
                                )
                            )

                            for x in badframeglob:
                                shutil.move(x, badframesdir)

                            # update the database with the new locations of the
                            # fits, fistar, fiphot, and wcs, and set frameisok
                            # to false

                            # old paths
                            framefiphot = frame.replace('.fits','.fiphot')
                            framewcs = frame.replace('.fits','.wcs')
                            newfiphot = os.path.abspath(
                                os.path.join(badframesdir,
                                             os.path.basename(framefiphot))
                            )
                            newwcs = os.path.abspath(
                                os.path.join(badframesdir,
                                             os.path.basename(framewcs))
                            )
                            newfistar = os.path.abspath(
                                os.path.join(badframesdir,
                                             os.path.basename(framefistar))
                            )
                            newfits = os.path.abspath(
                                os.path.join(badframesdir,
                                             os.path.basename(frame))
                            )

                            # update the database
                            fistarup = dbupdate_calibratedframe(
                                os.path.abspath(frame), 'fistar', newfistar
                            )
                            fiphotup = dbupdate_calibratedframe(
                                os.path.abspath(frame), 'fiphot', newfiphot
                            )
                            wcsup = dbupdate_calibratedframe(
                                os.path.abspath(frame), 'wcs', newwcs
                            )
                            statup = dbupdate_calibratedframe(
                                os.path.abspath(frame), 'frameisok', False
                            )
                            # update the fits path last, since it's the selector
                            # for everything else
                            fitsup = dbupdate_calibratedframe(
                                os.path.abspath(frame), 'fits', newfits
                            )

                            print('WRN! %sZ: SHIFT HAS WARPED '
                                  'IMAGE, moved %s and '
                                  'metadata to %s, updated DB' %
                                  (datetime.utcnow().isoformat(),
                                   frame, badframesdir))

                            return frame, None

                    # if we're not checking for warps, just check if the image
                    # was shifted fine
                    else:

                        print('%sZ: SHIFT OK %s -> %s' %
                              (datetime.utcnow().isoformat(),
                               frame, shifted_frame))

                        return frame, shifted_frame


                else:

                    print('ERR! %sZ: SHIFT OPERATION FAILED for %s' %
                          (datetime.utcnow().isoformat(), frame))
                    return frame, None

            else:

                print('ERR! %sZ: SHIFT CALCULATION FAILED for %s' %
                      (datetime.utcnow().isoformat(), frame))
                return frame, None

        else:
            print('ERR! %sZ: could not find fistar for frame: %s' %
                  (datetime.utcnow().isoformat(), frame))
            return frame, None

    except Exception as e:

        print('ERR! %sZ: could not shift frame %s to astromref, error was: %s' %
              (datetime.utcnow().isoformat(), frame, e))
        return frame, None



def framelist_make_xtrnsfits(fitsfiles,
                             fitsdir=sv.REDPATH,
                             fitsglob=sv.LOCAL_GLOBPATTERN,
                             outdir=None,
                             refinfo=REFINFO,
                             overwrite=False,
                             warpcheck=False,
                             warpthreshold=15000.0,
                             warpmargins=100,
                             nworkers=16,
                             observatory='hatpi',
                             maxworkertasks=1000,
                             fieldinfo=None):
    """
    This calculates the shifts between frames in fitsfiles and the appropriate
    astromref for the projectid, field and CCD, then shifts each frame to the
    astromref's coordinate system, generating -xtrns.fits files.

    Per the docstring of imageutils.check_frame_warping, "warpcheck" by default
    is set OFF, because it depends on a detailed (manual) empirical calibration
    step.
    """

    # check if astrometric translation was already done.
    existing = glob.glob(
        os.path.join(fitsdir, fitsglob.replace('.fits', '-xtrns.fits'))
    )

    requested = list(map(os.path.basename, fitsfiles))
    alreadyexists = list(map(os.path.basename, existing))

    if len(existing) > 0 and not overwrite:

        # substitute out the hash string
        alreadyexists = [ae.replace('-xtrns.fits','') for ae in alreadyexists]
        requested = [r.replace('.fits','') for r in requested]

        setdiff = np.setdiff1d(requested, alreadyexists)

        if len(setdiff) == 0:
            print('WRN! %sZ: already astrom-shifted all requested frames.' %
                  (datetime.utcnow().isoformat(),))
            return

        else:
            fitsfiles = [os.path.join(fitsdir, sd+'.fits') for sd in setdiff]


    print('%sZ: %s files to astrometrically shift' %
          (datetime.utcnow().isoformat(), len(fitsfiles)))

    tasks = [(x, outdir, refinfo, warpcheck,
              {'threshold':warpthreshold, 'margins':warpmargins},
             observatory, fieldinfo)
             for x in fitsfiles if os.path.exists(x)]

    # fire up the pool of workers
    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
    results = pool.map(frames_astromref_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    print('%sZ: done with astrometric shifting' %
          (datetime.utcnow().isoformat()))

    xtrns = glob.glob(fitsdir+ fitsglob.replace('.fits','-xtrns.fits'))
    wcsfiles = glob.glob(fitsdir+ fitsglob.replace('.fits','.wcs'))

    if len(xtrns) != len(wcsfiles):
        if len(wcsfiles) - len(xtrns) < 10:
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
                '\nN_xtrns: {:d}'.format(len(xtrns))
            )

    return {x:y for (x,y) in results}



##################################
## PHOTOMETRIC REFERENCE FRAMES ##
##################################

def generate_photref_candidates_from_xtrns(fitsfiles,
                                           minframes=50,
                                           observatory='hatpi',
                                           maxhourangle=3.0,
                                           maxmoonphase=25.0,
                                           maxmoonelev=0.0,
                                           maxzenithdist=30.0,
                                           maxbackgroundstdevpctile=100.,
                                           maxbackgroundmedianpctile=10.,
                                           minngoodobjectpctile=90.,
                                           forcecollectinfo=False,
                                           nworkers=8,
                                           maxworkertasks=1000):
    """
    This uses ism.select_photref_frames run on fitsfiles to get photref
    candidates.

    fitsfiles must be a list of frames, which have been already transformed to
    the astromref, and are all from a single projectid, ccd, field combination
    for this operation to make sense.

    Args:
        minframes: minimum number of candidate frames needed to construct
        photometric reference

        observatory: 'hatpi' or 'tess'

        maxhourangle: ignored if TESS

        maxmoonphase: ignored if TESS

        maxmoonelev: ignored if TESS

        maxzenithdist: ignored if TESS

        maxbackgroundstdevpctile: percentile (given in %) from array of
        background standard deviations from each frame. I don't understand why
        this would be a good selector. Set as `100.` to not do anything for
        this. ???

        maxbackgroundmedianpctile: percentile (given in %) from array of
        background medians from each frame. For example, `10.0` would give the
        top 10% of frames with small background medians (good for few clouds,
        or if moon/earth in TESS FoV).

        minngoodobjectpctile: if ndetpercentile=90, will select frames from the
        top 90% of "ngoodobjects". If there are clouds, there won't be many
        well-detected objects.

        forcecollectinfo:

    Returns:

        photrefinfo, a dictionary with information about chosen photometric
        reference frames.
    """

    if not os.path.exists(FRAMEINFOCACHEDIR):
        os.mkdir(FRAMEINFOCACHEDIR)

    # first, get all the info from these fits files.
    frameinfo = fitslist_frameinfo(fitsfiles,
                                   forcecollectinfo=False,
                                   nworkers=nworkers,
                                   maxworkertasks=maxworkertasks)

    # this is the cachekey used to store the photref selection info
    cachekey = '%s-%i-%.1f-%.1f-%.1f-%.1f-%.1f-%.1f-%.1f' % (repr(fitsfiles),
                                                             minframes,
                                                             maxhourangle,
                                                             maxmoonphase,
                                                             maxmoonelev,
                                                             maxzenithdist,
                                                             maxbackgroundstdevpctile,
                                                             maxbackgroundmedianpctile,
                                                             minngoodobjectpctile)
    cachekey = md5(cachekey.encode('utf-8')).hexdigest()
    cachedir = os.path.join(FRAMEINFOCACHEDIR,'TM-photref-%s' % cachekey)
    cacheinfofile = os.path.join(cachedir, 'selection-info.pkl.gz')

    # get the data from the cache if it exists and we're allowed to use it
    if ((not forcecollectinfo) and
        os.path.exists(cachedir) and
        os.path.exists(cacheinfofile)):

        with gzip.open(cacheinfofile) as infd:
            photrefinfo = pickle.load(infd)

        print('%sZ: candidate photref JPEGs in: %s, '
              'cached photrefinfo from: %s' %
              (datetime.utcnow().isoformat(), cachedir, cacheinfofile))

        return photrefinfo

    # apply our conditions to these fits files to generate a list of
    # photref candidates

    if observatory=='hatpi':
        # filter on hour angle
        haind = np.fabs(frameinfo['hourangle']) < maxhourangle
        print('%sZ: %s frames with hour angle < %s' %
              (datetime.utcnow().isoformat(), len(np.where(haind)[0]),
               maxhourangle))

        # get dark nights
        moonind = ((np.fabs(frameinfo['moonphase']) < maxmoonphase) |
                   (frameinfo['moonelev'] < maxmoonelev))
        print('%sZ: %s frames with moon phase < %s or moon elev < %s' %
              (datetime.utcnow().isoformat(), len(np.where(moonind)[0]),
               maxmoonphase, maxmoonelev))

        # get low zenith distance nights
        zenithind = frameinfo['zenithdist'] < maxzenithdist
        print('%sZ: %s frames with zenith distance < %s' %
              (datetime.utcnow().isoformat(), len(np.where(zenithind)[0]),
               maxzenithdist))

    # get nights with background stdev < max_bgv_stdev (to possibly remove bad
    # frames)
    maxbackgroundstdev = np.nanpercentile(frameinfo['stdsrcbgv'],
                                          maxbackgroundstdevpctile)
    backgroundstdevind = frameinfo['stdsrcbgv'] < maxbackgroundstdev
    print('%sZ: %s frames with background stdev < %s' %
          (datetime.utcnow().isoformat(), len(np.where(backgroundstdevind)[0]),
           maxbackgroundstdev))

    # get nights with background median < maxbackgroundmedian (to possibly
    # remove cloudy frames, or frames with lots of scattered light from the
    # moon or other very bright objects)
    maxbackgroundmedian = np.nanpercentile(frameinfo['medsrcbgv'],
                                           maxbackgroundmedianpctile)
    backgroundmedind = frameinfo['medsrcbgv'] < maxbackgroundmedian
    print('%sZ: %s frames with background median < %s' %
          (datetime.utcnow().isoformat(), len(np.where(backgroundmedind)[0]),
           maxbackgroundmedian))

    # get nights with ngoodobjects > minngoodobjects (to possibly
    # remove cloudy nights). minus 1 because for space-based data, you can
    # sometimes have frame stacks with identical numbers of "good objects"
    # parsed.
    minngoodobjects = (
        np.nanpercentile(frameinfo['ngoodobjects'], minngoodobjectpctile)
        - 1
    )
    ngoodobjectind = frameinfo['ngoodobjects'] > minngoodobjects

    print('%sZ: %s frames with ngoodobjects > %s' %
          (datetime.utcnow().isoformat(),
           len(np.where(ngoodobjectind)[0]),
           minngoodobjects))

    # this is the final operating set of frames that will be sorted for the
    # following tests
    if observatory=='hatpi':
        selectind = haind & moonind & zenithind & backgroundstdevind & ngoodobjectind
    elif observatory=='tess':
        selectind = backgroundstdevind & ngoodobjectind & backgroundmedind
    else:
        raise NotImplementedError

    selected_frames = frameinfo['frames'][selectind]
    selected_ngoodobj = frameinfo['ngoodobjects'][selectind]

    selected_medmagerr = frameinfo['medmagerr'][selectind]
    selected_magerrmad = frameinfo['magerrmad'][selectind]

    selected_medsrcbgv = frameinfo['medsrcbgv'][selectind]
    selected_stdsrcbgv = frameinfo['stdsrcbgv'][selectind]

    selected_medsvalue = frameinfo['medsval'][selectind]
    selected_meddvalue = frameinfo['meddval'][selectind]

    print('\n%sZ: selected %s frames with acceptable '
          'HA, Z, moon phase, background, elevation, and ngoodobjects '
          'for further filtering...\n' %
          (datetime.utcnow().isoformat(), len(selected_frames)))

    # we select in the following order
    # 1. D closest to 0
    # 2. largest S

    # then we filter out any images that have background > maxbackgroundmedian
    # and backgroundstdev > maxbackgroundstdev

    # first sort selector
    stage1_sort_ind = (np.argsort(selected_medsvalue))[::-1]

    stage1_frames = selected_frames[stage1_sort_ind[:2*minframes]]
    stage1_median_bgv = selected_medsrcbgv[stage1_sort_ind[:2*minframes]]
    stage1_stdev_bgv = selected_stdsrcbgv[stage1_sort_ind[:2*minframes]]
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
    try:

        candidate_master_photref = final_frames[np.nanargmin(final_svalues)]
        final_jpegs = []

        # make JPEGs of the selected photref frames and copy them to the
        # cachedir
        if not os.path.exists(cachedir):
            print('WRN! %sZ: making new photref cache directory: %s' %
                  (datetime.utcnow().isoformat(), cachedir))
            os.mkdir(cachedir)

        for final_frame in final_frames:

            framejpg = fits_to_full_jpeg(
                final_frame,
                out_fname=os.path.join(
                    cachedir,
                    ('JPEG-PHOTREF-%s.jpg' %
                     os.path.basename(final_frame).rstrip('.fits.fz'))
                    )
                )
            final_jpegs.append(framejpg)

        photrefinfo = {
            'framelist':fitsfiles,
            'frameinfo':frameinfo,
            'cachekey':cachekey,
            'minframes':minframes,
            'maxhourangle':maxhourangle,
            'maxmoonphase':maxmoonphase,
            'maxmoonelev':maxmoonelev,
            'maxzenithdist':maxzenithdist,
            'maxbackgroundstdev':maxbackgroundstdev,
            'maxbackgroundmedian':maxbackgroundmedian,
            'masterphotref':os.path.abspath(candidate_master_photref),
            'photrefs':[os.path.abspath(x) for x in final_frames],
            'photrefjpegs':final_jpegs
        }

        # dump the photrefinfo to a pickle
        with gzip.open(cacheinfofile,'wb') as outfd:
            pickle.dump(photrefinfo, outfd, pickle.HIGHEST_PROTOCOL)

        print('%sZ: candidate photref JPEGs in: %s, photrefinfo dumped to: %s' %
              (datetime.utcnow().isoformat(), cachedir, cacheinfofile))

        return photrefinfo

    except Exception as e:

        print('ERR! %sZ: selection failed, some criteria '
              'may be too strict for this frame list' %
              (datetime.utcnow().isoformat()))

        return {'framelist':fitsfiles,
                'frameinfo':frameinfo,
                'cachekey':cachekey,
                'maxhourangle':maxhourangle,
                'maxmoonphase':maxmoonphase,
                'maxmoonelev':maxmoonelev,
                'maxzenithdist':maxzenithdist,
                'maxbackgroundstdev':maxbackgroundstdev,
                'maxbackgroundmedian':maxbackgroundmedian,
                'masterphotref':None,
                'photrefs':None,
                'photrefjpegs':None}



def amend_candidate_photrefs(photrefinfo):
    """
    This is an interactive way to update masterphotref, photrefs, and
    photrefjpegs after reviewing them.

    This will automatically update the photrefinfo cache.

    Args:
        photrefinfo: dictionary returned by
        generate_photref_candidates_from_xtrns

    Returns:
        photrefinfo: updated version of same dictionary
    """

    cachekey = photrefinfo['cachekey']
    cachedir = os.path.join(FRAMEINFOCACHEDIR,'TM-photref-%s' % cachekey)
    cacheinfofile = os.path.join(cachedir, 'selection-info.pkl.gz')

    print('reviewing photrefinfo for %s\n' % cachedir)

    # now deal with the photrefs:
    print('-- CANDIDATE PHOTREFS --\n')

    initialphotrefs = sorted(photrefinfo['photrefs'][::])
    initialphotrefjpegs = sorted(photrefinfo['photrefjpegs'][::])

    for frame, jpeg in zip(initialphotrefs, initialphotrefjpegs):

        breakloop = False

        photref_prompt = (
            'photref = %s, jpeg = %s\n'
            '[ENTER] to keep this, or [x] to remove: ' %
            (frame, os.path.basename(jpeg))
        )

        while not breakloop:

            photref_check = raw_input(photref_prompt)

            if photref_check and photref_check == 'x':

                photrefinfo['photrefs'].remove(frame)
                photrefinfo['photrefjpegs'].remove(jpeg)
                os.remove(jpeg)

                print('REMOVED photref %s' % frame)
                breakloop = True

            elif not photref_check:
                breakloop = True

    print('\nfinal photrefs set to:')
    for frame in photrefinfo['photrefs']:
        print(frame)

    # next, update the masterphotref
    masterphotref_prompt = (
        'current masterphotref = %s\n'
        '[ENTER] to keep this, or new masterphot: ' %
        photrefinfo['masterphotref']
    )

    breakloop = False

    print('\n-- MASTERPHOTREF --\n')

    # loop until masterphotref is satisfied
    while not breakloop:

        masterphotref_amendment = raw_input(masterphotref_prompt)

        if masterphotref_amendment and os.path.exists(masterphotref_amendment):

            photrefinfo['masterphotref'] = masterphotref_amendment[::]

            masterphotref_prompt = (
                'new masterphotref = %s\n'
                '[ENTER] to keep this, or new masterphot: ' %
                photrefinfo['masterphotref']
            )

        elif ((masterphotref_amendment) and
              (not os.path.exists(masterphotref_amendment))):

            masterphotref_prompt = (
                'masterphotref = %s does not exist\n'
                'new masterphot: ' %
                masterphotref_amendment
            )

        elif not masterphotref_amendment:
            breakloop = True

    print('\nmasterphotref set to %s' % photrefinfo['masterphotref'])

    # update the cache info file
    print('\nupdating photref cached selection-info pickle...')

    # dump the photrefinfo to a pickle
    with gzip.open(cacheinfofile,'wb') as outfd:
        pickle.dump(photrefinfo, outfd, pickle.HIGHEST_PROTOCOL)

    print('%sZ: candidate photref JPEGs in: %s, photrefinfo dumped to: %s' %
          (datetime.utcnow().isoformat(), cachedir, cacheinfofile))

    return photrefinfo



def generate_combined_photref(
        photrefinfo,
        photreftype,
        dbtype,
        ra_nom,
        dec_nom,
        photref_reformedfovcat=None,
        makeactive=True,
        field=None,
        ccd=None,
        projectid=None,
        refdir=REFBASEDIR,
        refinfo=REFINFO,
        fovcatdir=FIELDCAT_DIR,
        combinemethod='median',
        kernelspec='b/4;i/4;d=4/4',
        ccdgain=None,
        zeropoint=None,
        ccdexptime=None,
        extractsources=True,
        photreffluxthreshold=25000,
        apertures='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
        framewidth=None,
        searchradius=8.0,
        nworkers=8,
        maxworkertasks=1000,
        fieldinfo=None,
        observatory='hatpi',
        overwrite=False,
        useimagenotfistar=False,
        astrometrydownsample=2,
        pixelerror=0.3,
        uniformize=10,
        reformed_cat_file=None):
    """
    This generates a combined photref from photref target and candidates and
    updates the sqlite or postgres database.

    Use this after reviewing the results from
    generate_photref_candidates_from_xtrns function above. Amend the
    photrefinfo['masterphotref'], photrefinfo['photrefs'], and
    photrefinfo['photrefjpegs'] arrays as needed using the
    amend_candidate_photrefs function above.

    Args:
        photreffluxthreshold: This is the threshold flux used to extract stars
        from the photometric reference, if extractsources is True.

        photrefinfo is the output of generate_photref_candidates_from_xtrns

        photreftype is the type of the combined photref produced. it must be
        one of the following strings:

            'oneframe' -> single HATPI frame
            'onehour' -> up to 120 HATPI frames
            'onenight' -> up to 960 HATPI frames

        dbtype is either 'sqlite' or 'postgres'.

        searchradius: astrometry.net solver search radius (in degrees)

        fieldinfo: optional dict (used if reducing tess data) of form
            {'ccd': 1, 'camera': 1, 'field': 'ete6_field0', 'projectid': 42},

        fovcatalog: the REFORMED FoV catalog file (12 columns)

        useimagenotfistar (bool): if True, does astrometry.net internal source
        extract (when observatory=='tess'). Otherwise, uses the fistar source
        positions in the astrometry solver. A benefit of setting to True for
        wide fields is that it enables easy downsampling, which can help with
        wide-field astrometry convergence problems.

    Returns:

        updates photrefinfo with the following dict and keys:

        'combinedphotref':{'frame': -> combined photref frame path
                           'jpeg' -> combined photref jpeg path
                           'cmrawphot' -> cmrawphot file path
                           'regfile' -> convolution registration file path
                           'combinemethod'- > combine type
                           'reftype' -> combined photref type
                           'phottype' -> either 're-extracted' or 'cat-projected'
                           'photaps' -> photometry apertures for combined photref
                           'fovcat' -> fovcat file used for photometry
                           'kernelspec' -> convolution kernel specs}

        updates the cached selection-info pickle file as well.

        the output combined photref frame, jpeg, cmrawphot (and byproducts) go to
        the REFBASEDIR, using the following prototype for the filename:

        {REFBASEDIR}/proj{projid}-{field}-ccd{ccd}-combinedphotref-{photreftype}.XXX
    """

    assert (dbtype == 'postgres') or (dbtype == 'sqlite')

    # get the field, ccd, projectid first (from the convolvetarget =
    # masterphotref)

    masterphotref = photrefinfo['masterphotref']

    if observatory=='hatpi':
        if fieldinfo is not None:
            cam = fieldinfo['camera']
            ccd = fieldinfo['ccd']
            masterphotrefinfo = {'field':fieldinfo['field'],
                                 'ccd':fieldinfo['ccd'],
                                 'cam':fieldinfo['camera'],
                                 'projectid':fieldinfo['projectid']}
        else:
            # Infer field info
            frameelems = get_header_keyword_list(masterphotref,
                                                 ['object','projid'])
            felems = FRAMEREGEX.findall(
                os.path.basename(masterphotref)
            )

            if felems and felems[0]:
                cam = 0
                ccd = felems[0][2]
                masterphotrefinfo = {'field':frameelems['object'],
                                     'cam':cam,
                                     'ccd':int(ccd),
                                     'projectid':frameelems['projid']}

    elif observatory=='tess':
        cam = fieldinfo['camera']
        ccd = fieldinfo['ccd']
        masterphotrefinfo = {'field':fieldinfo['field'],
                             'ccd':fieldinfo['ccd'],
                             'cam':fieldinfo['camera'],
                             'projectid':fieldinfo['projectid']}

    else:

        print('ERR! %sZ: could not figure out CCD for masterphotref: %s' %
              (datetime.utcnow().isoformat(), masterphotref))
        return

    # make the convolution registration file

    photreffname = ('proj{projid}-{field}-cam{cam}-ccd{ccd}'
                    '-combinedphotref-{photreftype}.{fileext}')

    regfpath = os.path.join(
        refdir,
        photreffname.format(
            projid=masterphotrefinfo['projectid'],
            field=masterphotrefinfo['field'],
            cam=masterphotrefinfo['cam'],
            ccd=masterphotrefinfo['ccd'],
            photreftype=photreftype,
            fileext='reg'
        )
    )

    # the output combinedphotref path
    combinedphotrefpath = os.path.join(
        refdir,
        photreffname.format(
            projid=masterphotrefinfo['projectid'],
            field=masterphotrefinfo['field'],
            cam=masterphotrefinfo['cam'],
            ccd=masterphotrefinfo['ccd'],
            photreftype=photreftype,
            fileext='fits'
        )
    )

    if (overwrite is False and
        os.path.exists(regfpath) and os.path.exists(combinedphotrefpath)):
        print(
            'WRN! {:s}Z: found regfpath {:s} and combinedphotrefpath {:s} '.
            format(datetime.utcnow().isoformat(),
                   regfpath,
                   combinedphotrefpath)+
            'continuing'
        )
        return

    masterphotref_fistar = masterphotref.replace('-xtrns.fits','.fistar')

    if not os.path.exists(masterphotref_fistar):
        print('ERR! %sZ: no fistar available for masterphotref: %s' %
              (datetime.utcnow().isoformat(), masterphotref))
        return

    # conv registration file
    ism.genreg(masterphotref_fistar, regfpath)

    if not os.path.exists(regfpath):
        print('ERR! %sZ: could not make regfile for masterphotref: %s' %
              (datetime.utcnow().isoformat(), masterphotref))
        return

    # convolve all candidate photrefs to the masterphotref
    convresult = ism.convolve_photref_frames(photrefinfo['photrefs'],
                                             masterphotref,
                                             regfpath,
                                             kernelspec=kernelspec,
                                             nworkers=nworkers,
                                             maxworkertasks=maxworkertasks)

    # get the convolved photref frames
    convphotrefs = [convresult[x] for x in convresult
                    if os.path.exists(convresult[x])]

    if len(convphotrefs) == 0:
        print('ERR! %sZ: convolution of photrefs to masterphotref: %s failed' %
              (datetime.utcnow().isoformat(), masterphotref))
        return

    # combine all the convolved photrefs into a single combinedphotref, using
    # combinemethod

    combinedphotref = ism.combine_frames(convphotrefs,
                                         combinedphotrefpath,
                                         combinemethod=combinemethod)

    if not (combinedphotref[1] and os.path.exists(combinedphotref[1])):
        print('ERR! %sZ: combining conv photrefs '
              'into masterphotref: %s failed' %
              (datetime.utcnow().isoformat(), masterphotref))
        return

    # rearrange the returned combinedphotref filename
    combinedphotref = combinedphotref[1]

    if not photref_reformedfovcat:

        # find the fovcat file for the field, ccd, projectid, photreftype combo
        # photreftype = 'oneframe' -> default field-gri.catalog
        # photreftype = 'onehour' -> default field-gri-18.0.catalog
        # photreftype = 'onenight' -> default field-gri-20.0.catalog

        fovcat_template = '{field}{bandspec}{magspec}.catalog'

        if photreftype == 'oneframe':
            photref_reformedfovcatpath = os.path.join(
                fovcatdir,
                fovcat_template.format(
                    field=masterphotrefinfo['field'],
                    bandspec='-gri',
                    magspec=''
                    )
                )
        elif photreftype == 'onehour':
            photref_reformedfovcatpath = os.path.join(
                fovcatdir,
                fovcat_template.format(
                    field=masterphotrefinfo['field'],
                    bandspec='-gri',
                    magspec='-18.0'
                    )
                )
        elif photreftype == 'onenight':
            photref_reformedfovcatpath = os.path.join(
                fovcatdir,
                fovcat_template.format(
                    field=masterphotrefinfo['field'],
                    bandspec='-gri',
                    magspec='-20.0'
                    )
                )
        else:
            print('ERR! %sZ: unknown photreftype: %s specified '
                  'can\'t continue...' %
                  (datetime.utcnow().isoformat(), photreftype))
            return

    else:
        photref_reformedfovcatpath = photref_reformedfovcat

    if not os.path.exists(photref_reformedfovcatpath):
        print('ERR! %sZ: no FOV catalog available '
              'for field %s, photreftype %s, '
              'can\'t do photometry on combinedphotref %s' %
              (datetime.utcnow().isoformat(),
               masterphotref, photreftype, combinedphotref))
        return


    # run photometry on the combinedphotref and generate a cmrawphot file. this
    # produces the base photometry values that we'll be diffing from those
    # found in the difference images to get difference magnitudes.
    # if extractsources==False, a placeholder fistar file with SDK values is
    # nonetheless generated, to be used for photometric reference frame
    # statistical bookkeeping.
    cphotref_photometry = ism.photometry_on_combined_photref(
        combinedphotref,
        photref_reformedfovcatpath,
        ra_nom,
        dec_nom,
        masterphotrefinfo['ccd'],
        ccdgain=ccdgain,
        zeropoint=zeropoint,
        ccdexptime=ccdexptime,
        extractsources=extractsources,
        apertures=apertures,
        framewidth=framewidth,
        searchradius=searchradius,
        photreffluxthreshold=photreffluxthreshold,
        observatory=observatory,
        useimagenotfistar=useimagenotfistar,
        astrometrydownsample=astrometrydownsample,
        pixelerror=pixelerror,
        uniformize=uniformize,
        reformed_cat_file=reformed_cat_file,
        projid=masterphotrefinfo['projectid']
    )

    if not (cphotref_photometry and
            cphotref_photometry[1] and
            os.path.exists(cphotref_photometry[1])):
        print('ERR! %sZ: photometry failed for combinedphotref %s '
              'can\'t continue...' %
              (datetime.utcnow().isoformat(), combinedphotref))
        return

    # update the cache photref selection-info.pkl.gz file
    combinedphotrefinfo = {
        'reftype':photreftype,
        'frame':combinedphotref,
        'jpeg':combinedphotref.replace(
            '.fits', '.jpg').replace('proj', 'JPEG-COMBINED-proj'),
        'cmrawphot':cphotref_photometry[1],
        'regfile':regfpath,
        'combinemethod':combinemethod,
        'kernelspec':kernelspec,
        'phottype':'re-extracted' if extractsources else 'cat-projected',
        'photaps':apertures,
        'fovcat':photref_reformedfovcatpath,
    }
    photrefinfo['combinedphotref'] = combinedphotrefinfo

    cachekey = photrefinfo['cachekey']
    cachedir = os.path.join(FRAMEINFOCACHEDIR,'TM-photref-%s' % cachekey)
    cacheinfofile = os.path.join(cachedir, 'selection-info.pkl.gz')

    with gzip.open(cacheinfofile, 'wb') as outfd:
        print('%sZ: combined photref JPEG: %s, photrefinfo updated: %s' %
              (datetime.utcnow().isoformat(),
               photrefinfo['combinedphotref']['jpeg'],
               cacheinfofile))
        pickle.dump(photrefinfo, outfd, pickle.HIGHEST_PROTOCOL)


    # update the TM-refinfo.sqlite database, or the psql database, as the user
    # decided.

    # first, get the frame info from the combinedphotref
    combinedphotref_fistar = combinedphotref.replace('.fits','.fistar')
    if not os.path.exists(combinedphotref_fistar):
        print('WRN! %sZ: did not find %s.' %
              (datetime.utcnow().isoformat(), combinedphotref_fistar)
             )
        print('It is needed for SDK values which are used to so making it '
             )
        _ = ap.extract_frame_sources(fits, None,
                                     fluxthreshold=photreffluxthreshold,
                                     ccdgain=ccdgain, zeropoint=zeropoint,
                                     exptime=ccdexptime)

    _, photref_frameinfo = get_frame_info(combinedphotref)

    if not photref_frameinfo:
        print('ERR! %sZ: could not extract frame info from combinedphotref %s' %
              (datetime.utcnow().isoformat(), combinedphotref))
        return

    if dbtype == 'sqlite':
        db = sqlite3.connect(
            refinfo,
            detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
        )
        query = ("insert into photrefs ("
                 "field, projectid, camera, ccd, photreftype, isactive, unixtime, "
                 "framepath, jpegpath, "
                 "convolvetarget, convolveregpath, cmrawphotpath, "
                 "target_zenithdist, target_moondist, target_moonelev, "
                 "target_moonphase, target_hourangle, target_ndet, "
                 "target_medmagerr, target_magerrmad, target_medsrcbgv, "
                 "target_stdsrcbgv, target_medsval, target_meddval, "
                 "photrefinfo"
                 ") values ("
                 "?, ?, ?, ?, ?, ?, ?, "
                 "?, ?, "
                 "?, ?, ?, "
                 "?, ?, ?, "
                 "?, ?, ?, "
                 "?, ?, ?, "
                 "?, ?, ?, "
                 "?"
                 ")")

    elif dbtype == 'postgres':
        db = pg.connect(user=PGUSER,
                        password=PGPASSWORD,
                        database=PGDATABASE,
                        host=PGHOST)
        query = ("insert into photrefs ("
                 "field, projectid, camera, ccd, photreftype, isactive, unixtime, "
                 "framepath, jpegpath, "
                 "convolvetarget, convolveregpath, cmrawphotpath, "
                 "target_zenithdist, target_moondist, target_moonelev, "
                 "target_moonphase, target_hourangle, target_ndet, "
                 "target_medmagerr, target_magerrmad, target_medsrcbgv, "
                 "target_stdsrcbgv, target_medsval, target_meddval, "
                 "photrefinfo"
                 ") values ("
                 "%s, %s, %s, %s, %s, %s, %s, "
                 "%s, %s, "
                 "%s, %s, %s, "
                 "%s, %s, %s, "
                 "%s, %s, %s, "
                 "%s, %s, %s, "
                 "%s, %s, %s, "
                 "%s"
                 ")")

    params = (
        masterphotrefinfo['field'],
        masterphotrefinfo['projectid'],
        masterphotrefinfo['cam'],
        masterphotrefinfo['ccd'],
        photreftype,
        1 if makeactive else 0,
        time.time(),

        photrefinfo['combinedphotref']['frame'],
        photrefinfo['combinedphotref']['jpeg'],

        masterphotref,
        photrefinfo['combinedphotref']['regfile'],
        photrefinfo['combinedphotref']['cmrawphot'],

        photref_frameinfo['zenithdist'],
        photref_frameinfo['moondist'],
        photref_frameinfo['moonelev'],

        photref_frameinfo['moonphase'],
        photref_frameinfo['hourangle'],
        photref_frameinfo['ngoodobjects'],

        photref_frameinfo['medmagerr'],
        photref_frameinfo['magerrmad'],
        photref_frameinfo['medsrcbgv'],

        photref_frameinfo['stdsrcbgv'],
        photref_frameinfo['medsval'],
        photref_frameinfo['meddval'],

        json.dumps(photrefinfo['combinedphotref'],ensure_ascii=True)
    )


    try:

        cur = db.cursor()
        cur.execute(query, params)
        db.commit()

        print('%sZ: will use combinedphotref %s for '
              'field %s, cam %s, ccd %s, project id %s, database updated.' %
              (datetime.utcnow().isoformat(),
               combinedphotref,
               masterphotrefinfo['field'],
               masterphotrefinfo['cam'],
               masterphotrefinfo['ccd'],
               masterphotrefinfo['projectid']))

    except Exception as e:

        print('ERR! %sZ: could not update refinfo DB! error was: %s' %
              (datetime.utcnow().isoformat(), e))
        db.rollback()

    db.close()

    # return the updated photrefinfo dict
    return photrefinfo



def get_combined_photref(projectid,
                         field,
                         ccd,
                         photreftype,
                         dbtype='postgres',
                         refinfo=REFINFO,
                         camera=0):
    """
    This gets the combined photref for the given combo of projid, field, ccd.

    Used for the convsubphot functions below.

    dbtype: 'postgres' or 'sqlite'. By default, 'postgres'.
    """

    assert (dbtype == 'postgres') or (dbtype == 'sqlite')

    if dbtype == 'sqlite':
        db = sqlite3.connect(
            refinfo,
            detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
        )
        query = (
            'select field,projectid,ccd,photreftype,unixtime,'
            'framepath,jpegpath,convolvetarget,convolveregpath,'
            'cmrawphotpath,target_zenithdist,target_moondist,'
            'target_moonelev,target_moonphase,target_hourangle,'
            'target_ndet,target_medmagerr,target_magerrmad,'
            'target_medsrcbgv,target_stdsrcbgv,target_medsval,'
            'target_meddval,photrefinfo from photrefs where '
            '(isactive = 1) and '
            '(projectid = ?) and '
            '(ccd = ?) and '
            '(field = ?) and '
            '(photreftype = ?)'
        )

    elif dbtype == 'postgres':
        db = pg.connect(user=PGUSER,
                        password=PGPASSWORD,
                        database=PGDATABASE,
                        host=PGHOST)
        query = (
            'select field,projectid,camera,ccd,photreftype,unixtime,'
            'framepath,jpegpath,convolvetarget,convolveregpath,'
            'cmrawphotpath,target_zenithdist,target_moondist,'
            'target_moonelev,target_moonphase,target_hourangle,'
            'target_ndet,target_medmagerr,target_magerrmad,'
            'target_medsrcbgv,target_stdsrcbgv,target_medsval,'
            'target_meddval,photrefinfo from photrefs where '
            '(isactive = 1) and '
            '(projectid = %s) and '
            '(camera = %s) and '
            '(ccd = %s) and '
            '(field = %s) and '
            '(photreftype = %s)'
        )

    params = (projectid, camera, ccd, field, photreftype)
    cur = db.cursor()

    try:

        cur.execute(query, params)
        rows = cur.fetchone()

        if rows is None:
            print('ERR! %sZ: No combinedphotref in database for '
                  'projectid = %s, camera = %s, ccd = %s, field = %s, '
                  'photreftype = %s' %
                  (datetime.utcnow().isoformat(), projectid,
                   camera, ccd, field, photreftype))
            db.close()
            return None

        cphotref = {x:y for (x,y) in zip(('field','projectid','camera','ccd',
                                          'photreftype','unixtime',
                                          'framepath','jpegpath',
                                          'convolvetarget','convolveregpath',
                                          'cmrawphotpath',
                                          'target_zenithdist',
                                          'target_moondist',
                                          'target_moonelev',
                                          'target_moonphase',
                                          'target_hourangle',
                                          'target_ndet',
                                          'target_medmagerr',
                                          'target_magerrmad',
                                          'target_medsrcbgv',
                                          'target_stdsrcbgv',
                                          'target_medsval',
                                          'target_meddval',
                                          'photrefinfo'),rows)}

        if not type(cphotref['photrefinfo']) == dict:
            # load the JSON string for photrefinfo. NB postgres does this
            # automatically.
            cphotref['photrefinfo'] = json.loads(cphotref['photrefinfo'])

        returnval = cphotref

    except Exception as e:

        print('ERR! %sZ: could not get combinedphotref info '
              'from DB! error was: %s' %
              (datetime.utcnow().isoformat(), e))
        returnval = None

        raise

    db.close()
    return returnval



#######################
## IMAGE SUBTRACTION ##
#######################

def xtrnsfits_convsub_worker(task, **kwargs):
    """This is a parallel worker for framelist_convsub_photref below.

    task[0] = xtrnsfits file
    task[1] = photreftype to use <"oneframe"|"onehour"|"onenight">
    task[2] = outdir
    task[3] = kernelspec
    task[4] = reversesubtract boolean
    task[5] = refinfo
    task[6] = observatory
    task[7] = fieldinfo (if observatory is tess, otherwise None)

    This only does convolution and subtraction. Finding new objects is handled
    by another function in transients.py, and doing photometry on the subtracted
    frames is handled by convsubfits_photometry_worker below.
    """

    (frame, photreftype, outdir,
     kernelspec, reversesubtract, refinfo, observatory, fieldinfo) = task

    try:

        # first, figure out the input frame's projid, field, and ccd
        if observatory=='hatpi':
            if fieldinfo is None:
                frameelems = get_header_keyword_list(frame, ['object', 'projid'])
                felems = FRAMEREGEX.findall(os.path.basename(frame))
                field, ccd, projectid = (frameelems['object'], int(felems[0][2]),
                                         frameelems['projid'])
                camera = 0
            else:
                field, ccd, projectid, camera = (fieldinfo['field'],
                                                 fieldinfo['ccd'],
                                                 fieldinfo['projectid'],
                                                 fieldinfo['camera'])
        elif observatory=='tess':
            camera = fieldinfo['camera']
            field = fieldinfo['field']
            projectid = fieldinfo['projectid']
            ccd = fieldinfo['ccd']

        # then, find the associated combined photref frame, regfile, cmrawphot
        cphotref = get_combined_photref(projectid, field, ccd, photreftype,
                                        refinfo=refinfo, camera=camera)
        cphotref_frame = cphotref['framepath']
        cphotref_reg = cphotref['convolveregpath']
        cphotref_cmrawphot = cphotref['cmrawphotpath']

        # do the subtraction (take care of reversesubtract here)
        _, convsub = ism.subframe_convolution_worker(
            (frame, cphotref_frame, cphotref_reg,
             kernelspec, outdir, reversesubtract, photreftype),
            **kwargs
        )

        if not (convsub and os.path.exists(convsub)):
            print('ERR! %sZ: convolution and subtraction failed on frame %s' %
                  (datetime.utcnow().isoformat(), frame))
            return frame, None

        else:

            return frame, convsub

    except Exception as e:

        print('ERR! %sZ: could not do convsubphot on frame %s, error was: %s' %
              (datetime.utcnow().isoformat(), frame, e))

        return frame, None



def parallel_xtrnsfits_convsub(xtrnsfits,
                               photreftype,
                               fitsdir=sv.REDPATH,
                               fitsglob=sv.LOCAL_GLOBPATTERN,
                               observatory='hatpi',
                               fieldinfo=None,
                               overwrite=False,
                               outdir=None,
                               refinfo=REFINFO,
                               reversesubtract=True,
                               kernelspec='b/4;i/4;d=4/4',
                               nworkers=16,
                               maxworkertasks=1000,
                               colorscheme=None):
    """
    This convolves, and subtracts all FITS files in the xtrnsfits list.

    WARNING: KNOWN BUG is that if reversesubtract is set to be false ("nsub"),
    you introduce a sign error in the resulting magnitudes.
    """

    # first, check if the convolved, subtracted frames already exist. if so,
    # and overwrite == False, then do not run them.

    existingcsframes = glob.glob(os.path.join(
        fitsdir, '[r|n]sub-????????-'+fitsglob.replace('.fits','-xtrns.fits')))

    if len(existingcsframes) > 0 and not overwrite:

        requested = list(map(os.path.basename, xtrnsfits))
        alreadyexists = list(map(os.path.basename, existingcsframes))

        # use lazy matching to substitute out the hash string
        alreadyexists = [re.sub('[r|n]sub-.*?-','',ae) for ae in alreadyexists]

        setdiff = np.setdiff1d(requested, alreadyexists)

        if len(setdiff) == 0:

            print('WRN! %sZ: every requested frame already found to exist.' %
                  (datetime.utcnow().isoformat(),))
            print('WRN! %sZ: skipping convolution & subtraction step.' %
                  (datetime.utcnow().isoformat(),))
            return

        else:

            xtrnsfits = [fitsdir+sd for sd in setdiff]

    tasks = [(x, photreftype, outdir, kernelspec,
              reversesubtract, refinfo, observatory, fieldinfo)
             for x in xtrnsfits if os.path.exists(x)]

    print('%sZ: %s files to convolve and subtract' %
          (datetime.utcnow().isoformat(), len(tasks)))

    if len(tasks) > 0:

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        # fire up the pool of workers
        kwargs = {'colorscheme':colorscheme}
        results = pool.map(
            partial(xtrnsfits_convsub_worker, **kwargs), tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}

    else:

        print('ERR! %sZ: none of the files specified exist, bailing out...' %
              (datetime.utcnow().isoformat(),))
        return



#########################################
## PHOTOMETRY ON THE SUBTRACTED FRAMES ##
#########################################

def convsubfits_staticphot_worker(task):
    """
    This does subtracted frame photometry on already-known objects from the
    photref.

    task[0] = subframe
    task[1] = photreftype
    task[2] = kernelspec
    task[3] = lcapertures
    task[4] = disjointradius
    task[5] = outdir
    task[6] = refinfo
    task[7] = observatory
    task[8] = fieldinfo
    task[9] = photparams
    task[10] = domagfit option

    currently produces iphot files.

    TODO: should this write to the database?
    """

    (subframe, photreftype, kernelspec,
     lcapertures, disjrad, outdir, refinfo,
     observatory, fieldinfo, photparams, domagfit) = task

    try:

        # generate the convsubfits hash
        convsubhash = ism.get_convsubfits_hash(
            photreftype,
            ('reverse' if os.path.basename(subframe).startswith('rsub')
             else 'normal'),
            kernelspec
        )

        if observatory=='hatpi':
            frameinfo = FRAMEREGEX.findall(os.path.basename(subframe))
            if fieldinfo is None:
                # first, figure out the input frame's projid, field, and ccd
                frameelems = get_header_keyword_list(subframe, ['object', 'projid'])
                field, ccd, projectid = (frameelems['object'], int(frameinfo[0][2]),
                                         frameelems['projid'])
                camera = 0
            else:
                field = fieldinfo['field']
                ccd = fieldinfo['ccd']
                projectid = fieldinfo['projectid']
                camera = fieldinfo['camera']

        elif observatory=='tess':
            field = fieldinfo['field']
            ccd = fieldinfo['ccd']
            projectid = fieldinfo['projectid']
            camera = fieldinfo['camera']

        # then, find the associated combined photref frame, regfile, cmrawphot
        cphotref = get_combined_photref(projectid, field, ccd, photreftype,
                                        refinfo=refinfo, camera=camera)
        cphotref_frame = cphotref['framepath']
        cphotref_reg = cphotref['convolveregpath']
        cphotref_cmrawphot = cphotref['cmrawphotpath']

        # find matching kernel, itrans, and xysdk files for each subtracted
        # frame

        photrefbit = (
            'rsub' if os.path.basename(subframe).startswith('rsub') else 'nsub'
        )

        if observatory=='hatpi':

            kernelf = '%s-%s-%s-%s_%s-xtrns.fits-kernel' % (photrefbit,
                                                            convsubhash,
                                                            frameinfo[0][0],
                                                            frameinfo[0][1],
                                                            frameinfo[0][2])
            kernel = os.path.abspath(os.path.join(os.path.dirname(subframe),kernelf))

            itransf = '%s-%s_%s.itrans' % (frameinfo[0][0],
                                           frameinfo[0][1],
                                           frameinfo[0][2])
            itrans = os.path.abspath(os.path.join(os.path.dirname(subframe),itransf))

            xysdkf = '%s-%s_%s.xysdk' % (frameinfo[0][0],
                                         frameinfo[0][1],
                                         frameinfo[0][2])
            xysdk = os.path.abspath(os.path.join(os.path.dirname(subframe),xysdkf))

        if observatory=='tess':

            namesub = re.findall('tess20.*?-[0-9][0-9][0-9][0-9]_cal_img_bkgdsub', subframe)
            if not len(namesub) == 1:
                raise AssertionError(
                    'expected only one subframe, got {:s}'.
                    format(repr(namesub)))
            namesub = namesub[0]

            kernelf = '%s-%s-%s-xtrns.fits-kernel' % (photrefbit, convsubhash,
                                                      namesub)
            kernel = os.path.abspath(os.path.join(os.path.dirname(subframe),kernelf))

            itransf = '%s.itrans' % (namesub)
            itrans = os.path.abspath(os.path.join(os.path.dirname(subframe),itransf))

            xysdkf = '%s.xysdk' % (namesub)
            xysdk = os.path.abspath(os.path.join(os.path.dirname(subframe),xysdkf))

        # write the photometry file to /dev/shm by default
        # if outdir is None:
        #     outdir = '/dev/shm'

        _, subphot = ism.subframe_photometry_worker(
            (subframe, cphotref_cmrawphot, disjrad,
             kernel, itrans, xysdk, outdir,
             photreftype, kernelspec, lcapertures, observatory, photparams,
             domagfit)
        )

        if subphot and os.path.exists(subphot):

            print('%sZ: CONVSUBPHOT (STATIC) OK: '
                  'subtracted frame %s, photometry file %s' %
                  (datetime.utcnow().isoformat(), subframe, subphot))

            return subframe, subphot

        else:

            print('%sZ: CONVSUBPHOT (STATIC) FAILED: subtracted frame %s' %
                  (datetime.utcnow().isoformat(), subframe))

            return subframe, None


    except Exception as e:

        message = ('could not do CONVSUBPHOT (STATIC) for %s, '
                   'exception follows' % subframe)
        print('EXC! %sZ: %s\n%s' %
               (datetime.utcnow().isoformat(), message, format_exc()) )

        return subframe, None



def parallel_convsubfits_staticphot(
        subfitslist,
        fitsdir=sv.REDPATH,
        fitsglob=sv.LOCAL_GLOBPATTERN,
        photreftype='oneframe',
        kernelspec='b/4;i/4;d=4/4',
        lcapertures='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
        photdisjointradius=2,
        outdir=None,
        refinfo=REFINFO,
        nworkers=16,
        maxworkertasks=1000,
        observatory='hatpi',
        fieldinfo=None,
        overwrite=False,
        photparams=None,
        domagfit=False):
    """
    This does static object photometry on the all subtracted FITS in
    subfitslist.
    """

    # check if the convolved, subtracted frames already have photometry. if so,
    # and overwrite == False, then do not re-run photometry.

    existingiphot = glob.glob(os.path.join(
        fitsdir,'[r|n]sub-????????-'+ fitsglob.replace('.fits','.iphot')))

    if len(existingiphot) > 0 and not overwrite:

        requested = list(map(os.path.basename, subfitslist))
        alreadyexists = list(map(os.path.basename, existingiphot))

        # substitute out the hash string
        alreadyexists = [re.sub('[r|n]sub-.*?-','',ae).replace('.iphot','') for
                         ae in alreadyexists]
        requested = [re.sub('[r|n]sub-.*?-','',r).replace('-xtrns.fits','') for
                     r in requested]

        setdiff = np.setdiff1d(requested, alreadyexists)

        if len(setdiff) == 0:
            print('WRN! %sZ: photometry already done on requested frames.' %
                  (datetime.utcnow().isoformat(),))
            return

        else:
            subfitslist = [fitsdir+sd+'.iphot' for sd in setdiff]

    tasks = [(x, photreftype, kernelspec, lcapertures, photdisjointradius,
              outdir, refinfo, observatory, fieldinfo, photparams, domagfit)
             for x in subfitslist if os.path.exists(x)]

    if len(tasks) > 0:

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        # fire up the pool of workers
        results = pool.map(convsubfits_staticphot_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}

    else:

        print('ERR! {:s}Z:'.format(datetime.utcnow().isoformat()) + ' the '+
              'files that reached convsubfits_staticphot_worker do not exist.'+
              ' bailing out...' )
        return 42



##########################################
## SQLITE3 STYLE PHOT INDEX DB IN PGSQL ##
##########################################

PHOT_SELECT_QUERY = ("select rjd, framekey, photline from "
                     "photindex_iphots where objectid = %s order by rjd")
CSTORE_SELECT_QUERY = ("select rjd, framekey, photline from "
                       "iphots_cstore where objectid = %s order by rjd")


def insert_phots_into_database(framedir,
                               frameglob='rsub-*-xtrns.fits',
                               photdir=None,
                               photglob='rsub-*-%s.iphot',
                               maxframes=None,
                               overwrite=False,
                               database=None):
    """
    This makes photometry index rows in the postgresql database.  Intended for
    use when the sqlite3 databases get out of hand.
    """

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        cursor = database.cursor()
        closedb = True


    # first, figure out the directories
    if not photdir:
        photdir = framedir


    # start work here
    try:

        if isinstance(framedir, list):
            framelist = framedir
        else:
            # first, find all the frames
            framelist = glob.glob(os.path.join(os.path.abspath(framedir),
                                               frameglob))

        # restrict to maxframes max frames
        if maxframes:
            framelist = framelist[:maxframes]


        # turn off table logging and drop indexes for speed
        cursor.execute('drop index if exists photindex_iphots_rjd_idx')
        cursor.execute('drop index if exists photindex_iphots_objectid_idx')

        starttime = time.time()

        # go through all the frames
        for ix, frame in enumerate(framelist):

            print('%sZ: inserting %d frame %s into pg database' %
                  (datetime.utcnow().isoformat(), ix, frame))

            # generate the names of the associated phot and sourcelist files
            frameinfo = FRAMEREGEX.findall(os.path.basename(frame))
            framekey = '%s-%s_%s' % (frameinfo[0][0],
                                     frameinfo[0][1],
                                     frameinfo[0][2])

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

                # NOTE: this is the ORIGINAL FITS frame, since the subtracted
                # one contains some weird JD header (probably inherited from the
                # photref frame)
                framerjd = get_header_keyword(originalframe, 'JD')

                # now get the phot file and read it
                photf = open(phot, 'rb')
                photo = StringIO()

                for line in photf:
                    hatid = line.split()[0]
                    photo.write('%.5f,%s,%s,%s' % (framerjd,
                                                   hatid,
                                                   framekey,
                                                   line))
                photf.close()
                photo.seek(0)


                # do a fast insert using pg's copy protocol
                cursor.copy_from(photo,'photindex_iphots',sep=',')
                photo.close()

            # if some associated files don't exist for this frame, ignore it
            else:

                print('WRN! %sZ: ignoring frame %s, '
                      'photometry for this frame is not available!' %
                      (datetime.utcnow().isoformat(), frame))

        # now we're all done with frame inserts

        # regenerate the indexes and reset table logging for durability
        print('%sZ: recreating indexes' % (datetime.utcnow().isoformat()))
        cursor.execute('create index on photindex_iphots(rjd)')
        cursor.execute('create index on photindex_iphots(objectid)')
        cursor.execute('analyze photindex_iphots')

        # commit the transaction
        database.commit()
        print('%sZ: done, time taken: %.2f minutes' %
              (datetime.utcnow().isoformat(), (time.time() - starttime)/60.0))

        returnval = (framedir, True)

    # catch the overwrite = False scenario
    except pg.IntegrityError as e:

        database.rollback()

        message = ('failed to insert photometry from %s '
                   'into DB because some of it exists already '
                   'and overwrite = False'
                   % framedir)
        print('EXC! %sZ: %s\n%s' %
               (datetime.utcnow().isoformat(), message, format_exc()) )
        returnval = (framedir, False)


    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to insert photometry from %s into DB' % framedir
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (framedir, False)


    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def insert_phots_into_cstore(framedir,
                             frameglob='rsub-*-xtrns.fits',
                             photdir=None,
                             photglob='rsub-*-%s.iphot',
                             maxframes=None,
                             overwrite=False,
                             database=None):
    """
    This makes photometry index rows in the postgresql database. This was an
    attempt to use the postgres column store extension to speed up ingestion of
    iphots. It did not speed it up.
    """

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        cursor = database.cursor()
        closedb = True


    # first, figure out the directories
    if not photdir:
        photdir = framedir


    # start work here
    try:

        if isinstance(framedir, list):
            framelist = framedir
        else:
            # first, find all the frames
            framelist = glob.glob(os.path.join(os.path.abspath(framedir),
                                               frameglob))

        # restrict to maxframes max frames
        if maxframes:
            framelist = framelist[:maxframes]


        starttime = time.time()

        # go through all the frames
        for frame in framelist:

            print('%sZ: working on frame %s' %
                  (datetime.utcnow().isoformat(), frame))

            # generate the names of the associated phot and sourcelist files
            frameinfo = FRAMEREGEX.findall(os.path.basename(frame))
            framekey = '%s-%s_%s' % (frameinfo[0][0],
                                     frameinfo[0][1],
                                     frameinfo[0][2])

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

                # NOTE: this is the ORIGINAL FITS frame, since the subtracted
                # one contains some weird JD header (probably inherited from the
                # photref frame)
                framerjd = get_header_keyword(originalframe, 'JD')

                # now get the phot file and read it
                photf = open(phot, 'rb')
                photo = StringIO()

                # write to the output cstringio to generate CSVs in the format
                # we want
                for line in photf:
                    hatid = line.split()[0]
                    photo.write('%.5f,%s,%s,%s' % (framerjd,
                                                   hatid,
                                                   framekey,
                                                   line))
                photf.close()
                photo.seek(0)


                # do a fast insert using pg's copy protocol
                cursor.copy_from(photo,'iphots_cstore',sep=',')
                photo.close()

            # if some associated files don't exist for this frame, ignore it
            else:

                print('WRN! %sZ: ignoring frame %s, '
                      'photometry for this frame is not available!' %
                      (datetime.utcnow().isoformat(), frame))

        # now we're all done with frame inserts
        cursor.execute('analyze iphots_cstore')

        # commit the transaction
        database.commit()
        print('%sZ: done, time taken: %.2f minutes' %
              (datetime.utcnow().isoformat(), (time.time() - starttime)/60.0))

        returnval = (framedir, True)

    # catch the overwrite = False scenario
    except pg.IntegrityError as e:

        database.rollback()

        message = ('failed to insert photometry from %s '
                   'into DB because some of it exists already '
                   'and overwrite = False'
                   % framedir)
        print('EXC! %sZ: %s\n%s' %
               (datetime.utcnow().isoformat(), message, format_exc()) )
        returnval = (framedir, False)


    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to insert photometry from %s into DB' % framedir
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (framedir, False)


    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def cstore_collect_imagesubphot_lightcurve(
        hatid,
        outdir,
        skipcollected=True,
        mindetections=50,
        database=None):
    """
    This collects an ISM LC using the cstore tables in Postgres.  This makes
    photometry index rows in the postgresql database. This was an attempt to
    use the postgres column store extension to speed up ingestion of iphots. It
    did not speed it up.
    """

    # prepare the output file
    outfile = os.path.join(os.path.abspath(outdir), '%s.ilc' % hatid)

    # if the file already exists and skipcollected is True, then return
    # that file instead of processing any further
    if os.path.exists(outfile) and skipcollected:

        print('WRN! %sZ: object %s LC already exists, '
              'not overwriting: %s' %
              (datetime.utcnow().isoformat(), hatid, outfile))

        return hatid, outfile


    # open a database connection otherwise
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        # this is a readonly query so we don't need a transaction
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    # start the collection process
    try:

        # find the photometry in the database for this hatid
        starttime = time.time()
        cursor.execute(CSTORE_SELECT_QUERY, (hatid,))
        rows = cursor.fetchall()
        print('lookup complete in %.2f seconds' % (time.time() - starttime))

        # make sure we have enough rows to make it worth our while
        if rows and len(rows) >= mindetections:

            outf = open(outfile, 'wb')

            # go through the phots and sourcelists, picking out the
            # timeseries information for this hatid
            for row in rows:

                # unpack the row to get our values
                framerjd, framekey, photline = row
                out_line = '%s %s %s\n' % (framerjd, framekey, photline)
                outf.write(out_line.encode('utf-8'))

            # close the output LC once we're done with it
            outf.close()

            print('%sZ: object %s -> %s' %
                  (datetime.utcnow().isoformat(), hatid, outfile))

            # this is the return val
            returnval = (hatid, outfile)

        # if we don't have enough detections, ignore this light curve
        else:

            print('ERR! %sZ: object %s: %s dets < %s mindetections,'
                  ' ignoring...' %
                  (datetime.utcnow().isoformat(), hatid,
                   len(rows), mindetections))
            returnval = (hatid, None)

    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to get photometry for %s from DB' % hatid
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (hatid, None)

    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def dbphot_collect_imagesubphot_lightcurve(hatid,
                                           outdir,
                                           skipcollected=True,
                                           mindetections=50,
                                           database=None):
    """
    This collects an ISM LC using the photindex info in Postgres.
    """

    # prepare the output file
    outfile = os.path.join(os.path.abspath(outdir), '%s.ilc' % hatid)

    # if the file already exists and skipcollected is True, then return
    # that file instead of processing any further
    if os.path.exists(outfile) and skipcollected:

        print('WRN! %sZ: object %s LC already exists, '
              'not overwriting: %s' %
              (datetime.utcnow().isoformat(), hatid, outfile))

        return hatid, outfile


    # open a database connection otherwise
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        # this is a readonly query so we don't need a transaction
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    # start the collection process
    try:

        # find the photometry in the database for this hatid
        starttime = time.time()
        cursor.execute(PHOT_SELECT_QUERY, (hatid,))
        rows = cursor.fetchall()
        print('lookup complete in %.2f seconds' % (time.time() - starttime))

        # make sure we have enough rows to make it worth our while
        if rows and len(rows) >= mindetections:

            outf = open(outfile, 'wb')

            # go through the phots and sourcelists, picking out the
            # timeseries information for this hatid
            for row in rows:

                try:

                    # unpack the row to get our values
                    framerjd, phot, photline = row

                    # generate the framekey
                    rstfc_elems = FRAMEREGEX.findall(
                        os.path.basename(phot)
                    )
                    rstfc = '%s-%s_%s' % (rstfc_elems[0])
                    out_line = '%s %s %s\n' % (framerjd, rstfc, photline)
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

            # this is the return val
            returnval = (hatid, outfile)

        # if we don't have enough detections, ignore this light curve
        else:

            print('ERR! %sZ: object %s: %s dets < %s mindetections,'
                  ' ignoring...' %
                  (datetime.utcnow().isoformat(), hatid,
                   len(rows), mindetections))
            returnval = (hatid, None)

    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to get photometry for %s from DB' % hatid
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (hatid, None)

    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def get_hatidlist_from_cmrawphot(projectid, field, ccd, photreftype,
                                 refinfo=REFINFO):
    """
    This gets the hatidlist from the cmrawphot of a combined photref.
    """

    cphotref = get_combined_photref(projectid, field, ccd, photreftype,
                                    refinfo=refinfo)

    cmrawphot = cphotref['cmrawphotpath']

    if os.path.exists(cmrawphot):

        hatidlist = []

        with open(cmrawphot,'rb') as infd:
            for line in infd:
                if not line.startswith('#'):
                    hatid = line.split()[0]
                    hatidlist.append(hatid)

        print('%sZ: %s objects found in cmrawphot: %s' %
              (datetime.utcnow().isoformat(), len(hatidlist), cmrawphot))

    else:

        print('ERR! %sZ: expected cmrawphot does not exist!' %
              (datetime.utcnow().isoformat(), ))


    return hatidlist



def parallel_dbphot_collect_worker(task):
    """
    This is the parallel worker for the function below.

    task[0] = hatid
    task[1] = outdir
    task[2] = skipcollected
    task[3] = mindetections
    """

    hatid, outdir, skipcollected, mindetections = task
    return dbphot_collect_imagesubphot_lightcurve(hatid,
                                                  outdir,
                                                  skipcollected=skipcollected,
                                                  mindetections=mindetections)



def parallel_dbphot_lightcurves_hatidlist(hatidlist,
                                          outdir,
                                          skipcollectedlcs=True,
                                          mindetections=50,
                                          nworkers=24,
                                          maxworkertasks=1000):
    """
    This collects light curves for the provided list of hatids.

    Get hatidlist by reading in the .cmrawphot file for a projectid-ccd
    combination using get_hatidlist_from_cmrawphot. In the future, we'll add
    these to the database.
    """

    # first, check if the output directory exists
    if not os.path.exists(outdir):
        os.mkdir(outdir)


    # generate the list of tasks
    tasks = [(x, outdir, skipcollectedlcs, mindetections) for x in
             hatidlist]

    # now start up the parallel collection
    print('%sZ: %s HATIDs to get LCs for, starting...' %
          (datetime.utcnow().isoformat(), len(hatidlist), ))
    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(parallel_dbphot_collect_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    return {x:y for (x,y) in results}



def parallel_dbphot_lightcurves_projectid(projectid,
                                          field,
                                          ccd,
                                          outdir,
                                          photreftype='oneframe',
                                          refinfo=REFINFO,
                                          skipcollectedlcs=True,
                                          mindetections=50,
                                          nworkers=24,
                                          maxworkertasks=1000):
    """
    This collects LCs for specific projectids.
    """

    hatidlist = get_hatidlist_from_cmrawphot(projectid,
                                             field,
                                             ccd,
                                             photreftype,
                                             refinfo=refinfo)

    if hatidlist:
        return parallel_dbphot_lightcurves_hatidlist(
            hatidlist,
            outdir,
            skipcollectedlcs=skipcollectedlcs,
            mindetections=mindetections,
            nworkers=nworkers,
            maxworkertasks=maxworkertasks
        )

    else:
        print('ERR! no hatids found for project: %s' % (repr([projectid,
                                                              field,
                                                              ccd,
                                                              photreftype])))
        return None




#########################
## PHOTOMETRY DATABASE ##
#########################

def parse_iphot_line(iphotline):

    """
    This parses the iphot line and returns a row of formatted elems.

    These can be then attached to other metadata and written directly to the
    database.
    """

    photelemline = iphotline.rstrip(' \n')
    photelem = photelemline.split()

    if len(photelem) > 0:

        # get the station, framenumber, subframe, and ccd
        framekeyelems = FRAMESUBREGEX.findall(
            os.path.basename(iphot)
        )
        stf, cfn, cfs, fccd = (
            framekeyelems[0][0],
            framekeyelems[0][1],
            framekeyelems[0][2],
            framekeyelems[0][3]
        )
        net = 'HP' # this will be 'HP' for HATPI

        # these are things we get from the iphot line
        # hat, xcc, ycc, xic, yic
        # fsv, fdv, fkv, bgv, bge
        # ifl1, ife1, irm1, ire1, irq1
        # ifl2, ife2, irm2, ire2, irq2
        # ifl2, ife2, irm2, ire2, irq2

        objectid, xcc, ycc, xic, yic = photelem[:5]
        fsv, fdv, fkv, bgv, bge = photelem[5:10]
        ifl1, ife1, irm1, ire1, irq1 = photelem[10:15]
        ifl2, ife2, irm2, ire2, irq2 = photelem[15:20]
        ifl3, ife3, irm3, ire3, irq3 = photelem[20:]

        # format these as needed
        xcc = smartcast(xcc, float)
        ycc = smartcast(ycc, float)
        xic = smartcast(xic, float)
        yic = smartcast(yic, float)

        fsv = smartcast(fsv, float)
        fdv = smartcast(fdv, float)
        fkv = smartcast(fkv, float)
        bgv = smartcast(bgv, float)
        bge = smartcast(bge, float)

        ifl_000 = smartcast(ifl1, float)
        ife_000 = smartcast(ife1, float)
        irm_000 = smartcast(irm1, float)
        ire_000 = smartcast(ire1, float)
        irq_000 = smartcast(irq1, str)

        ifl_001 = smartcast(ifl2, float)
        ife_001 = smartcast(ife2, float)
        irm_001 = smartcast(irm2, float)
        ire_001 = smartcast(ire2, float)
        irq_001 = smartcast(irq2, str)

        ifl_002 = smartcast(ifl3, float)
        ife_002 = smartcast(ife3, float)
        irm_002 = smartcast(irm3, float)
        ire_002 = smartcast(ire3, float)
        irq_002 = smartcast(irq3, str)

        return {'objectid':objectid,
                'xcc':xcc,
                'ycc':ycc,
                'xic':xic,
                'yic':yic,
                'fsv':fsv,
                'fdv':fdv,
                'fkv':fkv,
                'bgv':bgv,
                'bge':bge,
                'ifl_000':ifl_000,
                'ife_000':ife_000,
                'irm_000':irm_000,
                'ire_000':ire_000,
                'irq_000':irq_000,
                'ifl_001':ifl_001,
                'ife_001':ife_001,
                'irm_001':irm_001,
                'ire_001':ire_001,
                'irq_001':irq_001,
                'ifl_002':ifl_002,
                'ife_002':ife_002,
                'irm_002':irm_002,
                'ire_002':ire_002,
                'irq_002':irq_002}

    else:

        return None




def convsub_photometry_to_ismphot_database(convsubfits,
                                           convsubphot,
                                           photreftype,
                                           subtracttype,
                                           kernelspec,
                                           lcapertures,
                                           projectid=None,
                                           field=None,
                                           ccd=None,
                                           overwrite=False,
                                           database=None):

    """
    This inserts the ISM photometry from a single convsub FITS into the DB.

    If projectid, field, ccd are not provided, gets them from the FITS
    file. Also gets the photreftype from the filename of the
    convolved-subtracted photometry iphot file.
    """

    # open a database connection
    if database:
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        cursor = database.cursor()
        closedb = True

    # start work here
    try:

        # figure out the projectid, field, ccd, photreftype
        # first, figure out the input frame's projid, field, and ccd
        frameelems = get_header_keyword_list(convsubfits,
                                             ['object',
                                              'projid'])
        felems = FRAMEREGEX.findall(
            os.path.basename(convsubfits)
        )

        if not (projectid and field and ccd):

            field, ccd, projectid = (frameelems['object'],
                                     int(felems[0][2]),
                                     frameelems['projid'])

        convsubdir = os.path.abspath(os.path.dirname(convsubfits))

        if not os.path.exists(iphotpath):
            print('ERR! %sZ: expected iphot %s for '
                  'convsub FITS %s does not exist, '
                  'not processing...' %
                  (datetime.utcnow().isoformat(), iphotpath, convsubfits))
            return (convsubfits, False)

        # find the frame's original FITS file (unsubtracted calibrated frame)
        originalfitsbasename = '%s-%s_%s.fits' % (felems[0][0],
                                                  felems[0][1],
                                                  felems[0][2])
        originalfitspath = os.path.join(convsubdir, originalfitsbasename)

        if not os.path.exists(originalfitspath):
            print('%ERR! sZ: expected original FITS %s '
                  'for convsub FITS %s does not exist, '
                  'not processing...' %
                  (datetime.utcnow().isoformat(),
                   originalfitspath, convsubfits))
            return (convsubfits, False)

        # figure out the frame's info from the original frame's header
        framerjd = get_header_keyword_list(originalfitspath,
                                           ['JD',''])

        # also get some metadata from the frameheader


        # now open the accompanying iphot file, and stream the photometry to the
        # database
        with open(convsubphot,'rb') as infd:

            # prepare the statement
            query = ("insert into ism_photometry ("
                     "frameutcdt, objectid, framekey, photkey, "
                     "xcc, ycc, xic, yic, bgv, bge, fsv, fdv, fkv, "
                     "ifl_000, ife_000, irm_000, ire_000, irq_000, "
                     "ifl_001, ife_001, irm_001, ire_001, irq_001, "
                     "ifl_002, ife_002, irm_002, ire_002, irq_002"
                     ") values ("
                     "%s, %s, %s, %s, "
                     "%s, %s, %s, %s, %s, %s, %s, %s, %s, "
                     "%s, %s, %s, %s, %s, %s, "
                     "%s, %s, %s, %s, %s, %s, "
                     "%s, %s, %s, %s, %s, %s"
                     ")")

            # prepare the input params
            # TODO: finish this


            for line in infd:

                parsedline = parse_iphot_line(line)


        # update the iphotfiles table file with all of this info. if there's a
        # uniqueness conflict, i.e. this same combination exists, then overwrite
        # if told to do so
        if overwrite:

            print('WRN! %sZ: overwriting existing photometry info in DB for %s'
                  %
                  (datetime.utcnow().isoformat(), convsubfits))

            query = ("insert into iphotfiles "
                     "(projectid, field, ccd, photreftype, convsubtype, "
                     "isactive, iphotfilepath, framerjd, framefilepath) "
                     "values ("
                     "%s, %s, %s, %s, %s, "
                     "%s, %s, %s, %s"
                     ") on conflict on constraint iphotfiles_pkey "
                     "do update "
                     "set projectid = %s, field = %s, ccd = %s, "
                     "photreftype = %s, convsubtype = %s, "
                     "isactive = %s, iphotfilepath = %s, framerjd = %s, "
                     "framefilepath = %s, entrytimestamp = current_timestamp")

            params = (projectid, field, ccd, photreftype, subtractiontype,
                      True, iphotpath, framerjd, originalfitspath,
                      projectid, field, ccd, photreftype, subtractiontype,
                      True, iphotpath, framerjd, originalfitspath)

        else:

            query = ("insert into iphotfiles "
                     "(projectid, field, ccd, photreftype, convsubtype, "
                     "isactive, iphotfilepath, framerjd, framefilepath) "
                     "values ("
                     "%s, %s, %s, %s, %s, "
                     "%s, %s, %s, %s"
                     ")")
            params = (projectid, field, ccd, photreftype, subtractiontype,
                      True, iphotpath, framerjd, originalfitspath)


        # execute the query to insert the object
        cursor.execute(query, params)
        database.commit()

        # update the iphotobjects table with all of these objects. if there's a
        # uniqueness conflict, i.e. this same combination exists, then overwrite
        # if told to do so

        if overwrite:

            query = ("insert into iphotobjects "
                     "(projectid, field, ccd, photreftype, convsubtype, "
                     "isactive, objectid, iphotfilepath, iphotfileline) "
                     "values ("
                     "%s, %s, %s, %s, %s, "
                     "%s, %s, %s, %s"
                     ") on conflict on constraint iphotobjects_pkey "
                     "do update set "
                     "projectid = %s, field = %s, ccd = %s, photreftype = %s, "
                     "convsubtype = %s, isactive = %s, objectid = %s, "
                     "iphotfilepath = %s, iphotfileline = %s, "
                     "entrytimestamp = current_timestamp")

        else:

            query = ("insert into iphotobjects "
                     "(projectid, field, ccd, photreftype, convsubtype, "
                     "isactive, objectid, iphotfilepath, iphotfileline) "
                     "values ("
                     "%s, %s, %s, %s, %s, "
                     "%s, %s, %s, %s"
                     ")")

        # execute statements for all of the iphot objects
        for ind, objectid in enumerate(iphotobjects):

            if overwrite:
                params = (projectid, field, ccd, photreftype, subtractiontype,
                          True, objectid, iphotpath, ind,
                          projectid, field, ccd, photreftype, subtractiontype,
                          True, objectid, iphotpath, ind,)
            else:
                params = (projectid, field, ccd, photreftype, subtractiontype,
                          True, objectid, iphotpath, ind)

            cursor.execute(query, params)

        database.commit()

        print('%sZ: convsub FITS %s with iphot %s and %s objects '
              'inserted into DB OK' %
              (datetime.utcnow().isoformat(),
               convsubfits,
               iphotpath,
               len(iphotobjects)) )

        # return True if everything succeeded
        returnval = (convsubfits, True)


    # catch the overwrite = False scenario
    except pg.IntegrityError as e:

        database.rollback()

        message = ('failed to insert photometry from %s '
                   'into DB because it exists already '
                   'and overwrite = False'
                   % convsubfits)
        print('EXC! %sZ: %s\n%s' %
               (datetime.utcnow().isoformat(), message, format_exc()) )
        returnval = (convsubfits, False)


    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to insert photometry from %s into DB' % convsubfits
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (convsubfits, False)
        raise


    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval



def parallel_convsubphotdb_worker(task):
    """
    This wraps the function above for use with the parallel driver below.

    task[0] = convsubfits
    task[1] = {'projectid', 'field', 'ccd', 'overwrite'}
    """

    convsubfits = task[0]
    kwargs = task[1]

    return convsub_photometry_to_ismphot_databse(convsubphots,**kwargs)



def parallel_convsubphot_to_db(convsubfitslist,
                               projectid=None,
                               field=None,
                               ccd=None,
                               overwrite=False,
                               nworkers=16,
                               maxworkertasks=1000):
    """
    This runs a convsubphot ingest in parallel.
    """

    tasks = [(x, {'projectid':projectid, 'field':field,
                  'ccd':ccd, 'overwrite':overwrite})
             for x in convsubfitslist if os.path.exists(x)]

    print('%sZ: %s files to process' %
          (datetime.utcnow().isoformat(), len(tasks)))

    if len(tasks) > 0:

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)


        # fire up the pool of workers
        results = pool.map(parallel_convsubphotdb_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}

    else:

        print('ERR! %sZ: none of the files specified exist, bailing out...' %
              (datetime.utcnow().isoformat(),))
        return




############################
## LIGHT CURVE PRODUCTION ##
############################

# we'll make hatlc.sqlite type files, collecting them in /P/LC, under the
# following organization:
# {primary_field}/{hatid}-DR{datarelease}-V{lcversion}.hatlc.sqlite(.gz)
# we'll collect all photometry across observed fields and CCDs in the same file

def collect_lightcurve(objectid,
                       datarelease,
                       lcversion,
                       objectfield=None,
                       lcbasedir=LCBASEDIR,
                       updateifexists=True,
                       database=None):
    """
    This collects lightcurves for objectid into a hatlc.sqlite file.

    We'll collect all photometry across observed fields and CCDs in the same
    file, organized as below:

    {LCBASEDIR} / DR{datarelease} / {primary_field} /
    {hatid}-DR{datarelease}-V{lcversion}.hatlc.sqlite

    If an LC doesn't exist for this objectid/datarelease/lcversion, then creates
    a new sqlitecurve. If one already exists and updateifexists is True, updates
    the light curve with new observations.

    FIXME: if we're going to run realtime imagesubphot, this will require that a
    collect_lightcurve is run after every image is taken. This will probably be
    stupidly slow...
    """

    # open a database connection
    if database:
        database.autocommit = True # this is a readonly connection
        cursor = database.cursor()
        closedb = False
    else:
        database = pg.connect(user=PGUSER,
                              password=PGPASSWORD,
                              database=PGDATABASE,
                              host=PGHOST)
        database.autocommit = True
        cursor = database.cursor()
        closedb = True

    # start work here
    try:

        # find the photometry for the specified objectid
        query = ("select a.projectid, a.field, a.ccd, a.photreftype, "
                 "a.convsubtype, a.iphotfilepath, a.framefilepath, "
                 "a.framerjd, b.iphotfileline "
                 "from iphotfiles a join iphotobjects b on "
                 "(a.iphotfilepath = b.iphotfilepath) "
                 "where "
                 "(isactive = true) and "
                 "(b.objectid = %s) order by a.framerjd asc")
        params = (objectid, )

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # if there are any rows, then there are some detections of this object
        if rows and len(rows) > 0:

            if HATIDREGEX.match(objectid):
                primary_field = objectid.split('-')[1]
            elif objectfield:
                primary_field = objectfield
            else:
                primary_field = 'nofield'

            # prepare the output directory
            lcdir = os.path.join(LCBASEDIR,
                                 'DR%s' % datarelease,
                                 '%s' % primary_field)

            # make the light curve directory if it doesn't exist
            if not os.path.exists(lcdir):
                os.mkdirs(lcdir)

            # this is the path to the light curve file itself
            lcfbasename = '{objectid}-DR{datarelease}-V{lcversion}-hatlc.sqlite'
            lcfpath = os.path.join(lcdir, lcfbasename)

            # if it doesn't exist, we can make a new file
            if not os.path.exists(lcfpath):

                sqlcdb = sqlite3.connect(lcfpath)
                sqlcur = sqlcdb.cursor()

                # read in the hatlc.sql file and execute it to generate the
                # tables for the hatlc
                hatlcsqlf = os.path.join(os.path.dirname(__file__),
                                         'hatlc.sql')
                with open(hatlcsqlf,'rb') as infd:
                    hatlcsql = infd.read()

                sqlcur.executescript(hatlcsql)
                # the lightcurve is now ready to use, but has no information in
                # it, we'll add this at a later step


            # check if the path to it exists and if updateifexisting is True
            # if so, we can update it
            if updateifexisting:

                print('WRN! %sZ: objectid %s has an '
                      'existing LC %s, updating...' %
                      (datetime.utcnow().isoformat(), objectid, lcfpath) )

            # if we're not allowed to update it, fail out
            else:

                print('ERR! %sZ: objectid %s has an existing LC %s '
                      'and updateifexisting = False, skipping this object...' %
                      (datetime.utcnow().isoformat(), objectid, lcfpath) )
                return objectid, None

            # now read in the iphot files one-by-one and write their photometry
            # lines to the sqlite
            for row in rows:

                (projectid, ofield, ccd, photreftype,
                 convsubtype, iphot, frame, rjd, iphotline) = row

                # open the iphot file
                with open(iphot,'rb') as infd:
                    for ind, line in enumerate(infd):
                        if ind == iphotline:
                            photelemline = line.rstrip(' \n')
                            break

                photelem = photelemline.split()

                if len(photelem) > 0:

                    # get the station, framenumber, subframe, and ccd
                    framekeyelems = FRAMESUBREGEX.findall(
                        os.path.basename(iphot)
                    )
                    stf, cfn, cfs, fccd = (
                        framekeyelems[0][0],
                        framekeyelems[0][1],
                        framekeyelems[0][2],
                        framekeyelems[0][3]
                    )
                    net = 'HP' # this will be 'HP' for HATPI

                    # these are things we get from the iphot line
                    # hat, xcc, ycc, xic, yic
                    # fsv, fdv, fkv, bgv, bge
                    # ifl1, ife1, irm1, ire1, irq1
                    # ifl2, ife2, irm2, ire2, irq2
                    # ifl2, ife2, irm2, ire2, irq2

                    hat, xcc, ycc, xic, yic = photelem[:5]
                    fsv, fdv, fkv, bgv, bge = photelem[5:10]
                    ifl1, ife1, irm1, ire1, irq1 = photelem[10:15]
                    ifl2, ife2, irm2, ire2, irq2 = photelem[15:20]
                    ifl3, ife3, irm3, ire3, irq3 = photelem[20:]

                    # format these as needed
                    xcc = smartcast(xcc, float)
                    ycc = smartcast(ycc, float)
                    xic = smartcast(xic, float)
                    yic = smartcast(yic, float)

                    fsv = smartcast(fsv)
                    fdv = smartcast(fdv)
                    fkv = smartcast(fkv)
                    bgv = smartcast(bgv)
                    bge = smartcast(bge)

                    # TODO: finish this


    # if everything goes wrong, exit cleanly
    except Exception as e:

        database.rollback()

        message = 'failed to collect light curve for %s, DR%s, V%s' % (
            objectid, datarelease, lcversion
        )
        print('EXC! %sZ: %s\nexception was: %s' %
               (datetime.utcnow().isoformat(),
                message, format_exc()) )
        returnval = (objectid, None)
        raise


    finally:

        cursor.close()
        if closedb:
            database.close()

    return returnval


##############################
## 'FORCED PHOTOMETRY TOOLS ##
##############################

# 1. convert coords of forced source to x,y on photref
# 2. generate a fake .cmrawphot using the COMBINEDREFPHOTCMD fiphot, allow a
#    custom aperture string
# 3. once we have the fake cmrawphot, use it as input for SUBFRAMEPHOTCMD
# 4. this will make iphots for the target
# 5. insert into the DB in the forcedphot table, assign a HPT-XXX-YYYYYYY name,
#    insert into the forcedhatids table as well
# 6. collect the light curve the usual way
# 7. run EPD
# 8. run TFA (how?)

def forcedphot_generate_cmrawphot(
        objectid,
        ra,
        decl,
        projectid,
        field,
        ccd,
        photreftype,
        outfile,
        apertures='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
        ccdgain=None,
        zeropoint=None,
        ccdexptime=None,
        extractsources=True,
        refinfo=REFINFO):
    """
    This generates a cmrawphot file for objectid on the photref.

    objectid, ra, decl are either scalars or lists for the objects to do forced
    photometry for.
    """

    # first, get the photrefinfo
    cphotref = get_combined_photref(projectid,
                                    field,
                                    ccd,
                                    photreftype,
                                    refinfo=refinfo)

    # get the path to the photref fits
    framepath = cphotref['framepath']

    # find the WCS header of the cphotref
    wcspath = framepath.replace('.fits','.wcs')

    if os.path.exists(wcspath):
        photrefwcs = wcs.WCS(wcspath)
    else:
        print("ERR! %sZ: no WCS header found for %s, can't continue" %
              (datetime.utcnow().isoformat(), framepath))
        return None

    # get info from the frame
    # get the required header keywords from the FITS file
    header = imageutils.get_header_keyword_list(framepath,
                                                ['GAIN',
                                                 'GAIN1',
                                                 'GAIN2',
                                                 'EXPTIME',
                                                 'RAC',
                                                 'DECC',
                                                 'FOV'])

    # get the RA and DEC from the frame header for astrometry
    if 'RAC' in header and 'DECC' in header:
        frame_ra = header['RAC']*360.0/24.0
        frame_dec = header['DECC']
    else:
        print('ERR! %sZ: no RAC or DECC defined for %s' %
              (datetime.utcnow().isoformat(),
               framepath))
        return None

    # figure out the width of the frame for astrometry
    if 'FOV' in header:
        framewidth = header['FOV']
    elif framewidth is None:
        print('ERR! %sZ: no frame width defined for %s, '
              'astrometry not possible' %
              (datetime.utcnow().isoformat(),
               framepath))
        return None

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
        print('ERR! %sZ: no GAIN or EXPTIME defined for %s' %
              (datetime.utcnow().isoformat(),
               framepath))
        return None

    # handle the zeropoints
    # if the zeropoint isn't provided and if this is a HAT frame, the ccd
    # number will get us the zeropoint in the ZEROPOINTS dictionary
    if not zeropoint and ccdnumber in ZEROPOINTS:
        zeropoint = ZEROPOINTS[ccdnumber]
    else:
        print('ERR! %sZ: no zeropoint magnitude defined for %s' %
              (datetime.utcnow().isoformat(),
               framepath))
        return None


    ##############################
    # NOW GENERATE THE CMRAWPHOT #
    ##############################

    # put the objectid, ra, decl into the correct format

    if not isinstance(objectid, list) or not isinstance(objectid, tuple):
        objectid = [objectid]
    if not isinstance(ra, list) or not isinstance(ra, tuple):
        ra = [ra]
    if not isinstance(decl, list) or not isinstance(decl, tuple):
        decl = [decl]

    objectids = np.array(objectid)
    ras = np.array(ra)
    decls = np.array(decl)
    coords = np.column_stack((ras,decls))

    # convert RA/DEC to pixel x/y
    pixcoords = photrefwcs.all_world2pix(coords,1)

    # generate a sourcelist temp file
    srclist = tempfile.NamedTemporaryFile(delete=False)
    srclistf = srclist.name
    for o, r, d, pc in zip(objectids, ras, decls, pixcoords):
        srclist.write('{:s} {:.5f} {:.5f} {:.3f} {:.3f}\n'.
                      format(o, r, d, pc[0], pc[1]).encode('utf-8'))

    srclist.close()

    # use this srclist as input to the cmrawphot command
    cmdtorun = ism.COMBINEDPHOTREFCMD.format(
        photref=framepath,
        srclist=srclistf,
        srclist_idcol='1',
        srclist_xycol='4,5',
        ccdgain=ccdgain,
        zeropoint=zeropoint,
        exptime=ccdexptime,
        aperturestring=apertures,
        photrefbase=os.path.splitext(os.path.basename(framepath))[0],
        outfile=outfile
    )

    print('fiphot command: %s' % cmdtorun)

    returncode = os.system(cmdtorun)

    # remove the tempfile
    if os.path.exists(srclistf):
        os.remove(srclistf)
    photrefwcs.close()

    if returncode == 0:
        print('%sZ: forced photometry on photref %s OK -> %s' %
              (datetime.utcnow().isoformat(), framepath, outfile))
        return framepath, outfile
    else:
        print('ERR! %sZ: forced photometry on photref %s failed!' %
              (datetime.utcnow().isoformat(), framepath))
        if os.path.exists(outfile):
            os.remove(outfile)
        return framepath, None



def forcedphot_subphot_worker(task):
    """
    This does subtracted frame photometry on forced-phot objects.

    task[0] = subframe
    task[1] = photreftype
    task[2] = kernelspec
    task[3] = lcapertures
    task[4] = disjointradius
    task[5] = outdir
    task[6] = frcmrawphot
    """

    (subframe, photreftype, kernelspec,
     lcapertures, disjrad, outdir, frcmrawphot) = task

    try:

        # generate the convsubfits hash
        convsubhash = ism.get_convsubfits_hash(
            photreftype,
            ('reverse' if os.path.basename(subframe).startswith('rsub')
             else 'normal'),
            kernelspec
        )

        frameinfo = FRAMEREGEX.findall(
            os.path.basename(subframe)
        )

        # first, figure out the input frame's projid, field, and ccd
        frameelems = get_header_keyword_list(subframe,
                                             ['object',
                                              'projid'])
        field, ccd, projectid = (frameelems['object'],
                                 int(frameinfo[0][2]),
                                 frameelems['projid'])

        # then, find the associated cmrawphot
        cphotref_cmrawphot = frcmrawphot

        # find matching kernel, itrans, and xysdk files for each subtracted
        # frame

        photrefbit = (
            'rsub' if os.path.basename(subframe).startswith('rsub') else 'nsub'
        )

        kernelf = '%s-%s-%s-%s_%s-xtrns.fits-kernel' % (photrefbit,
                                                        convsubhash,
                                                        frameinfo[0][0],
                                                        frameinfo[0][1],
                                                        frameinfo[0][2])
        kernel = os.path.abspath(os.path.join(os.path.dirname(subframe),kernelf))

        itransf = '%s-%s_%s.itrans' % (frameinfo[0][0],
                                       frameinfo[0][1],
                                       frameinfo[0][2])
        itrans = os.path.abspath(os.path.join(os.path.dirname(subframe),itransf))

        xysdkf = '%s-%s_%s.xysdk' % (frameinfo[0][0],
                                     frameinfo[0][1],
                                     frameinfo[0][2])
        xysdk = os.path.abspath(os.path.join(os.path.dirname(subframe),xysdkf))


        # write the photometry file to /dev/shm by default
        # if outdir is None:
        #     outdir = '/dev/shm'

        _, subphot = ism.subframe_photometry_worker(
            (subframe, cphotref_cmrawphot, disjrad,
             kernel, itrans, xysdk, outdir,
             photreftype, kernelspec, lcapertures)
        )

        if subphot and os.path.exists(subphot):

            print('%sZ: CONVSUBPHOT (FORCED) OK: '
                  'subtracted frame %s, photometry file %s' %
                  (datetime.utcnow().isoformat(), subframe, subphot))

            return subframe, subphot

        else:

            print('%sZ: CONVSUBPHOT (FORCED) FAILED: subtracted frame %s' %
                  (datetime.utcnow().isoformat(), subframe))

            return subframe, None


    except Exception as e:

        message = ('could not do CONVSUBPHOT (FORCED) for %s, '
                   'exception follows' % subframe)
        print('EXC! %sZ: %s\n%s' %
               (datetime.utcnow().isoformat(), message, format_exc()) )

        return subframe, None



def parallel_convsubfits_forcedphot(
        subfitslist,
        forcedphot_cmrawphot,
        outdir,
        photreftype='oneframe',
        kernelspec='b/4;i/4;d=4/4',
        lcapertures='1.95:7.0:6.0,2.45:7.0:6.0,2.95:7.0:6.0',
        photdisjointradius=2,
        nworkers=16,
        maxworkertasks=1000,):
    """
    This does forced object photometry on the all subtracted FITS in
    subfitslist using the forcedphot_cmrawphot as input.

    Make sure the outdir is NOT the same as the dirname for the usual output
    iphot files. A good plan is to append forcedphot- in the directory name.
    """

    tasks = [(x, photreftype, kernelspec,
              lcapertures, photdisjointradius,
              outdir, forcedphot_cmrawphot)
             for x in subfitslist if os.path.exists(x)]

    if len(tasks) > 0:

        # make sure the output directory exists
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        # fire up the pool of workers
        results = pool.map(forcedphot_subphot_worker, tasks)

        # wait for the processes to complete work
        pool.close()
        pool.join()

        return {x:y for (x,y) in results}

    else:

        print('ERR! %sZ: none of the files specified exist, bailing out...' %
              (datetime.utcnow().isoformat(),))
        return


# AFTER THE ABOVE FOR FORCED PHOTOMETRY:

# 1. run the usual collection function: insert_phots_into_database with photdir
#    set to that of the forcedphot- output iphot directory.
# 2. then for each 'hatid', run the light curve collection function:
#    dbphot_collect_imagesubphot_lightcurve, where 'hatid' is usally a special
#    objectid that we gave to the object we're doing forced photometry for


##############################
## MOVING OBJECT PHOTOMETRY ##
##############################

# TODO: figure out how to do moving object photometry: one way would be to use
# the orbital elements or whatever to figure out the ra/dec of the moving object
# at each center time of each frame in framedir, then generate a forced
# photometry cmrawphot for each of these positions as separate 'subhatids'. then
# run parallel_subhot_forcedphot as usual. this should give you iphots with
# information for each subhatid. then, using a map between subhatid and
# framenumber, collect the correct light curve and write out to a file.



###################
## VISUALIZATION ##
###################


def subfits_to_jpeg_series(subframedir,
                           subframeglob='rsub-*-xtrns.fits',
                           origframedir=None,
                           outdir=None,
                           makemovie=False,
                           moviefps=10):
    """
    This generates JPEGs for all subtracted FITS in subframedir.

    origframedir is directory of the original FITS to get JD from.
    """

    subframes = sorted(glob.glob(os.path.join(subframedir, subframeglob)))

    if origframedir is None:
        origframedir = subframedir

    if outdir is None:
        outdir = subframedir

    if subframes:

        nsubframes = len(subframes)

        for ind, frame in enumerate(subframes):

            frameinfo = FRAMEREGEX.findall(os.path.basename(frame))
            if '.fz' in frame:
                originalframe = '%s-%s_%s.fits.fz' % (frameinfo[0][0],
                                                      frameinfo[0][1],
                                                      frameinfo[0][2])
                outfname = os.path.join(
                    outdir,
                    os.path.basename(frame).replace('.fits.fz',
                                                    '.jpg')
                )
            else:
                originalframe = '%s-%s_%s.fits' % (frameinfo[0][0],
                                                   frameinfo[0][1],
                                                   frameinfo[0][2])
                outfname = os.path.join(outdir,
                                        os.path.basename(frame).replace('.fits',
                                                                        '.jpg'))

            originalframe = os.path.join(origframedir, originalframe)

            # generate the JPEG
            jpeg = fits_to_full_jpeg(frame,
                                     out_fname=outfname,
                                     fits_jdsrc=originalframe)
            print('(%s/%s) subframe: %s -> jpeg: %s OK' %
                  (ind+1, nsubframes, frame, jpeg))


        # make a movie if we're told to do so
        if makemovie:

            movie_fname = os.path.join(outdir, 'subframe-movie.mp4')
            jpgglob = subframeglob.replace('.fits','.jpg')
            moviefile = make_frame_movie(outdir,
                                         movie_fname,
                                         framerate=moviefps,
                                         jpegglob=jpgglob)
            return outdir, moviefile

        else:

            return outdir, None

    # if no subframes were found in this directory, do nothing
    else:
        print('no subtracted frames found in %s' % subframedir)
        return None, None



def subfits_radec_to_jpeg_series(subframedir,
                                 radecspec,
                                 astromrefwcs,
                                 subframeglob='rsub-*-xtrns.fits',
                                 origframedir=None,
                                 outdir=None,
                                 makemovie=False,
                                 moviefps=10):
    """
    This generates JPEGs for all subtracted FITS in subframedir.

    origframedir is directory of the original FITS to get JD from.

    radecspec is a list with four elements:

    [racenter (decimal), declcenter (decimal),
     ra width (decimal), decl height (decimal)]

    astromrefwcs is the file to get the WCS info from. this must be from the
    astrometric reference (i.e. the frame all other frames were shifted
    to). this is used because the individual WCS corresponding to each
    subtracted frame are usually the same as the original frame WCS, which may
    have been shifted around to match the astromref so any RA/DEC -> x/y
    transform will give the wrong results.
    """

    subframes = sorted(glob.glob(os.path.join(subframedir, subframeglob)))

    if origframedir is None:
        origframedir = subframedir

    if outdir is None:
        outdir = subframedir
    elif outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    if subframes:

        nsubframes = len(subframes)

        for ind, frame in enumerate(subframes):

            frameinfo = FRAMEREGEX.findall(os.path.basename(frame))
            if '.fz' in frame:
                originalframe = '%s-%s_%s.fits.fz' % (frameinfo[0][0],
                                                      frameinfo[0][1],
                                                      frameinfo[0][2])
                outfname = os.path.join(
                    outdir,
                    os.path.basename(frame).replace('.fits.fz',
                                                    '.jpg')
                )
            else:
                originalframe = '%s-%s_%s.fits' % (frameinfo[0][0],
                                                   frameinfo[0][1],
                                                   frameinfo[0][2])
                outfname = os.path.join(outdir,
                                        os.path.basename(frame).replace('.fits',
                                                                        '.jpg'))

            originalframe = os.path.join(origframedir, originalframe)

            # generate the JPEG
            jpeg = frame_radecbox_to_jpeg(frame,
                                          wcsfrom=astromrefwcs,
                                          radeccenter=radecspec,
                                          jdsrc=originalframe,
                                          out_fname=outfname)
            print('(%s/%s) subframe: %s -> jpeg: %s OK' %
                  (ind+1, nsubframes, frame, jpeg))


        # make a movie if we're told to do so
        if makemovie:

            movie_fname = os.path.join(outdir, 'subframe-movie.mp4')
            jpgglob = subframeglob.replace('.fits','.jpg')
            moviefile = make_frame_movie(outdir,
                                         movie_fname,
                                         framerate=moviefps,
                                         jpegglob=jpgglob)
            return outdir, moviefile

        else:

            return outdir, None

    # if no subframes were found in this directory, do nothing
    else:
        print('no subtracted frames found in %s' % subframedir)
        return None, None



def subfits_pixbox_to_jpeg_series(subframedir,
                                  pixspec,
                                  pixspectype='center',
                                  subframeglob='rsub-*-xtrns.fits',
                                  origframedir=None,
                                  outdir=None,
                                  makemovie=False,
                                  moviefps=10):
    """
    This generates JPEGs for all subtracted FITS in subframedir.

    origframedir is directory of the original FITS to get JD from.

    if pixspectype = 'center':

    radecspec is a list with four elements:

    [pixcenter (decimal), pixcenter (decimal),
     pix width (decimal), pix height (decimal)]

    elif pixspectype = 'box':

    radecspec is a list with four elements:

    [xminpix, xmaxpix, yminpx, ymaxpx]
    """

    subframes = sorted(glob.glob(os.path.join(subframedir, subframeglob)))

    if origframedir is None:
        origframedir = subframedir

    if outdir is None:
        outdir = subframedir
    elif outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    if subframes:

        nsubframes = len(subframes)

        for ind, frame in enumerate(subframes):

            frameinfo = FRAMEREGEX.findall(os.path.basename(frame))
            if '.fz' in frame:
                originalframe = '%s-%s_%s.fits.fz' % (frameinfo[0][0],
                                                      frameinfo[0][1],
                                                      frameinfo[0][2])
                outfname = os.path.join(
                    outdir,
                    os.path.basename(frame).replace('.fits.fz',
                                                    '.jpg')
                )
            else:
                originalframe = '%s-%s_%s.fits' % (frameinfo[0][0],
                                                   frameinfo[0][1],
                                                   frameinfo[0][2])
                outfname = os.path.join(outdir,
                                        os.path.basename(frame).replace('.fits',
                                                                        '.jpg'))

            originalframe = os.path.join(origframedir, originalframe)

            # generate the JPEG
            if pixspectype == 'center':
                jpeg = fitscoords_to_jpeg(frame,
                                          coordcenter=pixspec,
                                          jdsrc=originalframe,
                                          out_fname=outfname)
            elif pixspectype == 'box':
                jpeg = fitscoords_to_jpeg(frame,
                                          coordbox=pixspec,
                                          jdsrc=originalframe,
                                          out_fname=outfname)


            print('(%s/%s) subframe: %s -> jpeg: %s OK' %
                  (ind+1, nsubframes, frame, jpeg))


        # make a movie if we're told to do so
        if makemovie:

            movie_fname = os.path.join(outdir, 'subframe-movie.mp4')
            jpgglob = subframeglob.replace('.fits','.jpg')
            moviefile = make_frame_movie(outdir,
                                         movie_fname,
                                         framerate=moviefps,
                                         jpegglob=jpgglob)
            return outdir, moviefile

        else:

            return outdir, None

    # if no subframes were found in this directory, do nothing
    else:
        print('no subtracted frames found in %s' % subframedir)
        return None, None


#############################
## LIGHT CURVE EPD AND TFA ##
#############################



###########################
## CRON ROLLUP FUNCTIONS ##
###########################
