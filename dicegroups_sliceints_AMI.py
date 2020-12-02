#! /usr/bin/env python

import glob
import os
import numpy as np
from astropy.io import fits


'''
Dice by groups:
Is there a detector nonlinearity oing up the ramp that affects achievable contrast? 
Take a 4d file (ramp file), create (e.g.) two 4d files where the first half of the groups
of each integration is in one file, and the upper half in another file. Use for charge migration
testing

Slice  by ints:
Take a 4d ramp file and slice it by integration, so we create (e.g) two files where one contains
the first half of the integrations (all groups of those integrations) and the second contains the 
second half of the integrations (all groups). Use for persistence testing
'''



def dice_groups(fn, ndice, outdir=None, overwrite=False):
    '''
    Dice input 4d uncal files into chunks of groups, discarding the remaining groups.
    E.g. for a file with dimensions (10, 5, 80, 80) and ndice=2,
    it will create two files, each with shape (10, 2, 80, 80)
    comprising the first 2 groups and the second 2 groups and discarding the 5th (last) group.
    Affected header keywords are updated based on the number of groups in output files,
    but may be inaccurate due to overheads.
    Files are named with suffix _Ngroups_chunkX.fits, where N is how many groups are in each
    chunk and X is the chunk number.

    Inputs:
        fn: (str) filename of uncal AMI file to dice up
        ndice: (int) number of chunks to divide groups into. if ngroup % ndice != 0, remove last groups.
        outdir: (str) output directory to write diced-up files to. If None, will write to cwd
        overwrite: (bool) overwrite existing FITS files if they exist? Default False
    '''
    # housekeeping
    if outdir == None:
        outdir = './'  # use the current directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print('Created output directory %s' % outdir)

    # read in the data and header
    with fits.open(fn) as hdu1:
        data = hdu1[1].data
        header = hdu1[0].header
        shape = data.shape
        print('Input data shape:', shape)
        # check that dimensions match ngroups keyword
        ngroups = header['NGROUPS']
        if shape[1] != ngroups:
            raise ValueError('ERROR: NGROUPS value in header inconsistent with data shape.')
        # check that ndice input is appropriate
        if ndice > ngroups:
            raise ValueError(
                'ERROR: number of chunks to divide into must be less than or equal to the number of groups')
        # see if groups can be divided nicely into ndice
        remainder = ngroups % ndice
        if remainder != 0:
            print('Will discard the last %i group(s)' % remainder)
        # collect some other keyword values to be used
        tframe = header['TFRAME']
        nframes = header['NFRAMES']
        grpgap = header['GROUPGAP']
        drpframes = 0  # always true for NIRISS AMI(?)
        tgroup = header['TGROUP']
        tgroup0 = tgroup  # seems to be true so far
        nints = header['NINTS']

        ngroups_out = shape[1] // ndice  # how many groups in each output file
        print('Will produce %i output files, each with shape (%i, %i, %i, %i)' % (
        ndice, shape[0], ngroups_out, shape[2], shape[3]))
        effexptm_out = tframe * (ngroups_out * nframes + (ngroups_out - 1) * grpgap + drpframes) * nints
        duration_out = tframe * (1 + (ngroups_out * nframes))
        effinttm_out = (ngroups_out - 1) * tgroup + (tgroup0)
        # recalculate exposure timing keywords with new number of groups used.
        # PROBABLY INACCURATE
        print('\t exptime in:', header['EFFEXPTM'], '\t exptime out:', effexptm_out)
        print('\t inttime in:', header['EFFINTTM'], '\t inttime out:', effinttm_out)
        print('\t duration in:', header['DURATION'], '\t duration out:', duration_out)

        # Iterate over however many chunks (ndice) were requested
        start = 0
        for chunk in np.arange(ndice) + 1:
            print('Making file', chunk)
            stop = start + ngroups_out
            print('\t Starting index: %i, stopping index: %i' % (start, stop))
            outdata = data[:, start:stop, :, :]
            print('\t Output data shape:', outdata.shape)

            # update output header vals with new ngroups and updated timing values
            fn_out = os.path.basename(fn).split('.fits')[0] + '_%igroups_chunk%i.fits' % (ngroups_out, chunk)

            hdu1[1].data = outdata
            hdu1[0].header['NGROUPS'] = ngroups_out
            hdu1[0].header['EFFEXPTM'] = effexptm_out
            hdu1[0].header['DURATION'] = duration_out
            hdu1[0].header['EFFINTTM'] = effinttm_out
            hdu1[0].header['HISTORY'] = ('Exposure timing keywords EFFEXPTM, DURATION, and EFFINTM \
                                         have been updated to reflect new NGROUPS value but may be inaccurate')

            hdu1.writeto(os.path.join(outdir, fn_out), overwrite=overwrite)
            print('\t Saved output file to', fn_out)

            # update the start value for the next chunk
            start = stop


def slice_ints(fn, nslice, outdir=None, overwrite=False):
    '''
    Slice up input 4d uncal files into chunks of integrations, discarding remainder ints.
    E.g. for a file with dimensions (10, 5, 80, 80) and nslice=2,
    it will create two files, each with shape (5, 5, 80, 80)
    comprising the first 5 ints (all groups of each int) and the second 5 ints.
    Affected header keywords are updated based on the number of ints in output files,
    but may be inaccurate due to asymmetrical overheads.
    Files are named with suffix _Nints_chunkX.fits, where N is how many ints are in each
    chunk and X is the chunk number.

    Inputs:
        fn: (str) filename of uncal AMI file to slice up
        ndice: (int) number of chunks to divide ints into. if nints % ndice != 0, remove remainder ints.
        outdir: (str) output directory to write sliced-up files to. If None, will write to cwd
        overwrite: (bool) overwrite existing FITS files if they exist? Default False
    '''
    # Housekeeping
    if outdir == None:
        outdir = './'  # use the current directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print('Created output directory %s' % outdir)

    # read in the data and header
    with fits.open(fn) as hdu1:
        data = hdu1[1].data
        header = hdu1[0].header
        shape = data.shape
        print('Input data shape:', shape)
        # check that dimensions match nints keyword
        nints = header['NINTS']
        if shape[0] != nints:
            raise ValueError('ERROR: NINTS value in header inconsistent with data shape.')
        # check that nslice input is appropriate
        if nslice > nints:
            raise ValueError('ERROR: number of chunks to divide into must be less than or equal to the number of ints')
        # see if ints can be divided nicely into nslice chunks
        remainder = nints % nslice
        if remainder != 0:
            print('Will discard the last %i int(s)' % remainder)
        # collect some other keyword values to be used
        tframe = header['TFRAME']
        nframes = header['NFRAMES']
        grpgap = header['GROUPGAP']
        drpframes = 0  # always true for NIRISS AMI(?)
        tgroup = header['TGROUP']
        tgroup0 = tgroup  # seems to be true so far
        ngroups = header['NGROUPS']

        nints_out = shape[0] // nslice  # how many groups in each output file
        print('Will produce %i output files, each with shape (%i, %i, %i, %i)' % (
        nslice, nints_out, shape[1], shape[2], shape[3]))
        effexptm_out = tframe * (ngroups * nframes + (ngroups - 1) * grpgap + drpframes) * nints_out

        # recalculate exposure timing keywords with new number of groups used.
        # effexptm is the only one that uses ints
        # PROBABLY INACCURATE
        print('\t exptime in:', header['EFFEXPTM'], '\t exptime out:', effexptm_out)

        # Iterate over however many chunks (nslice) were requested
        start = 0
        for chunk in np.arange(nslice) + 1:
            print('Making file', chunk)
            stop = start + nints_out
            print('\t Starting index: %i, stopping index: %i' % (start, stop))
            outdata = data[start:stop, :, :, :]
            print('\t Output data shape:', outdata.shape)

            # update output header vals with new ngroups and updated timing values
            fn_out = os.path.basename(fn).split('.fits')[0] + '_%iints_chunk%i.fits' % (nints_out, chunk)

            hdu1[1].data = outdata
            hdu1[0].header['NINTS'] = nints_out
            hdu1[0].header['EFFEXPTM'] = effexptm_out
            hist1 = 'Exposure timing keyword EFFEXPTM was updated to reflect new NINTS value but may be inaccurate'
            hist2 = 'This is %i of %i files diced into %i ints each, from original file %s' % (
            chunk, nslice, nints_out, fn)
            hdu1[0].header['HISTORY'] = hist1
            hdu1[0].header['HISTORY'] = hist2

            hdu1.writeto(os.path.join(outdir, fn_out), overwrite=overwrite)
            print('\t Saved output file to', os.path.join(outdir, fn_out))

            # update the start value for the next chunk
            start = stop