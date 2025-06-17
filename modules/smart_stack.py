"""
Copyright 2010 - 2025 Michael Fitzgerald.

(E)ducation (V)alue (A)dded Pipeline (EVAPipeline)

Formally known as Unnamed (2010-2014), OSSPipeline (2014-2024), EVApipeline (2024-continuing)

Creative Commons Attribution­NonCommercial­NoDerivatives 4.0 International

Copyright © 2010 - 2025 Michael Fitzgerald

You are free to:
  • Share — copy and redistribute this material in any medium or format

Under the following terms:
  • Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
  • NonCommercial — You may not use the material for commercial purposes.
  • NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.

Notices:
  • No warranties are given. The software is provided “as-is,” without warranty of any kind.
  • If you wish to propose changes or collaborate, please submit pull-requests; public redistribution of modified code is prohibited without express permission.
  
For licensing inquiries, please contact: Michael Fitzgerald, psyfitz@gmail.com

Paper on OSSPipeline: https://rtsre.org/index.php/rtsre/article/view/12/12


"""

import numpy as np
from astropy.io import fits
import traceback
from astropy import wcs
import bottleneck as bn
import math
import os
from multiprocessing.pool import Pool
from multiprocessing import TimeoutError as MPTimeoutError
from pathlib import Path
import astropy.units as u
from reproject.mosaicking import find_optimal_celestial_wcs
import psutil
import subprocess
from datetime import datetime
import pickle
import time
import warnings
import gc
from modules.image_functions import thresh, mask_cosmics, detect_and_mask_stars
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage import median_filter
from astropy.stats import sigma_clip
import logging

kernel = Gaussian2DKernel(x_stddev=2,y_stddev=2)
from astropy.stats import mad_std


def load_header(file):
    """Load a FITS header saved via pickle."""
    with open(file.replace('.npy', '.head'), 'rb') as fp:
        return pickle.load(fp)


def SNPprocess_files_with_timeouts(file_list, timeout_total=120 , maximum_cpus_ever=8):
    max_workers = max(min(5, os.cpu_count() // 2, maximum_cpus_ever), 1)
    
    with Pool(processes=max_workers) as pool:
        # schedule the entire batch
        async_map = pool.map_async(saltandpepper_process_file, file_list)
        try:
            # wait up to `timeout_total` seconds for *all* files
            results = async_map.get(timeout=timeout_total)
        except (MPTimeoutError, TimeoutError):
            logging.info("Salt and Pepper Batch processing timed out; terminating pool.")
            pool.terminate()  # kills all workers immediately
            pool.join()
            # fallback: you could return partial results, or a list of None
            return [None] * len(file_list)
    
    return results

def saltandpepper_process_file(file):
    try:
        imagedata = np.load(file)
        newheader = load_header(file)
        _, noise_mask, _ = mitigate_salt_and_pepper_noise(
            imagedata, threshold=12, fwhm=float(newheader['FWHMpix']), pixscale=float(newheader['PIXSCALE'])
        )
        imagedata[noise_mask] = np.nan
        np.save(file, imagedata)
    except:
        logging.info(f"Salt and pepper noise mitigation failed for {file}")
        logging.info(traceback.format_exc())

def smart_stack(fileList, telescope, basedirectory, memmappath, calibration_directory, codedir):

    # First detect missing WCS headers... no point including them if they don't have WCS!
    deleteList=[]
    for file in fileList:
        prihdr = load_header(file)

        w = wcs.WCS(prihdr)
        logging.info (file)
        logging.info (w)
        if not w.has_celestial:
            logging.info (file + " doesn't seem to have a WCS, removing from the smartstack input list")
            deleteList.append(file)
        elif (not 'CRVAL1' in prihdr) or (not 'CRPIX1' in prihdr) or (not 'CTYPE1' in prihdr) or (not 'CTYPE2' in prihdr):
            logging.info (file + " has missing values in the WCS, removing from the smartstack input list")
            deleteList.append(file)
        elif prihdr['CRVAL1'] == 0.0 or prihdr['CRPIX1'] == 0.0:
            logging.info (file + " has zero values in the WCS, removing from the smartstack input list")
            deleteList.append(file)

    for entry in deleteList:
        fileList.remove(entry)

    if len(fileList) == 1 or len(fileList) == 0:
        return

    maximum_cpus_ever=len(fileList)

    # Make sure there aren't any significant positional outliers
    RAcollect=[]
    DECcollect=[]
    for file in fileList:
        prihdr = load_header(file)
        RAcollect.append(float(prihdr['CRVAL1']))
        DECcollect.append(float(prihdr['CRVAL2']))
    RAcollect=np.asarray(RAcollect)
    DECcollect=np.asarray(DECcollect)

    RAmed=bn.nanmedian(RAcollect)
    DECmed=bn.nanmedian(DECcollect)
    RAstd=bn.nanstd(RAcollect)
    DECstd=bn.nanstd(DECcollect)
    logging.info ("RA median position and standard deviation before rejections")
    logging.info (RAmed)
    logging.info (RAstd)
    logging.info ("DEC median position and standard deviation before rejections")
    logging.info (DECmed)
    logging.info (DECstd)


    if np.isnan(RAmed) or np.isnan(DECmed):
        return

    if fileList==[]:
        return

    # We want to reject outliers, but also images that are
    # a significant fraction (6 times) higher than the traditional
    # dithering in a smartstack from the list.
    # so 300 pixels
    dithering_threshold=300 * (float(prihdr['PIXSCALE']) / 3600)

    RAcollect=[]
    DECcollect=[]
    for file in fileList:
        prihdr = load_header(file)

        logging.info ("************")
        logging.info (file)
        logging.info ("CRVAL1 :" + str(prihdr['CRVAL1']))
        logging.info ("CRVAL2 :" + str(prihdr['CRVAL2']))


        if float(prihdr['CRVAL1']) > (RAmed + 5*RAstd) or float(prihdr['CRVAL1']) > (RAmed + abs(dithering_threshold * math.cos(math.radians(DECmed)))):
            logging.info ("Reject as too high an RA")
            logging.info (prihdr['CRVAL1'])
            logging.info (RAmed + 5*RAstd)
            logging.info ("REJECT as positional outlier " + str(file))
            try:
                fileList.remove(file)
            except:
                logging.info ("Already deleted?")
        elif float(prihdr['CRVAL1']) < (RAmed - 5*RAstd) or float(prihdr['CRVAL1']) < (RAmed - abs(dithering_threshold * math.cos(math.radians(DECmed)))):
            logging.info ("Reject as too low an RA")
            logging.info (prihdr['CRVAL1'])
            logging.info (RAmed - 5*RAstd)
            logging.info ("REJECT as positional outlier " + str(file))
            try:
                fileList.remove(file)
            except:
                logging.info ("Already deleted?")
        elif float(prihdr['CRVAL2']) < (DECmed - 5*DECstd) or float(prihdr['CRVAL2']) < (DECmed - dithering_threshold):
            logging.info ("Reject as too low a DEC")
            logging.info (prihdr['CRVAL2'])
            logging.info ((DECmed - 5*DECstd))
            logging.info ((DECmed - dithering_threshold))

            logging.info ("REJECT as positional outlier " + str(file))
            try:
                fileList.remove(file)
            except:
                logging.info ("Already deleted?")
        elif float(prihdr['CRVAL2']) > (DECmed + 5*DECstd) or float(prihdr['CRVAL2']) > (DECmed + dithering_threshold):
            logging.info ("Reject as too high a DEC")
            logging.info (prihdr['CRVAL2'])
            logging.info ((DECmed + 5*DECstd))
            logging.info ((DECmed + dithering_threshold))
            logging.info ("REJECT as positional outlier " + str(file))
            try:
                fileList.remove(file)
            except:
                logging.info ("Already deleted?")
        else:
            RAcollect.append(float(prihdr['CRVAL1']))
            DECcollect.append(float(prihdr['CRVAL2']))

    logging.info ("RA median position and standard deviation after rejections")
    logging.info (bn.nanmedian(RAcollect))
    logging.info (bn.nanstd(RAcollect))
    logging.info ("DEC median position and standard deviation after rejections")
    logging.info (bn.nanmedian(DECcollect))
    logging.info (bn.nanstd(DECcollect))    
        
    # Here we go through and reject any images with overly high FWHM or skylevels
    # On the assumption that lower values are generally better in both cases.
    rejected=1
    while rejected==1:
        fwhm_list=[]
        sky_level=[]
        rejected=0
        for file in fileList:
            prihdr = load_header(file)
            if not prihdr['FWHM'] == 'Unknown':
                fwhm_list.append(prihdr['FWHM'])
            if not prihdr['SKYLEVEL'] == 'Unknown':        
                sky_level.append(prihdr['SKYLEVEL'])
        
        fwhm_median=np.nanmedian(fwhm_list)
        fwhm_std=np.nanstd(fwhm_list)
        sky_level_median=np.nanmedian(sky_level)
        sky_level_std= np.nanstd(sky_level)
        
        
        logging.info ("FWHM    : " + str(fwhm_median) + " STD: " + str(fwhm_std))
        logging.info ("SKYLEVEL: " + str(sky_level_median) + " STD: " + str(sky_level_std))
        
        for file in fileList:
            prihdr = load_header(file)
            if prihdr['FWHM'] == 'Unknown':
                rejected=1
                fileList.remove(file)
            elif prihdr['FWHM'] > (fwhm_median + 4*fwhm_std):
                logging.info ("Rejected high FWHM: " + str(prihdr['FWHM'] ))
                rejected=1
                fileList.remove(file)
            elif prihdr['SKYLEVEL'] > (sky_level_median + 4*sky_level_std):
                logging.info ("Rejected high SKYLEVEL: " + str(prihdr['SKYLEVEL'] ))
                rejected=1
                fileList.remove(file)
    
    logging.info (fwhm_list)
    logging.info (sky_level)
            
    try:
        do_salt_and_pepper=prihdr['DOSNP']
    except:
        do_salt_and_pepper=False
    
    del prihdr

    if len(fileList) >= 5 and do_salt_and_pepper:  
        logging.info ('Doing Salt and Pepper Routine')
        SNPprocess_files_with_timeouts(fileList, timeout_total=180, maximum_cpus_ever=maximum_cpus_ever)

    # Sort out a new header
    expHolder=[]
    inputList=[]
    i=0

    # We need to calculate image rotation so as to crop nan edges due to rotation relative to RA and Dec
    largest_image_rotation=0
    
    
    # if we are going to deal with LCO variance we need to build up the median varianceframe
    # it shouldn't be too offensive to hold them in memory and squish them. 
    lco_variance_frames=[]

    for file in fileList:
        if i == 0:
            logging.info (0)
            logging.info (i)

            logging.info (file)
            imagedata= np.load(file)
            image_shape=imagedata.shape
            newheader = load_header(file)

            cleanhdu=fits.PrimaryHDU()
            cleanhdu.data=np.asarray(imagedata)
            del imagedata
            cleanhdu.header=newheader
            cleanhdu.writeto(file.replace('.npy', '.fits'), output_verify='silentfix', overwrite=True)

            del cleanhdu
            
            if telescope == 'lco':
                # Get LCO variance file
                origfilename=file.split('/')[-1]
                try:                
                    lco_variance_frames.append(np.load(basedirectory+'/lcoerrays/'+origfilename))
                except:
                    logging.info ("No variance frame for lc0 file")
                    lco_variance_frames.append(None)


            inputList.append(file.replace('.npy', '.fits'))
            try:
                cdonetwo=newheader['PC1_2']
                cdoneone=newheader['PC1_1']
            except:
                cdonetwo=newheader['CD1_2']
                cdoneone=newheader['CD1_1']
            imagerot=abs(math.degrees(math.atan(cdoneone/cdonetwo)))

            # sort out largest_image_rotation into the right domain if it isn't in the normal around 0 domain
            if 45 < imagerot <= 135:
                imagerot = abs(90-imagerot)
            elif 135 < imagerot <= 225:
                imagerot = abs(180-imagerot)
            elif 225 < imagerot <= 315:
                imagerot = abs(270-imagerot)


            if imagerot > largest_image_rotation:
                largest_image_rotation = imagerot

            logging.info ("image rotation: " +str(imagerot))

            userId=newheader['USERID'].replace('+','').replace('.','d').replace('_','u').replace('googledoauth','').replace('@gmail.com','').replace('@sacredheart.wa.edu.au','').replace(' ','').replace('.','d').replace('_','').replace('-','d').replace(':','d').replace('.com','').replace('@','at').replace('.','d').replace('|','').replace('ossatncc2dvicdedudau','')

            telId=newheader['TELID']

            newheader.add_comment('EVA Averaged frames:')
            newheader.add_comment(file)


            #Prepping the header and filename changes
            expTime=newheader['EXPTIME']
            expHolder.append(expTime)
            airMass=newheader['AIRMASS']
            logging.info (expTime)
            if any("MJD-OBS" in s for s in newheader.keys()):
                mjd=float(newheader['MJD-OBS'])
            if any("MJD-MID" in s for s in newheader.keys()):
                mjdmid=float(newheader['MJD-MID'])
            if any("JD" in s for s in newheader.keys()):
                jd=float(newheader['JD'])
            if any("JD-MID" in s for s in newheader.keys()):
                jdmid=float(newheader['JD-MID'])
            if any("BJD" in s for s in newheader.keys()):
                bjd=float(newheader['BJD'])
            objectTemp=newheader['OBJECT'].replace('-','d').replace('+','p').replace('.','d').replace(' ','').replace('_','').replace('=','e').replace('(','').replace(')','').replace('<','').replace('>','').replace('/','')
            if 'FILTER1' in newheader:
                filterOne=(newheader['FILTER1']).replace(' ','').replace('_','')
                filterTwo=(newheader['FILTER2']).replace(' ','').replace('_','')
                filterThree=(newheader['FILTER3']).replace(' ','').replace('_','')
            else:
                filterOne=(newheader['FILTER'].replace('_','').replace('SDSS-','').replace('-011','').replace('RP','rp').replace('IP','ip').replace('-','d')).replace(' ','')
                filterTwo='NOTPRESENT'
                filterThree='NOTPRESENT'
            dateObs=(newheader['DATE'].replace('-','d').replace(':','d').replace('.','d'))
            instruMe=(newheader['INSTRUME'])
            instruMe=instruMe.replace(' ','').replace('_','d').replace('-','d').replace('+','p').replace('.','d').replace(' ','').replace('_','').replace('=','e').replace('(','').replace(')','').replace('<','').replace('>','').replace('/','')
            if any("REQNUM" in s for s in newheader.keys()):
                newheader['REQNUM'] = str(newheader['REQNUM']).replace("1","2")

        if i != 0:
            logging.info (i)
            sechdr = load_header(file)
            imagedata= np.load(file)
            
            cleanhdu=fits.PrimaryHDU()
            cleanhdu.data=np.asarray(imagedata)
            del imagedata
            cleanhdu.header=sechdr
            cleanhdu.writeto(file.replace('.npy', '.fits'), output_verify='silentfix', overwrite=True)

            del cleanhdu
            
            if telescope == 'lco':
                # Get LCO variance file
                origfilename=file.split('/')[-1]
                try:                
                    lco_variance_frames.append(np.load(basedirectory+'/lcoerrays/'+origfilename))
                except:
                    logging.info ("No variance frame for lc0 file")
                    lco_variance_frames.append(None)

            inputList.append(file.replace('.npy', '.fits'))

            try:
                cdonetwo=newheader['PC1_2']
                cdoneone=newheader['PC1_1']
            except:
                cdonetwo=newheader['CD1_2']
                cdoneone=newheader['CD1_1']
            imagerot=abs(math.degrees(math.atan(cdoneone/cdonetwo)))
            
            # sort out largest_image_rotation into the right domain if it isn't in the normal around 0 domain
            if 45 < imagerot <= 135:
                imagerot = abs(90-imagerot)
            elif 135 < imagerot <= 225:
                imagerot = abs(180-imagerot)
            elif 225 < imagerot <= 315:
                imagerot = abs(270-imagerot)
            if imagerot > largest_image_rotation:
                largest_image_rotation = imagerot

            logging.info ("image rotation: " +str(imagerot))

            newheader.add_comment(file)
            #sechdr=hdu2.header
            expTime=expTime+float(sechdr['EXPTIME'])
            expHolder.append(sechdr['EXPTIME'])
            logging.info (expTime)
            if any("MJD-OBS" in s for s in sechdr.keys()):
                mjd=mjd+float(sechdr['MJD-OBS'])
            if any("MJD-MID" in s for s in sechdr.keys()):
                mjdmid=mjdmid+float(sechdr['MJD-MID'])
            if any("JD" in s for s in sechdr.keys()):
                jd=jd+float(sechdr['JD'])
            if any("JD-MID" in s for s in sechdr.keys()):
                jdmid=jdmid+float(sechdr['JD-MID'])
            if any("BJD" in s for s in sechdr.keys()):
                bjd=bjd+float(sechdr['BJD'])

            airMass=airMass+sechdr['AIRMASS']

        i=i+1

    # median combine the variance frames
    if telescope == 'lco':
        
        lco_variance_frame_filename= basedirectory+'/lcoerrays/'+newheader['SMARTSTK']+'.npy'
        try:
            lco_variance_frames = np.stack(lco_variance_frames, axis=0)
            lco_variance_frames = np.nanmedian(lco_variance_frames, axis=0)
            # Crop to match original crops shape
            cropvalue_h=int(lco_variance_frames.shape[1]*0.05) + 20 
            cropvalue_w=int(lco_variance_frames.shape[0]*0.05) + 20
    
            lco_variance_frames  = lco_variance_frames [cropvalue_w:-cropvalue_w,cropvalue_h:-cropvalue_h]
            
            np.save(lco_variance_frame_filename, lco_variance_frames)
            del lco_variance_frames
        except:
            logging.info ("no lco variance frames?")
            logging.info ("putting a fake frame in")
            np.save(lco_variance_frame_filename, np.zeros(image_shape))
            
    logging.info ('largest image rotation: ' + str(largest_image_rotation))

    logging.info (newheader['PIXSCALE'])

    newheader['ORIGPXSC']= float(newheader['PIXSCALE'])
    originalpixelscale=float(newheader['PIXSCALE'])

    # Setup inputList
    if telescope == 'lco':
        newPixScale=float(0.74)*u.arcsec
        logging.info ("Drizzling to resolution: " + str(newPixScale)+ " arcseconds/pixel.")
        newheader['PIXSCALE']=float(0.5)
        
    else:

        newPixScale=float(newheader["DRZPIXSC"])*u.arcsec
        logging.info ("Drizzling to resolution: " + str(newPixScale)+ " arcseconds/pixel.")
        newheader['PIXSCALE']=float(newheader["DRZPIXSC"])
    try:        
        wcs_out, shape_out = find_optimal_celestial_wcs(inputList, resolution=newPixScale) #
    except:
        logging.info(traceback.format_exc())
        return
    logging.info (wcs_out)
    logging.info (shape_out)


    fileSize = shape_out[0] * shape_out[1]
    megapixels= fileSize / 1000000
    # It takes .13 GB per megapixel to process a single stack.
    #memory_burden_per_image = .18 * megapixels # originally!
    memory_burden_per_image = .04 * megapixels # originally!
    # Not the entirety of memory!! Most of it
    total_available_memory = (psutil.virtual_memory().total/1000000000) * 0.45
    concurrent_reprojections = (int(np.floor(total_available_memory / memory_burden_per_image)))

    logging.info ("Stacked Image Megapixels        : " + str(megapixels))
    logging.info ("Memory Burden Per Image         : " + str(memory_burden_per_image))
    logging.info ("Total Available Memory          : " + str(total_available_memory))
    logging.info ("Maximum concurrent reprojections: " + str(concurrent_reprojections))

    if concurrent_reprojections > int( min(math.floor(os.cpu_count() * 0.25), maximum_cpus_ever)):
        logging.info ("However, we only have " +str(int( min(math.floor(os.cpu_count() * 0.25), maximum_cpus_ever))) + " CPUS to use! So that is the maximum.")
        concurrent_reprojections =  int( min(math.floor(os.cpu_count() * 0.25), maximum_cpus_ever))
        if concurrent_reprojections==len(fileList) - 1:
            logging.info ("But that is only one CPU less than the total amount we need, so just sneakily adding an extra one in!")
            concurrent_reprojections=concurrent_reprojections+1

    if concurrent_reprojections == 0:
        total_available_memory_extreme = (psutil.virtual_memory().total/1000000000) * 0.8
        concurrent_reprojections_extreme = (int(np.floor(total_available_memory_extreme / memory_burden_per_image)))
        if concurrent_reprojections_extreme == 0:
            logging.info ("Stack too big to stack")
            return
        
    # Image reprojection array
    pldrivetempfile=memmappath + '/tempfile' +str(str(np.random.randint(10000)))+str(newheader['BJD'])
    PLDrive = np.memmap(pldrivetempfile, dtype='float32', mode= 'w+', shape = (shape_out[0],shape_out[1],len(inputList)))

    # Variance reprojection array
    variance_pldrivetempfile=memmappath + '/tempfilevariance' +str(str(np.random.randint(10000)))+str(newheader['BJD'])
    variance_PLDrive = np.memmap(variance_pldrivetempfile, dtype='float32', mode= 'w+', shape = (shape_out[0],shape_out[1],len(inputList)))


    # This needs to be subprocesses rather than multiprocessing as
    # the reproject command cannot do parallel processing in parallel processing
    # conditions. Running it on it's own doesn't use all of the available
    # CPU and memory - wasting valuable time. So multiple subprocesses it is.
    logging.info ("**********************************")
    logging.info ("Smart Stack Processing............")
    logging.info(datetime.now().strftime("%H:%M:%S"))

    # Chunking fileList into concurrent chunklets

    start = 0
    end = len(fileList)
    step = max(concurrent_reprojections,1)
    counter=0
    logging.info (range(start,end,step))
    for i in range(start, end, step):
        x = i
        chunk_of_list=(fileList[x:x+step])
        logging.info (len(chunk_of_list))
        if len(chunk_of_list) !=0:
            logging.info ("processing chunk: " + str(chunk_of_list))
            logging.info(datetime.now().strftime("%H:%M:%S"))
            procs=[]
            proccounter=0

            # To do the uncertainty array properly, we need to inform the reprojector of the appropriate frames to combine
            # 
            if telescope == 'lco':
                readnoise_variance_file=lco_variance_frame_filename
            else:
                readnoise_variance_file=calibration_directory + '/' + newheader['INSTRUME'] + '/' + newheader['TELID'] +'_' + newheader['INSTRUME'] + '_readnoise_variance_adu.npy'
            bias_file=calibration_directory + '/' + newheader['INSTRUME'] + '/' + newheader['TELID'] +'_' + newheader['INSTRUME'] + '_BIAS_master_bin1.npy'
            
            # smartstacks are currently set to 30s and 90s, so use requested exposure to distinguish
            if newheader['EXPREQ'] > 30: # narrowband biasdark
                biasdark_file=calibration_directory + '/' + newheader['INSTRUME'] + '/' + newheader['TELID'] +'_' + newheader['INSTRUME'] + '_narrowbandssBIASDARK_master_bin1.npy'
            else: # broadband biasdark
                biasdark_file=calibration_directory + '/' + newheader['INSTRUME'] + '/' + newheader['TELID'] +'_' + newheader['INSTRUME'] + '_broadbandssBIASDARK_master_bin1.npy'
           
            for file in chunk_of_list:
                procs.append(subprocess.Popen(['python3',os.path.expanduser(codedir) +'/subprocesses/reprojectprocess.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0))
                pickle.dump([file.replace('.npy','.fits'),counter,wcs_out,shape_out,readnoise_variance_file, bias_file, biasdark_file, telescope == 'lco' ], procs[proccounter].stdin)
                counter=counter+1
                proccounter=proccounter+1

            for p in procs:
                p.wait()

        logging.info(datetime.now().strftime("%H:%M:%S"))

    # We also need to reproject the variance frame for each individual reprojection.
    # If no sigmaclipped, we median the reprojection together and 
    # We reduce the variance by c * N, where c is 1.57 if we assume gaussian noise.
    #
    # If we sigmaclip median, then we use the same equation but we reduce N to the number of non-clipped frames   

    logging.info(datetime.now().strftime("%H:%M:%S"))
    logging.info ("**********************************")


    logging.info ("**********************************")
    logging.info ("Putting Image Files into memmap..........")
    logging.info(datetime.now().strftime("%H:%M:%S"))
    # Get all the files and dump them into the memmap
    counter=0
    slicedelete=[]
    for file in fileList:
        retrycounter=0
        finished=False
        while retrycounter <5 and not finished:
            try:
               
                array=np.load(file.replace('.npy','.'+str(counter)+'.reproject.npy'))
                finished=True
            except:
                logging.info ("problems loading npy: " + str(file.replace('.npy','.'+str(counter)+'.reproject.npy')))
                logging.info ("Waiting a moment to see if the filesystem clears")
                time.sleep(5)
                retrycounter=retrycounter+1
        if retrycounter >= 5:
            slicedelete.append(retrycounter)

        PLDrive[:,:,counter] = (array[:,:]) # Plonk the array in the memmap file
        counter=counter+1

    try:
        del array
    except:
        pass

    logging.info(datetime.now().strftime("%H:%M:%S"))
    logging.info ("**********************************")

    counter=0
    for file in fileList:
        try:
            os.remove(file.replace('.npy','.'+str(counter)+'.reproject.npy'))
        except:
            logging.info ("couldn't delete reproject file. Oh well!")
        counter=counter+1
        
    
    ########################### VARIANCE ARRAY
    
    logging.info(datetime.now().strftime("%H:%M:%S"))
    logging.info ("**********************************")


    logging.info ("**********************************")
    logging.info ("Putting Image Files into memmap..........")
    logging.info(datetime.now().strftime("%H:%M:%S"))
    # Get all the files and dump them into the memmap
    counter=0
    slicedelete=[]
    for file in fileList:
        retrycounter=0
        finished=False
        while retrycounter <5 and not finished:
            try:
               
                array=np.load(file.replace('.npy','.'+str(counter)+'.reprojectedvariance.npy'))
                finished=True
            except:
                logging.info ("problems loading npy: " + str(file.replace('.npy','.'+str(counter)+'.reprojectedvariance.npy')))
                logging.info ("Waiting a moment to see if the filesystem clears")
                time.sleep(5)
                retrycounter=retrycounter+1
        if retrycounter >= 5:
            slicedelete.append(retrycounter)

        variance_PLDrive[:,:,counter] = (array[:,:]) # Plonk the array in the memmap file
        counter=counter+1
    try:
        del array
    except:
        pass

    logging.info(datetime.now().strftime("%H:%M:%S"))
    logging.info ("**********************************")

    counter=0
    for file in fileList:
        try:
            os.remove(file.replace('.npy','.'+str(counter)+'.reprojectedvariance.npy'))
        except:
            logging.info ("couldn't delete reproject file. Oh well!")
        counter=counter+1
    
    # BACKGROUND MATCHING FOR IMAGE ARRAY ONLY
    logging.info ("Background Matching")

    tempimage=PLDrive[:,:,0][~np.isnan(PLDrive[:,:,0])]
    
    int_array_flattened=tempimage.ravel()[~np.isnan(tempimage.ravel())].astype(int)
    unique,counts=np.unique(int_array_flattened, return_counts=True)
    m=counts.argmax()
    stackMode=unique[m]
    logging.info ("First background Mode: " +str(stackMode))
    del tempimage

    start = 0
    end = len(fileList)
    step = concurrent_reprojections
    for i in range(start, end, step):
        x = i
        background_matching_arrays=[]
        for z in range(min(end-x,step)):
            logging.info (x+z)
            background_matching_arrays.append((PLDrive[:,:,x+z],stackMode))

        return_array=[]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with Pool( min(math.floor(os.cpu_count() * 0.25), maximum_cpus_ever)) as pool:
                for result in pool.map(background_match, background_matching_arrays):
                    return_array.append(result)

        del background_matching_arrays

        counter=0
        for resulting_image in return_array:
            PLDrive[:,:,x+counter]=resulting_image
            counter=counter+1
            logging.info ("Replacing: " +str(x+counter))

        del return_array

    ####################################### IMAGE MEDIAN STACKING
    while True:

        converged=1
        if len(inputList) < 5 and int(shape_out[0]) < 8000 and int(shape_out[1]) < 8000:
            logging.info ("**********************************")
            logging.info ("Median Stacking all arrays simultaneously from the Reprojections")
            logging.info(datetime.now().strftime("%H:%M:%S"))
            finalImage=bn.nanmedian(PLDrive, axis=2)#, overwrite_input=True)
            logging.info(datetime.now().strftime("%H:%M:%S"))
            logging.info ("**********************************")
        else:
            logging.info ("**********************************")
            logging.info ("Median Stacking each row individually from the Reprojections")
            logging.info(datetime.now().strftime("%H:%M:%S"))
            # Go through each pixel and calculate nanmedian. Can't do all arrays at once as it is hugely memory intensive
            finalImage=np.zeros(shape_out,dtype=float)
            totaltimer=time.time()
            mptask=[]
            counter=0
            for goog in range(shape_out[0]):
                mptask.append((pldrivetempfile,counter, (shape_out[0],shape_out[1],len(inputList))))
                counter=counter+1
    
            counter=0
            with Pool( min(math.floor(os.cpu_count() * 0.25), maximum_cpus_ever)) as pool:
                for result in pool.map(stack_nanmedian_row, mptask):
                    finalImage[counter,:]=result
                    counter=counter+1
    
            logging.info ("Total time to median: " + str(time.time() - totaltimer))
    
            logging.info(datetime.now().strftime("%H:%M:%S"))
            logging.info ("**********************************")
            
        counter=0
        differences=[]
        for image in range(len(inputList)):
            # Create a mask that is True only where both arrays are NOT NaN
            mask = ~np.isnan(finalImage) & ~np.isnan(PLDrive[:,:,counter])
            
            # Subtract only where both values are valid (i.e., not NaN)
            result = np.where(mask, finalImage - PLDrive[:,:,counter], np.nan)            
            
            difference=bn.nansum(result)
            if difference == 0.0:
                differences.append(np.nan)
            else:
                differences.append(difference)
            
            logging.info ("Difference for " + str(counter) + " is " + str(difference))
            counter=counter+1

        average_difference=np.nanmean(differences)
        stdev_difference=np.nanstd(differences)
        logging.info (counter)
        counter=0
        for image in range(len(inputList)):
            logging.info (counter)            
            logging.info ("difference out by " + str(differences[counter] - average_difference))
            logging.info ("stdev: " + str((differences[counter] - average_difference)/stdev_difference))
            
            if abs((differences[counter] - average_difference)/stdev_difference) > 3:
                logging.info ("REJECT!")
                PLDrive[:,:,counter].fill(np.nan)
                converged=0
            
            counter=counter+1
        
        if converged==1:
            break
        
        logging.info ("REDOING STACK AFTER REJECTION")

    PLDrive._mmap.close()
    del PLDrive
    gc.collect()
    os.remove(pldrivetempfile)
    
    
    ############ Variance Array Stack
    converged=1
    if len(inputList) < 5 and int(shape_out[0]) < 8000 and int(shape_out[1]) < 8000:
        logging.info ("**********************************")
        logging.info ("Median Stacking all arrays simultaneously from the Reprojections")
        logging.info(datetime.now().strftime("%H:%M:%S"))
        variance_finalImage=bn.nanmedian(variance_PLDrive, axis=2)#, overwrite_input=True)
        logging.info(datetime.now().strftime("%H:%M:%S"))
        logging.info ("**********************************")
    else:
        logging.info ("**********************************")
        logging.info ("Median Stacking each row individually from the Reprojections")
        logging.info(datetime.now().strftime("%H:%M:%S"))
        # Go through each pixel and calculate nanmedian. Can't do all arrays at once as it is hugely memory intensive
        variance_finalImage=np.zeros(shape_out,dtype=float)
        totaltimer=time.time()
        mptask=[]
        counter=0
        for goog in range(shape_out[0]):
            mptask.append((variance_pldrivetempfile,counter, (shape_out[0],shape_out[1],len(inputList))))
            counter=counter+1

        counter=0
        with Pool( min(math.floor(os.cpu_count() * 0.25), maximum_cpus_ever)) as pool:
            for result in pool.map(stack_nanmedian_row, mptask):
                variance_finalImage[counter,:]=result
                counter=counter+1


        # We reduce the variance by c * N, where c is 1.57 if we assume gaussian noise.
        variance_finalImage=variance_finalImage / (1.57 * len(inputList))

        logging.info ("Total time to median: " + str(time.time() - totaltimer))

        logging.info(datetime.now().strftime("%H:%M:%S"))
        logging.info ("**********************************")            

    variance_PLDrive._mmap.close()
    del variance_PLDrive
    gc.collect()
    os.remove(variance_pldrivetempfile)
    logging.info (wcs_out)

    # This long list gets rid of wacky WCS keys that can stay over and mess up the WCS in the stack.
    wcskeydelete=['WCSRFCAT','WCSIMCAT','WWCSNREF','WCSMATCH','WCCATTYP','WCNTERMS','WCSRDRES','WCSDELRA','WCSDELDE','WCSERR','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2','QV1_0','QV1_1','QV1_7','QV1_9','QV1_17','QV1_21','QV1_31','QV1_37','QV2_0','QV2_1','QV2_7','QV2_9','QV2_17','QV2_21','QV2_31','QV2_37','WCSAXES','EQUINOX','LONPOLE','LATPOLE','CUNIT1','CUNIT2','IMAGEW','IMAGEH','A_ORDER','A_0_0','A_0_1','A_0_2','A_1_0','A_1_1','A_1_2','A_2_0','B_ORDER','B_0_0','B_0_1','B_0_2','B_1_0','B_1_1','B_1_2','B_2_0','BP_ORDER','BP_0_0','BP_0_1','BP_0_2','BP_1_0','BP_1_1','BP_1_2','BP_2_0','AP_ORDER','AP_0_0','AP_0_1','AP_0_2','AP_1_0','AP_1_1','AP_1_2','AP_2_0','CDELT1','CDELT2','PC1_1','PC1_2','PC2_1','PC2_2','MJDREF','A_0_3','A_0_4','A_1_3','A_2_1','A_2_2','A_3_0','A_3_1','A_4_0','B_0_3','B_0_4','B_1_3','B_2_1','B_2_2','B_3_0','B_3_1','B_4_0','AP_0_3','AP_0_4','AP_1_3','AP_2_1','AP_2_2','AP_3_0','AP_3_1','AP_4_0','BP_0_3','BP_0_4','BP_1_3','BP_2_1','BP_2_2','BP_3_0','BP_3_1','BP_4_0','CRPIX1','CRPIX2','CUNIT1','CUNIT2','CTYPE1','CTYPE2','CRVAL1','CRVAL2','LONPOLE','LATPOLE','RADESYS','EQUINOX']
    for wcskey in wcskeydelete:
        if wcskey in list(newheader.keys()):
            del newheader[wcskey]

    newheader.update(wcs_out.to_header(relax=True))

    logging.info(newheader['CRPIX1'])
    logging.info(newheader['CRPIX2'])

    # Due to smartstack dithering, the edges are dodgy, so we need to remove them. 50 pixels dithering plus about 25 pixels to account for telescope drift
    # But due to reprojecting, this value needs to be expanded by the relative initial to final pixelscale

    zoomfactor=originalpixelscale/float(newPixScale / u.arcsec)
    cropvalue= int((75 * zoomfactor)+1)

    logging.info ('largest image rotation: ' + str(largest_image_rotation))

    # Due to image rotation, there will be nasty nan edges. We can calculate this
    # and crop it out from the known image rotations from the incoming WCS solutions
    nan_pixels_due_to_image_rotation = (math.tan(largest_image_rotation * (math.pi/180)) * max([finalImage.shape[0],finalImage.shape[1]]))
    logging.info ('Nan pixels due to image rotation: ' + str(nan_pixels_due_to_image_rotation))
    cropvalue=int(cropvalue + nan_pixels_due_to_image_rotation)

    logging.info ('Smartstack zoom factor: ' + str(zoomfactor))
    logging.info ('Smartstack crop value: ' + str(cropvalue))

    finalImage = finalImage[cropvalue:-cropvalue,cropvalue:-cropvalue]
    variance_finalImage = variance_finalImage[cropvalue:-cropvalue,cropvalue:-cropvalue]
    # Need to remove 75 pixels from CRPIX and 150 pixels from NAXIS as the pipeline crops 75 pixels out around the outside
    if any("CRPIX1" in s for s in newheader.keys()):
        newheader['CRPIX1']=newheader['CRPIX1']-cropvalue
        newheader['CRPIX2']=newheader['CRPIX2']-cropvalue
    newheader['NAXIS1']=newheader['NAXIS1']-(2*cropvalue)
    newheader['NAXIS2']=newheader['NAXIS2']-(2*cropvalue)
    if any("IMAGEH" in s for s in newheader.keys()):
        newheader['IMAGEH']=newheader['IMAGEH']-(2*cropvalue)
        newheader['IMAGEW']=newheader['IMAGEW']-(2*cropvalue)

    # Step Integer from mode to reject low values to np.nan
    try:
        finalImage, imageMode = thresh(finalImage, newheader['SATURATE'])
    except:
        imageMode=np.nan
        
    # # 3.0 is a reasonable read_noise
    try:
        readnoise_string=float(newheader['RDNOISE'])
    except:
        readnoise_string=3.0
    
    # We just set the gain to 1 to process the file as is. 
    try:
        gain_string=float(newheader['GAIN'])
    except:
        gain_string=1.0
    if 'T' in str(newheader['DOCOSMIC']) and not np.isnan(imageMode):
        try:
            finalImage = mask_cosmics(finalImage, gain=gain_string, rdnoise=readnoise_string, saturate=newheader['SATURATE'], imageMode=imageMode, pixscale=newheader['PIXSCALE'], telescopename='smartstack')
        except:
            logging.info ("did not do cosmics. Usually 'cause the gain, readnoise or pixel scale is unknown")

    # Interpolate any remaining nans
    finalImage=interpolate_replace_nans(finalImage,kernel)

    #Scaling image appropriately
    expHolder=np.asarray(expHolder)

    n_of_frames=len(fileList)
    newheader['EFFEXPN'] = n_of_frames
    #Updating  Header
    newheader['EFFEXPT']=np.sum(expHolder)
    newheader['EXPTIME']=np.median(expHolder)
    try:
        newheader['BJD']=bjd / (float(n_of_frames))
        newheader['TDB']=bjd / (float(n_of_frames))
        newheader['JD-MID']=jdmid / (float(n_of_frames))
        newheader['JD']=jd / (float(n_of_frames))
        newheader['MJD-MID']=mjdmid / (float(n_of_frames))
        newheader['MJD-OBS']=mjd / (float(n_of_frames))
        newheader['AIRMASS']=airMass / (float(n_of_frames))
        expTime=newheader['EXPTIME']
        effExpTime=newheader['EFFEXPT']
        #Tidying up Name
        expTimeF=(("%.3f" % expTime).replace('.','d'))
        effExpTimeF=(("%.3f" % effExpTime).replace('.','d'))
        mjdObsF = ("%.8f" % (bjd / float(n_of_frames))).replace('.','d')
        airMassF=("%.3f" % (airMass / (float(n_of_frames)))).replace('.','a')
    except:
        logging.info(traceback.format_exc())


    filterTwo=filterTwo.replace('NOTPRESENT','NP')
    filterThree=filterThree.replace('NOTPRESENT','NP')

    # PIXSCALE!!

    newName='SmSTACK-'+objectTemp + "_" + filterOne + "_" +filterTwo+ "_" + filterThree + "_" + mjdObsF + "_" + dateObs + "_" + airMassF + "_" +  expTimeF +'e'+ effExpTimeF + "_" + instruMe + "_" + userId + "_" + telId + ".fits"
    logging.info (newName.replace(' ','').replace('_NP_NP_','_'))
    # Report number of nans in array
    logging.info ("Number of nan pixels in image array: " + str(np.count_nonzero(np.isnan(finalImage))))
    base = Path(basedirectory)
    dest = base / 'sstacksdirectory' / newName.replace(' ','').replace('_NP_NP_','_')
    fits.writeto(dest, (np.asarray(finalImage).astype(np.float32)), newheader, output_verify='silentfix')
    var_dest = base / 'sstacksdirectory' / ('variance_' + newName.replace(' ','').replace('_NP_NP_','_'))
    fits.writeto(var_dest, (np.asarray(variance_finalImage).astype(np.float32)), newheader, output_verify='silentfix')
    
    del finalImage
    del variance_finalImage

# Function to mitigate S&P noise in a 2D array
def mitigate_salt_and_pepper_noise(image, threshold=12, imageMode=200, fwhm=12, pixscale=0.5):
    """
    Mitigates salt-and-pepper noise in a 2D NumPy array similar to process Alarcon et al. 2023. 

    Parameters:
    - image: 2D NumPy array representing the raw image.
    - threshold: Absolute difference threshold for masking outliers (default: 12 ADU).

    Returns:
    - denoised_image: The corrected image with S&P noise mitigated.
    - mask: A boolean mask where True indicates the pixels identified as S&P noise.
    """
    
    
    # Remove stars from the image as bright sources are not heavily affected by salt and pepper noise
    
    # Mask NaN and Inf values
    nanmask = np.isfinite(image)   
    
    # Estimate the noise level (standard deviation of the background)
    noise_level = mad_std(image[nanmask])
    image_star_masked=detect_and_mask_stars(imagedata=image,fwhm=fwhm,noise_level=noise_level, pixscale=pixscale)
    
    
    # Apply a 3x3 median filter to create the reference frame
    smoothed_image = median_filter(image, size=3)
    
    # Calculate the absolute difference between the raw and smoothed image
    difference = np.abs(image_star_masked - smoothed_image)

    while True:
        # Create a mask for S&P noise (difference greater than the threshold)
        # But then not so high it affects signal from objects
        mask = (difference > threshold) & (difference < 5 * threshold)
    
        # Replace noisy pixels with values from the smoothed image
        denoised_image = image.copy()
        denoised_image[mask] = smoothed_image[mask]
        
        number_of_salt_and_pepper_pixels=np.sum(mask)
        fraction_of_salt_and_pepper_pixels=number_of_salt_and_pepper_pixels / image.size
        
        logging.info ("Fraction of S&P: " + str(fraction_of_salt_and_pepper_pixels))
        logging.info ("Threshold      : " + str(threshold))
        
        
        if fraction_of_salt_and_pepper_pixels < 0.025:
            break
        
        threshold=threshold+1

    return denoised_image, mask, smoothed_image

def stack_nanmedian_row(inputinfo):
    (pldrivetempfiletemp,counter,shape) = inputinfo
    tempPLDrive = np.memmap(pldrivetempfiletemp, dtype='float32', mode= 'r', shape = shape )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)       
        
        # Perform sigma clipping
        clipped_data = sigma_clip(tempPLDrive[counter,:,:], sigma=2, maxiters=5, axis=0)        
        
        # Compute the median along the third dimension, ignoring masked values
        return bn.nanmedian(clipped_data, axis=1)

def nanmedian_row(row):
    try:
        return bn.nanmedian(row, axis=1)
    except:

        logging.info(traceback.format_exc())


def background_match(backdata):
    try:
        (image, stackMode) = backdata
        tempimage = image[~np.isnan(image)]
        int_array_flattened=tempimage.ravel()[~np.isnan(tempimage.ravel())].astype(int)
        unique,counts=np.unique(int_array_flattened, return_counts=True)
        m=counts.argmax()
        diffStackMode=unique[m] - stackMode
        logging.info ("Difference between images: " +str(diffStackMode))
        return image - diffStackMode
    except:

        logging.info(traceback.format_exc())