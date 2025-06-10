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
from modules.image_functions import find_nan_crop_limits, thresh, mask_cosmics
import os
from pathlib import Path
import traceback
from astropy.io import fits
from astropy import wcs
from astropy.io.fits import Header
import time
import subprocess
from astropy.table import Table
import math
from modules.platesolving import get_source_spread_through_catalogue
from astropy.stats import sigma_clip
import bottleneck as bn
import pickle
import shutil
import glob
import logging

def process_preastrom(file):

    """
    Processes an astronomical image file for astrometric calibration.

    Parameters:
        file (str): Path to the input file containing image data and parameters encoded in the filename.

    Processing Steps:
        1. Extracts parameters (exposure time, pixel scale, saturation level, cosmic ray processing flag, gain, and read noise) from the filename.
        2. Loads the image data and crops rows/columns containing NaN values.
        3. Applies thresholding to reject low-value pixels based on the saturation threshold.
        4. Optionally removes cosmic rays if exposure time is ≥ 10 seconds and cosmic ray removal is flagged True.
        5. Saves the processed image as a `.npy` file with the prefix 'FLATTED'.
        6. Deletes the original file after successful processing.

    Notes:
        - If cosmic ray removal parameters (gain, read noise, pixel scale) are invalid or missing, cosmic ray removal is skipped, and a warning is printed.
        - Processing errors or exceptions (such as thresholding failures) are caught and reported without interrupting the entire process.

    Returns:
        None
    """    

    logging.info(file)

    hdu1 = np.load(file)
    exptime = float(file.split('EXP')[1])
    pixscale = float(file.split('PIXSCALE')[1])
    image_saturation_level = float(file.split('SATURATE')[1])
    # If it has been substacked, then it has already been reduced at site and only needs the astrometry.
    docosmic=str(file.split('DOCOSMIC')[1])
    if 'T' in docosmic:
        docosmic=True
    else:
        docosmic=False
    gain = float(file.split('GAIN')[1])
    rdnoise = float(file.split('RDNOISE')[1])

    objdeflat=np.asarray(hdu1.data).astype(np.float32)
    firstrowcrop,lastrowcrop,firstcolcrop,lastcolcrop = find_nan_crop_limits(objdeflat)
    objdeflat=np.asarray(hdu1.data)[firstrowcrop:lastrowcrop,firstcolcrop:lastcolcrop]

    try:
        # Step Integer from mode to reject low values to np.nan
        objdeflat, imageMode = thresh(objdeflat, image_saturation_level)
        # At this point whack in cosmics. But not for short exposures where it is more harm than good
        if exptime >= 10 and docosmic:            
            try:                
                hdu1 = mask_cosmics(hdu1, gain=gain, rdnoise=rdnoise, saturate=image_saturation_level, imageMode=imageMode, pixscale=pixscale, telescopename='other')
            except:
                logging.info ("did not do cosmics. Usually 'cause the gain, readnoise or pixel scale is unknown")

        out_path = Path(file).with_name('FLATTED' + file.split('/')[-1].split('FILTER')[-1])
        np.save(out_path, objdeflat)
        del hdu1
        try:
            os.remove(file)
        except:
            pass
    except:
        logging.info("Failed to threshold this image")
        logging.info(traceback.format_exc())

def process_lco_preastrom(file):
    """
    Processes astronomical image files specifically from LCO (Las Cumbres Observatory) telescopes, preparing them for astrometric calibration.

    Parameters:
        file (str): Path to the input file containing image data and parameters encoded in the filename.

    Processing Steps:
        1. Extracts key parameters from the filename:
            - Exposure time (exptime)
            - Pixel scale (pixscale)
            - Image saturation level
            - Gain
            - Read noise (rdnoise)
        2. Loads the image data from the file.
        3. Applies thresholding to reject low-value pixels, setting them to NaN, based on saturation threshold.
        4. Performs cosmic ray removal (masking) if the exposure time is ≥ 10 seconds.
        5. Saves the processed image to a `.npy` file, prefixing the filename with 'FLATTED'.
        6. Deletes the original file upon successful processing.

    Notes:
        - Assumes image files adhere strictly to a naming convention embedding required parameters (e.g., EXP, PIXSCALE, SATURATE, GAIN, RDNOISE).
        - If thresholding or cosmic ray masking fails, the exception traceback is printed, but processing continues gracefully.

    Returns:
        None
    """

    logging.info(file)

    hdu1 = np.load(file)
    exptime = float(file.split('EXP')[1])
    pixscale = float(file.split('PIXSCALE')[1])
    image_saturation_level = float(file.split('SATURATE')[1])
    gain = float(file.split('GAIN')[1])
    rdnoise = float(file.split('RDNOISE')[1])

    try:

        # Step Integer from mode to reject low values to np.nan
        hdu1, imageMode = thresh(hdu1, image_saturation_level)
        
        # At this point whack in cosmics. 
        if exptime >= 10:
            hdu1 = mask_cosmics(hdu1, gain=gain, rdnoise=rdnoise, saturate=image_saturation_level, imageMode=imageMode, pixscale=pixscale, telescopename='lco')
        out_path = Path(file).with_name('FLATTED' + file.split('/')[-1].split('FILTER')[-1])
        np.save(out_path, hdu1)
        try:
            os.remove(file)
        except:
            pass
    except:
        logging.info("Failed to threshold this image")
        logging.info("Tends not to be a fatal error but due to a mostly blank image")
        out_path = Path(file).with_name('FLATTED' + file.split('/')[-1].split('FILTER')[-1])
        np.save(out_path, hdu1)
        try:
            os.remove(file)
        except:
            pass
        logging.info(traceback.format_exc())


def run_astrometry_net(file, codedir):
    """
    Performs astrometric calibration (plate solving) on astronomical image data using Astrometry.net.

    Parameters:
        file (str): Path to the input file with encoded metadata, including estimated coordinates, pixel scale, and image dimensions.

    Processing Steps:
        1. Extracts metadata directly from the filename:
            - Estimated Right Ascension (RAest)
            - Estimated Declination (DECest)
            - Pixel scale (arcsec/pixel)
            - Image dimensions (width, height)
        2. Creates a temporary working directory for intermediate files.
        3. Converts `.npy` image data to a temporary `.fits` file.
        4. Executes Source Extractor to detect sources and generate an object catalog, filtering for high signal-to-noise ratio sources.
        5. Selects the optimal subset of sources based on spatial distribution analysis.
        6. Measures the median sky level and computes the median and standard deviation of the FWHM (Full Width at Half Maximum) of detected stars, applying sigma clipping to exclude outliers.
        7. Stores the computed image statistics (FWHM, sky level, number of detected stars) in a `.fwhm` pickle file.
        8. Performs astrometric solving with Astrometry.net (`solve-field`) using adaptive tweak-order strategies:
            - Initially attempts a standard tweak-order (2 or 3) depending on image size.
            - If unsuccessful, retries with alternative tweak-orders, including higher-order adjustments.
        9. If successful, moves the resulting World Coordinate System (`.wcs`) file to the original working directory.
        10. Cleans up temporary files and directories after processing.

    Output Files:
        - `.wcs` file: Astrometric solution describing image coordinates.
        - `.fwhm` file: Pickle file containing image statistics and quality metrics.

    Notes:
        - Assumes Source Extractor and Astrometry.net are properly installed and configured.
        - Includes comprehensive error handling to gracefully handle and report failures without stopping subsequent processes.

    Returns:
        None
    """

    logging.info(file)
    RAest = float(file.split('RAdeg')[1])
    DECest = float(file.split('DECdeg')[1])
    pixscale = float(file.split('PIXSCALE')[1])
    imageh = int(file.split('IMAGEH')[1])
    imagew = int(file.split('IMAGEW')[1])
    pixlow = pixscale * 0.9
    pixhigh = pixscale * 1.1
    base_dir = Path(file).parent
    base_root = base_dir.parent
    orig_name = Path(file).name.split('PIXSCALE')[-1].replace('.npy', '')
    tempdir = base_dir / (str(time.time()).replace('.', 'd') + file.split('PIXSCALE')[-1].replace('.', 'd') + str(np.random.randint(10000)))
    tempdir.mkdir(parents=True, exist_ok=True)
    gain = float(file.split('GAIN')[1])
    rdnoise = float(file.split('RDNOISE')[1])
    image_saturation_level = float(file.split('SATURATE')[1])

    # Dump out a temporary image
    astrom_name = Path(file).name.split('PIXSCALE')[-1].replace('.npy','.fits')
    astromfitsfile = base_dir / astrom_name

    dest_wcs = base_root / (orig_name.replace('.fits.fz', '').replace('.fits', '').replace('.npy', '') +'.wcs')
    dest_fwhm = base_root / (orig_name.replace('.fits.fz', '').replace('.fits', '').replace('.npy', '') +'.fwhm')    

    wcs_header = None
    cleanhdu=fits.PrimaryHDU()
    cleanhdu.data=np.array(np.load(file.replace('.fits','.npy').replace('ASTROMTABLE','FLATTED')))
    cleanhdu.writeto(astromfitsfile)

    # run source extractor on image
    tempprocess = subprocess.Popen(
        ['source-extractor', astromfitsfile, '-c', os.path.expanduser(codedir) +'/photometryparams/default.sexfull', '-PARAMETERS_NAME', str(os.path.expanduser(codedir) +'/photometryparams/default.paramastrom'),
         '-CATALOG_NAME', str(tempdir / 'test.cat'), '-SATUR_LEVEL', str(image_saturation_level), '-GAIN', str(gain), '-READNOISE', str(rdnoise), '-BACKPHOTO_TYPE','LOCAL', '-DETECT_THRESH', str(1.5), '-ANALYSIS_THRESH',str(1.5),
         '-SEEING_FWHM', str(2.0), '-FILTER_NAME', str(os.path.expanduser(codedir) +'/photometryparams/sourceex_convs/gauss_2.0_5x5.conv')], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, bufsize=0)
    tempprocess.wait() 
        
    # Read the ASCII catalog
    acatalog = Table.read(tempdir / "test.cat", format='ascii')
    # Reject poor  ( <10 SNR) sources
    acatalog=acatalog[acatalog['FLUX_AUTO']/acatalog['FLUXERR_AUTO'] > 10]


    # If there are significant number of sources in the image, we should run a more precise psf approach to platesolving
    if len(acatalog) > 150:
        seeingfwhm=np.median(np.array(acatalog['FWHM_IMAGE'])) * pixscale
        photapertures=max(5.0/float(pixscale),3)
        
        # Calculate broadly minimum area from the seeing and the pixelscale
        fwhmpixels= seeingfwhm / pixscale
        psfarea= math.pi * pow((fwhmpixels /2),2)
        minarea= 1.0 * psfarea    
        backsize= 4 * fwhmpixels
    
        tempprocess=subprocess.Popen(['source-extractor' , astromfitsfile ,'-c',os.path.expanduser(codedir) +'/photometryparams/default.sexfull', '-PARAMETERS_NAME', str(os.path.expanduser(codedir) +'/photometryparams/default.paramprepsx'), '-CATALOG_NAME',str(tempdir / 'psf.cat'),'-CATALOG_TYPE','FITS_LDAC','-SATUR_LEVEL', str(image_saturation_level) , '-DETECT_THRESH', str(2.5), '-ANALYSIS_THRESH',str(2.5),'-BACKPHOTO_TYPE','LOCAL', '-BACK_SIZE', str(backsize), '-BACK_FILTERSIZE',str(4), '-DETECT_MINAREA', str(minarea), '-GAIN',str(gain),'-SEEING_FWHM',str(seeingfwhm),'-PHOT_APERTURES', str(photapertures),'-FILTER_NAME', str(os.path.expanduser(codedir) +'/photometryparams/sourceex_convs/gauss_2.0_5x5.conv')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
        tempprocess.wait()
        
        tempprocess=subprocess.Popen(['psfex', str(tempdir / 'psf.cat'), '-c',os.path.expanduser(codedir) +'/photometryparams/default.psfex','-CHECKPLOT_DEV','NULL','-CHECKIMAGE_TYPE','NONE','-PSF_DIR',str(tempdir)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
        tempprocess.wait()
        
        photapertures=max(3.0/float(pixscale),3)
        
        tempprocess=subprocess.Popen(['source-extractor','-PSF_NAME',str(tempdir / 'psf.psf'), astromfitsfile ,'-c',os.path.expanduser(codedir) +'/photometryparams/default.sexfull', '-PARAMETERS_NAME', str(os.path.expanduser(codedir) +'/photometryparams/default.paramactualpsx'),'-CATALOG_NAME',str(tempdir / 'psf.cat'),'-CATALOG_TYPE','ASCII', '-BACKPHOTO_TYPE','LOCAL', '-BACK_SIZE', str(backsize), '-BACK_FILTERSIZE',str(4), '-DETECT_THRESH', str(2.5), '-ANALYSIS_THRESH',str(2.5), '-DETECT_MINAREA', str(minarea),'-SATUR_LEVEL', str(image_saturation_level) ,'-GAIN',str(gain),'-PHOT_APERTURES', str(photapertures),'-FILTER_NAME', str(os.path.expanduser(codedir) +'/photometryparams/sourceex_convs/gauss_2.0_5x5.conv')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
        tempprocess.wait()

        # pick up the catalog again and trim it up
        acatalog = Table.read(tempdir / "psf.cat", format='ascii')
        
        # Remove index, RA and DEC columns
        acatalog.remove_columns(acatalog.colnames[:3])
        
        # Get the FWHM column from the end
        # Identify the second last column by index
        second_last_column_name = acatalog.colnames[-2]  # -2 for the second last column
        
        # Make the fifth column a direct copy of the second last column
        acatalog['col8'] = acatalog[second_last_column_name]
        
        # Keep only the first 5 columns using slicing
        acatalog = acatalog[acatalog.colnames[:5]]        
        
        # Renaming columns
        new_column_titles = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FWHM_IMAGE']
        for i, new_title in enumerate(new_column_titles):
            acatalog.rename_column(acatalog.colnames[i], new_title)
        
    # Make sure sources cover all of the image
    numpy_sources=np.column_stack((np.array(acatalog['X_IMAGE']), np.array(acatalog['Y_IMAGE']), np.array(acatalog['FLUX_AUTO'])))
    source_spreads, source_catalogues=get_source_spread_through_catalogue(numpy_sources)

    # Get the median variation in the array and the stdev
    if len(source_spreads) > 1:
        median_variation=np.median(np.asarray(source_spreads)[:,1])

        median_stdev=np.std(np.asarray(source_spreads)[:,1])

        # Then select the brightest catalogue that is within half a standard
        # Deviation of the median variation
        catalogue_number=0
        for entry in source_spreads:
            if entry[1] > (median_variation + 0 * median_stdev):
                pass
            else:
                break
            catalogue_number=catalogue_number+1
    else:
        catalogue_number = 0

    sources={'X_IMAGE': source_catalogues[catalogue_number][:,0],'Y_IMAGE': source_catalogues[catalogue_number][:,1],'FLUX_AUTO': source_catalogues[catalogue_number][:,2]}

    sources=Table(sources)

    # Write out to fits
    sources.write(tempdir / "test.fits", format="fits", overwrite=True)
    
    fwhm_values = acatalog['FWHM_IMAGE']  # Extract FWHM values
    
    # Remove NaN and zero values
    fwhm_values = fwhm_values[~np.isnan(fwhm_values)]  # Remove NaN
    fwhm_values = fwhm_values[fwhm_values > 0]  # Remove zero values

    # Apply sigma clipping with a 3-sigma threshold
    clipped_fwhm = sigma_clip(fwhm_values, sigma=3, maxiters=5, cenfunc='median', stdfunc='std')
    
    # Get the clipped values (remove masked elements)
    filtered_fwhm = clipped_fwhm[~clipped_fwhm.mask]
    
    fwhmpix=bn.nanmedian(filtered_fwhm)
    fwhmstd=bn.nanstd(filtered_fwhm)
    
    picklefwhm={}
    try:
        picklefwhm["SKYLEVEL"] = (bn.nanmedian(cleanhdu.data), "Sky Level without pedestal")
        picklefwhm["FWHM"] = (fwhmpix, 'FWHM in pixels')
        picklefwhm["FWHMpix"] = (fwhmpix, 'FWHM in pixels')
        picklefwhm["FWHMasec"] = (pixscale*fwhmpix, 'FWHM in arcseconds')
        picklefwhm["FWHMstd"] = (pixscale*fwhmstd, 'FWHM standard deviation in arcseconds')
        picklefwhm["NSTARS"] = ( len(filtered_fwhm), 'Number of star-like sources in image')

        temp_fwhm = base_dir / (Path(file).name.split('PIXSCALE')[-1].replace('.npy','.fwhm'))
        with open(temp_fwhm, 'wb') as fp:
            pickle.dump(picklefwhm, fp)
        shutil.move(str(temp_fwhm), dest_fwhm)
    
    except:
        logging.info(traceback.format_exc())
        
        picklefwhm["SKYLEVEL"] = (bn.nanmedian(cleanhdu.data), "Sky Level without pedestal")

        picklefwhm["FWHM"] = (-99, 'FWHM in pixels')
        picklefwhm["FWHMpix"] = (-99, 'FWHM in pixels')
        picklefwhm["FWHMasec"] = (-99, 'FWHM in arcseconds')
        picklefwhm["FWHMstd"] = ( -99, 'FWHM standard deviation in arcseconds')
        picklefwhm["NSTARS"] = ( len(filtered_fwhm ), 'Number of star-like sources in image')
        
        temp_fwhm = base_dir / (Path(file).name.split('PIXSCALE')[-1].replace('.npy','.fwhm'))
        with open(temp_fwhm, 'wb') as fp:
            pickle.dump(picklefwhm, fp)
        shutil.move(str(temp_fwhm), dest_fwhm)

    # Use tweak order 2 in smaller fields of view and tweak order 3 in larger fields.
    sizewidest= max(imageh*pixscale, imagew*pixscale) / 3600
    
    if sizewidest > 1.0:
        tweakorder=[3,2]
    else:
        tweakorder=[2,3]
    
    # Try once with tweak-order [0]   
    os.system("/usr/local/astrometry/bin/solve-field " + str(tempdir / 'test.fits') + " -D " + str(tempdir) + " --x-column X_IMAGE --y-column Y_IMAGE --sort-column FLUX_AUTO --crpix-center --tweak-order " + str(tweakorder[0]) + " --width " + str(imagew) + " --height " + str(imageh) + " --scale-units arcsecperpix --scale-low " + str(pixlow) + " --scale-high " + str(pixhigh) + " --scale-units arcsecperpix --ra " + str(RAest) + " --dec " + str(DECest) + " --radius 10 --cpulimit 300 --depth 1-100 --overwrite --no-verify --no-plots ")
    
    if os.path.exists(tempdir / 'test.wcs'):
        logging.info("A successful solve for " + str(astromfitsfile))
        os.remove(astromfitsfile)
        shutil.move(str(tempdir / 'test.wcs'), dest_wcs)
        hdr= fits.open(dest_wcs)[0].header
        wcs_header = wcs.WCS(hdr).to_header(relax=True)
    else:
        # Try once with tweak-order [1]    
        os.system("/usr/local/astrometry/bin/solve-field " + str(tempdir / 'test.fits') + " -D " + str(tempdir) + " --x-column X_IMAGE --y-column Y_IMAGE --sort-column FLUX_AUTO --crpix-center --tweak-order " + str(tweakorder[1]) + " --width " + str(imagew) + " --height " + str(imageh) + " --scale-units arcsecperpix --scale-low " + str(pixlow) + " --scale-high " + str(pixhigh) + " --scale-units arcsecperpix --ra " + str(RAest) + " --dec " + str(DECest) + " --radius 10 --cpulimit 300 --depth 1-100 --overwrite --no-verify --no-plots ")
        
        if os.path.exists(tempdir / 'test.wcs'):
            logging.info("A successful solve for " + str(astromfitsfile))
            os.remove(astromfitsfile)
            shutil.move(str(tempdir / 'test.wcs'), dest_wcs)
            hdr= fits.open(dest_wcs)[0].header
            wcs_header = wcs.WCS(hdr).to_header(relax=True)
        else:
            # Try once with tweak-order 4    
            os.system("/usr/local/astrometry/bin/solve-field " + str(tempdir / 'test.fits') + " -D " + str(tempdir) + " --x-column X_IMAGE --y-column Y_IMAGE --sort-column FLUX_AUTO --crpix-center --tweak-order 4 --width " + str(imagew) + " --height " + str(imageh) + " --scale-units arcsecperpix --scale-low " + str(pixlow) + " --scale-high " + str(pixhigh) + " --scale-units arcsecperpix --ra " + str(RAest) + " --dec " + str(DECest) + " --radius 10 --cpulimit 300 --depth 1-100 --overwrite --no-verify --no-plots ")
            
            if os.path.exists(tempdir / 'test.wcs'):
                logging.info("A successful solve for " + str(astromfitsfile))
                os.remove(astromfitsfile)
                shutil.move(str(tempdir / 'test.wcs'), dest_wcs)
                hdr= fits.open(dest_wcs)[0].header
                wcs_header = wcs.WCS(hdr).to_header(relax=True)
            else:
                logging.info("A failed solve for " + str(astromfitsfile))
                os.remove(astromfitsfile)                   

    for f in tempdir.glob('*'):
        try:
            f.unlink()
        except Exception:
            pass

    try:
        tempdir.rmdir()
    except Exception:
        pass

    # Remove masters directory (again, ignore errors)
    try:
        shutil.rmtree(tempdir, ignore_errors=True)
    except Exception:
        pass

    return orig_name, wcs_header
