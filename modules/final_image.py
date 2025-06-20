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

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import median_absolute_deviation
import bottleneck as bn

import sep
import copy
import os
from pathlib import Path
from modules.image_functions import calculate_image_fwhm, mid_stretch_jpeg
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import EarthLocation
from datetime import  timedelta
import traceback
import logging
import json
kernel = Gaussian2DKernel(x_stddev=2,y_stddev=2)


def _radial_profile(image, center, radius):
    """Return radii and pixel values around ``center`` up to ``radius`` pixels."""
    y, x = np.indices(image.shape)
    dx = x - center[0]
    dy = y - center[1]
    r = np.sqrt(dx ** 2 + dy ** 2)
    mask = r <= radius
    if not np.any(mask):
        return []
    radii = r[mask].ravel().astype(float)
    values = image[mask].ravel().astype(float)
    return list(map(lambda rv: [float(rv[0]), float(rv[1])], zip(radii, values)))


def _moffat(r, I0, alpha, beta):
    return I0 * (1 + (r / alpha) ** 2) ** (-beta)


def _fit_moffat(pairs):
    if len(pairs) == 0:
        return None
    radii = np.array([p[0] for p in pairs])
    fluxes = np.array([p[1] for p in pairs])
    try:
        p0 = [float(fluxes.max()), 1.0, 1.5]
        popt, _ = curve_fit(_moffat, radii, fluxes, p0=p0, maxfev=10000)
        return {
            'I0': float(popt[0]),
            'alpha': float(popt[1]),
            'beta': float(popt[2])
        }
    except Exception:
        return None


def multiprocess_final_image_construction_smartstack(file, base):

    tempheader=fits.open(file)[0]
    imagedata=tempheader.data
    tempheader=tempheader.header
    
    imagedata=interpolate_replace_nans(imagedata,kernel)

    # Calculate Image FWHM unless this is a variance frame
    if not Path(file).name.startswith('variance_'):
        sepimg=copy.deepcopy(imagedata).byteswap().newbyteorder()
        try:
            bkg = sep.Background(sepimg, bw=32, bh=32, fw=3, fh=3)
            bkg.subfrom(sepimg)
        except Exception:
            logging.info("Failed background (usually flat) for image: %s", file)

        try:
            tempheader = calculate_image_fwhm(sepimg, tempheader)
        except Exception:
            logging.info("Failed FWHM (usually blank) for image: %s", file)

    # Offset bias pedestal
    imagedata = imagedata + 200
    try:
        tempheader['PEDESTAL']= tempheader['PEDESTAL']+200
    except:
        tempheader['PEDESTAL']=200

    tempheader['RLEVEL']=96

    filenameonly = Path(file).name
    tempheader['SJPGNAME']= filenameonly.replace('.fits','.jpg').replace('EVA-','smalljpg-').replace('SmSTACK-','smalljpgSmSTACK-')
    tempheader['LJPGNAME']= filenameonly.replace('.fits','.jpg').replace('EVA-','previewjpg-').replace('SmSTACK-','previewjpgSmSTACK-')
    tempheader['TJPGNAME']=filenameonly.replace('.fits','.jpg').replace('EVA-','thumbnail-').replace('SmSTACK-','thumbnailSmSTACK-')
    tempheader['PNGNAME']=filenameonly.replace('.fits','.png').replace('EVA-','png-').replace('SmSTACK-','pngSmSTACK-')
    tempheader['PSXNAME']=filenameonly.replace('.fits','.psx').replace('EVA-','psxphot-').replace('SmSTACK-','psxphotSmSTACK-')
    tempheader['SEANAME']=filenameonly.replace('.fits','.sea').replace('EVA-','seaphot-').replace('SmSTACK-','seaphotSmSTACK-')
    tempheader['SEKNAME']=filenameonly.replace('.fits','.sek').replace('EVA-','sekphot-').replace('SmSTACK-','sekphotSmSTACK-')

    # Basic image statistics
    try:
        int_array = imagedata.ravel()[~np.isnan(imagedata.ravel())].astype(int)
        if int_array.size:
            unique, counts = np.unique(int_array, return_counts=True)
            mode_val = unique[counts.argmax()]
        else:
            mode_val = np.nan
    except Exception:
        mode_val = np.nan

    tempheader['IMGMIN'] = (bn.nanmin(imagedata), 'Minimum Value of Image Array')
    tempheader['IMGMAX'] = (bn.nanmax(imagedata), 'Maximum Value of Image Array')
    tempheader['IMGMEAN'] = (bn.nanmean(imagedata), 'Mean Value of Image Array')
    tempheader['IMGMODE'] = (mode_val, 'Mode Value of Image Array')
    tempheader['IMGMED'] = (bn.nanmedian(imagedata), 'Median Value of Image Array')
    tempheader['IMGMAD'] = (median_absolute_deviation(imagedata, ignore_nan=True), 'Median Absolute Deviation of Image Array')
    tempheader['IMGSTDEV'] = (bn.nanstd(imagedata), 'Standard Deviation of Image Array')
    tempheader['QANAME']=filenameonly.replace('.fits','.json').replace('EVA-','quickanalysis-').replace('SmSTACK-','quickanalysisSmSTACK-')
    tempheader['QANAME']=filenameonly.replace('.fits','.json').replace('EVA-','quickanalysis-').replace('SmSTACK-','quickanalysisSmSTACK-')

    dest = Path(base) / 'outputdirectory' / filenameonly
    fits.writeto(dest, imagedata, tempheader, output_verify='silentfix', overwrite=True)

    os.remove(file)

def multiprocess_final_image_construction_single_image(file, tempheader, humanreadablename, base):

    path = Path(base) / 'workingdirectory' / file
    imagedata=np.load(path)

    # For each image, simply crop 20 pixels around the edge
    # crop image
    imagedata = imagedata[20:-20,20:-20]
    # Need to remove 20 pixels from CRPIX and 40 pixels from NAXIS as the pipeline crops 20 pixels out around the outside
    if any("CRPIX1" in s for s in tempheader.keys()):
        tempheader['CRPIX1']=tempheader['CRPIX1']-20
        tempheader['CRPIX2']=tempheader['CRPIX2']-20
    tempheader['NAXIS1']=tempheader['NAXIS1']-40
    tempheader['NAXIS2']=tempheader['NAXIS2']-40
    if any("IMAGEH" in s for s in tempheader.keys()):
        tempheader['IMAGEH']=tempheader['IMAGEH']-40
        tempheader['IMAGEW']=tempheader['IMAGEW']-40

    imagedata=interpolate_replace_nans(imagedata,kernel)

    # Offset bias pedestal
    imagedata = imagedata + 200

    try:
        tempheader['PEDESTAL']= tempheader['PEDESTAL']+200
    except:
        tempheader['PEDESTAL']=200

    logging.info (humanreadablename)
    filenameonly=humanreadablename
    tempheader['SJPGNAME']= filenameonly.replace('.fits','.jpg').replace('EVA-','smalljpg-').replace('SmSTACK-','smalljpgSmSTACK-')
    tempheader['LJPGNAME']= filenameonly.replace('.fits','.jpg').replace('EVA-','previewjpg-').replace('SmSTACK-','previewjpgSmSTACK-')
    tempheader['TJPGNAME']=filenameonly.replace('.fits','.jpg').replace('EVA-','thumbnail-').replace('SmSTACK-','thumbnailSmSTACK-')
    tempheader['PNGNAME']=filenameonly.replace('.fits','.png').replace('EVA-','png-').replace('SmSTACK-','pngSmSTACK-')
    tempheader['PSXNAME']=filenameonly.replace('.fits','.psx').replace('EVA-','psxphot-').replace('SmSTACK-','psxphotSmSTACK-')
    tempheader['SEANAME']=filenameonly.replace('.fits','.sea').replace('EVA-','seaphot-').replace('SmSTACK-','seaphotSmSTACK-')
    tempheader['SEKNAME']=filenameonly.replace('.fits','.sek').replace('EVA-','sekphot-').replace('SmSTACK-','sekphotSmSTACK-')

    logging.info ("Interpolated and saved file: " + str(humanreadablename))
    dest = Path(base) / 'outputdirectory' / humanreadablename
    fits.writeto(dest, imagedata, tempheader, overwrite=True)

    del imagedata

def multiprocess_preview_images(file):

    hdusmalldata = np.array(fits.open(file)[0].data)

    # Any remaining nan's replace with mode
    if np.count_nonzero(np.isnan(hdusmalldata)) > 0:
        unique,counts=np.unique(hdusmalldata.ravel()[~np.isnan(hdusmalldata.ravel())].astype(int), return_counts=True)
        m=counts.argmax()
        tempMode=unique[m]
        hdusmalldata=np.nan_to_num(hdusmalldata, nan=tempMode, posinf=tempMode, neginf=tempMode)

    hdusmalldata = hdusmalldata - np.nanmin(hdusmalldata)

    stretched_data_float = mid_stretch_jpeg(hdusmalldata+1000)
    stretched_256 = 255 * stretched_data_float
    hot = np.where(stretched_256 > 255)
    cold = np.where(stretched_256 < 0)
    stretched_256[hot] = 255
    stretched_256[cold] = 0
    stretched_data_uint8 = stretched_256.astype("uint8")
    hot = np.where(stretched_data_uint8 > 255)
    cold = np.where(stretched_data_uint8 < 0)
    stretched_data_uint8[hot] = 255
    stretched_data_uint8[cold] = 0

    iy, ix = stretched_data_uint8.shape
    final_image = Image.fromarray(stretched_data_uint8)

    final_image.save(file.replace('EVA-','previewjpg-').replace('SmSTACK-','previewjpgSmSTACK-').replace('LoSTACK-','previewjpgLoSTACK-').replace('outputdirectory','previews').replace('.fits','.jpg'), format="JPEG", quality=80, optimize=True)

    if 'SmSTACK' in file:
        # Set up to save out a png image
        png_image= copy.deepcopy(final_image)
        png_image=png_image.convert("L")       
        
        # Get the original dimensions
        original_width, original_height = png_image.size
        
        # Determine the scaling factor
        if original_width > original_height:
            # Landscape orientation: width is the longest axis
            new_width = 5000
            new_height = int((5000 / original_width) * original_height)
        else:
            # Portrait orientation: height is the longest axis
            new_height = 5000
            new_width = int((5000 / original_height) * original_width)
            
        # Resize the image
        png_image = png_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        png_image.save(file.replace('EVA-','png-').replace('SmSTACK-','pngSmSTACK-').replace('LoSTACK-','pngLoSTACK-').replace('outputdirectory','previews').replace('.fits','.png'), optimize=True, format="PNG", compress_level=9, transparency=None)
    
    # UI Sized JPEG
    if iy == ix:
            ui_image = final_image.resize(
                (900, 900)
            )
    elif iy > ix:
            ui_image = final_image.resize(

                (int(900 * iy / ix), 900)

            )
    else:
        ui_image = final_image.resize((900, int(900 * iy / ix)))

    ui_image.save(file.replace('EVA-','smalljpg-').replace('SmSTACK-','smalljpgSmSTACK-').replace('LoSTACK-','smalljpgLoSTACK-').replace('outputdirectory','smalljpgs').replace('.fits','.jpg'), format="JPEG", quality=80, optimize=True)


    if iy == ix:
            thumbnail_image = final_image.resize(
                (300, 300)
            )
    elif iy > ix:
            thumbnail_image = final_image.resize(

                (int(300 * iy / ix), 300)

            )
    else:
        thumbnail_image = final_image.resize((300, int(300 * iy / ix)))

    thumbnail_image.save(file.replace('EVA-','thumbnail-').replace('SmSTACK-','thumbnailSmSTACK-').replace('LoSTACK-','thumbnailLoSTACK-').replace('outputdirectory','thumbnails').replace('.fits','.jpg'), format="JPEG", quality=80, optimize=True)

def make_banzai_file_out_of_EVA(file, telescope, basedirectory, calibration_directory):
    
    # Open the fits file early and if it isn't being ingested, don't make the BZESK file!
    tempopen=fits.open(file)[0]    
    eva_image_header=tempopen.header    
    if eva_image_header['EXPTIME'] < 1:
        logging.info ("Not making BZESK file as it isn't a file to be ingested")
        return

    psxfile=False
    sekfile=False

    # First check that there is an appropriate psx photometry file
    tempfilename = file.replace('outputdirectory', 'photometry').replace('.fits', '.psx').replace('EVA-','psxphot-').replace('SmSTACK-', 'psxphotSmSTACK-').replace('LoSTACK-', 'psxphotLoSTACK-')
    if os.path.exists(tempfilename):
        temp_banzai_photometry_filename=tempfilename

        # Check that there actually is a good set of sources in psx.
        # if it is less than 10, it is sorta a problem
        # also psfex still reports even if the PSF did not fit well, so shoots out zero fluxes
        source_counter=0
        try:
            with open(str(temp_banzai_photometry_filename.replace('photometry', 'fullphotcatalogues').replace('.psx','.fullpsx').replace('.sek','.fullsek')), 'r') as f:

                for line in f:
                    if '#' not in line:
                        # Remove spaces from line
                        line=line.split(' ')
                        list_line=[]
                        for entry in line:
                            entry=entry.replace('\n','')
                            if entry == '':
                                pass
                            else:
                                list_line.append(float(entry))
                        line=list_line

                        if float(line[5]) > 0:
                            source_counter=source_counter+1
            logging.info ("psx source counts: " + str(source_counter))
            if source_counter > 0:
                psxfile=True
        except UnicodeDecodeError as e:
            logging.error(f"Could not read photometry catalogue {temp_banzai_photometry_filename}: {e}")
            return

    if not psxfile:
        # If no psx photometry - potentially due to lack of sources, then check for sek
        tempfilename = file.replace('outputdirectory', 'photometry').replace('.fits', '.sek').replace('EVA-','sekphot-').replace('SmSTACK-', 'sekphotSmSTACK-').replace('LoSTACK-', 'sekphotLoSTACK-')
        if os.path.exists(tempfilename):
            temp_banzai_photometry_filename=tempfilename
            sekfile=True
        else:
            temp_banzai_photometry_filename='none'

        ## WILL NEED TO REPORT PSSC or SEK IN HEADER SOMEWHERE

    if not temp_banzai_photometry_filename=='none':
        temp_banzai_photometry_filename=temp_banzai_photometry_filename.replace('photometry', 'fullphotcatalogues').replace('.psx','.fullpsx').replace('.sek','.fullsek')
        
        # Glorious Pile of lists for BANZAI
        ra=[]
        dec=[]
        xcentroid=[]
        ycentroid=[]
        xcentroid_win=[]
        ycentroid_win=[]
        maxval_xindex=[]
        maxval_yindex=[]
        max_value=[]
        semimajor_sigma_value=[]
        semiminor_sigma_value=[]
        theta=[]
        ellipticity=[]
        kron_radius=[]
        flux=[]
        flux_err=[]
        covar_sigx2=[]
        covar_sigy2=[]
        covar_sigxy=[]
        background=[]
        flag=[]
        fwhm_image=[]
        # errx2_image=[]
        # erry2_image=[]
        # errxy_image=[]

        if psxfile:
            logging.info ("psxfile")
        else:
            logging.info ("sekfile")
        # set up the list of sources
        try:
            with open(str(temp_banzai_photometry_filename), 'r') as f:

                for line in f:
                    if '#' not in line:

                        # Remove spaces from line
                        line=line.split(' ')
                        list_line=[]
                        for entry in line:
                            entry=entry.replace('\n','')
                            if entry == '':
                                pass
                            else:
                                list_line.append(float(entry))
                        line=list_line

                        try:
                            if psxfile and (float(line[5]) > 0): # PSX can sometime present zero values
                                ra.append(float(line[1]))
                                dec.append(float(line[2]))
                                xcentroid.append(line[9])
                                ycentroid.append(line[10])
                                xcentroid_win.append(line[3])
                                ycentroid_win.append(line[4])
                                maxval_xindex.append(line[11])
                                maxval_yindex.append(line[12])
                                max_value.append(line[13])
                                semimajor_sigma_value.append(line[14])
                                semiminor_sigma_value.append(line[15])
                                theta.append(line[16])
                                ellipticity.append(line[17])
                                kron_radius.append(line[26])
                                flux.append(line[5])
                                flux_err.append(line[6])
                                covar_sigx2.append(line[18])
                                covar_sigy2.append(line[19])
                                covar_sigxy.append(line[20])
                                background.append(line[21])
                                flag.append(line[22])
                                fwhm_image.append(line[25])
                                # errx2_image.append(line[27])
                                # erry2_image.append(line[28])
                                # errxy_image.append(line[29])

                            elif sekfile and (float(line[7]) > 0): # SEK can sometime present negative flux values
                                ra.append(float(line[1]))
                                dec.append(float(line[2]))
                                xcentroid.append(line[11])
                                ycentroid.append(line[12])
                                xcentroid_win.append(line[3])
                                ycentroid_win.append(line[4])
                                maxval_xindex.append(line[13])
                                maxval_yindex.append(line[14])
                                max_value.append(line[15])
                                semimajor_sigma_value.append(line[16])
                                semiminor_sigma_value.append(line[17])
                                theta.append(line[18])
                                ellipticity.append(line[19])
                                kron_radius.append(line[28])
                                flux.append(line[7])
                                flux_err.append(line[8])
                                covar_sigx2.append(line[20])
                                covar_sigy2.append(line[21])
                                covar_sigxy.append(line[22])
                                background.append(line[23])
                                flag.append(line[24])
                                fwhm_image.append(line[27])
                                # errx2_image.append(line[29])
                                # erry2_image.append(line[30])
                                # errxy_image.append(line[31])

                        except:
                            logging.info ("Failed photometry layer of BZESK")
                            logging.info (temp_banzai_photometry_filename)
                            if psxfile:
                                logging.info("failed psx line: " + str(line[5]))
                                logging.info("failed psx line: " + str(line))
                            else:
                                logging.info("failed sek line: " + str(line[7]))
                                logging.info("failed sek line: " + str(line))
                            
                            
                    # flags (off initially but to be embedded)
                    #flag.append(0)
                    # Flag = 1 for sources with bad pixels
                    # Flag = 2 for sources that are deblended
                    # Flag = 4 for sources that have saturated pixels
                    # Flag = 8 if kron aperture falls off the image
                    # Flag = 16 if source has cosmic ray pixels

                    # # Fill in meaningless values
                    # for d in range(1, 7):
                    #     radius_arcsec = d / image.pixel_scale / 2.0
                    #     sources[f'fluxaper{d}'], sources[f'fluxerr{d}'] = catalog.circular_photometry(radius_arcsec)

                    # for r in [0.25, 0.5, 0.75]:
                    #     sources['fluxrad' + f'{r:.2f}'.lstrip("0.")] = catalog.fluxfrac_radius(r)

        except UnicodeDecodeError as e:
            logging.error(f"Could not read photometry file {temp_banzai_photometry_filename}: {e}")
            return

        # Create empty astropy table intended to create BANZAI catalogue
        sources = Table({'ra': ra, 'dec': dec, 'x': xcentroid, 'y': ycentroid,
                             'xwin': xcentroid_win, 'ywin': ycentroid_win,
                             'xpeak': maxval_xindex, 'ypeak': maxval_yindex,
                             'peak': max_value,
                             'a': semimajor_sigma_value, 'b': semiminor_sigma_value,
                             'theta': theta, 'ellipticity': ellipticity,
                             'kronrad': kron_radius,
                             'flux': flux, 'fluxerr': flux_err,
                             'x2': covar_sigx2, 'y2': covar_sigy2,
                             'xy': covar_sigxy,
                             'background': background, 'fwhm': fwhm_image, 'flag': flag})#, 'x2': errx2_image, 'y2': erry2_image, 'xy': errxy_image})

        # Add the units and description to the catalogs
        sources['ra'].unit = 'degree'
        sources['ra'].description = 'RA of the object'
        sources['dec'].unit = 'degree'
        sources['dec'].description = 'Dec of the object'

        sources['x'].unit = 'pixel'
        sources['x'].description = 'X coordinate of the object'
        sources['y'].unit = 'pixel'
        sources['y'].description = 'Y coordinate of the object'
        sources['xwin'].unit = 'pixel'
        sources['xwin'].description = 'Windowed X coordinate of the object'
        sources['ywin'].unit = 'pixel'
        sources['ywin'].description = 'Windowed Y coordinate of the object'
        sources['xpeak'].unit = 'pixel'
        sources['xpeak'].description = 'X coordinate of the peak'
        sources['ypeak'].unit = 'pixel'
        sources['ypeak'].description = 'Windowed Y coordinate of the peak'
        sources['flux'].unit = 'count'
        sources['flux'].description = 'Flux within a Kron-like elliptical aperture'
        sources['fluxerr'].unit = 'count'
        sources['fluxerr'].description = 'Error on the flux within Kron aperture'
        sources['peak'].unit = 'count'
        sources['peak'].description = 'Peak flux (flux at xpeak, ypeak)'
        # for diameter in [1, 2, 3, 4, 5, 6]:
        #     sources['fluxaper{0}'.format(diameter)].unit = 'count'
        #     sources['fluxaper{0}'.format(diameter)].description = 'Flux from fixed circular aperture: {0}" diameter'.format(diameter)
        #     sources['fluxerr{0}'.format(diameter)].unit = 'count'
        #     sources['fluxerr{0}'.format(diameter)].description = 'Error on Flux from circular aperture: {0}"'.format(diameter)

        sources['background'].unit = 'count'
        sources['background'].description = 'Average background value in the aperture'
        sources['fwhm'].unit = 'pixel'
        sources['fwhm'].description = 'FWHM of the object'
        # sources['fwtm'].unit = 'pixel'
        # sources['fwtm'].description = 'Full-Width Tenth Maximum'
        sources['a'].unit = 'pixel'
        sources['a'].description = 'Semi-major axis of the object'
        sources['b'].unit = 'pixel'
        sources['b'].description = 'Semi-minor axis of the object'
        sources['theta'].unit = 'degree'
        sources['theta'].description = 'Position angle of the object'
        sources['kronrad'].unit = 'pixel'
        sources['kronrad'].description = 'Kron radius used for extraction'
        sources['ellipticity'].description = 'Ellipticity'
        # sources['fluxrad25'].unit = 'pixel'
        # sources['fluxrad25'].description = 'Radius containing 25% of the flux'
        # sources['fluxrad50'].unit = 'pixel'
        # sources['fluxrad50'].description = 'Radius containing 50% of the flux'
        # sources['fluxrad75'].unit = 'pixel'
        # sources['fluxrad75'].description = 'Radius containing 75% of the flux'
        sources['x2'].unit = 'pixel^2'
        sources['x2'].description = 'Variance on X coordinate of the object'
        sources['y2'].unit = 'pixel^2'
        sources['y2'].description = 'Variance on Y coordinate of the object'
        sources['xy'].unit = 'pixel^2'
        sources['xy'].description = 'XY covariance of the object'
        sources['flag'].description = 'Bit mask of extraction/photometry flags'

        sources.sort('flux')
        sources.reverse()

        # Create the HDUList to hold the BZ-ESK file
        bzesque_file=fits.HDUList()
        
        # Convert from ADU to e- here
        banzai_image=tempopen.data * eva_image_header['GAIN']

        cleanhdu=fits.PrimaryHDU()
        banzai_image_header=cleanhdu.header

        # banzai_image_header['EXTNAME'] = 'SCI'
        # banzai_image_header['RLEVEL'] = 91
        try:
            # Try to mimic as close as possible the BANZAI header.
            try:
                banzai_image_header['ORIGIN']  = (eva_image_header['ORIGIN']          ,'Organization responsible for the data')
            except:
                banzai_image_header['ORIGIN']  = ('COMMUNITY   '           ,'Organization responsible for the data')
            banzai_image_header['SITEID']  = (eva_image_header['SITEID']         ,'ID code of the Observatory site')
            try:
                banzai_image_header['SITE']    = ( eva_image_header['SITE'] , 'Site of the Observatory')
            except:
                banzai_image_header['SITE']    = ( 'Photon Ranch Site','Site of the Observatory')
            try:
                banzai_image_header['ENCID']   = (eva_image_header['ENCID']           ,'ID code of the Enclosure')
                banzai_image_header['ENCLOSUR']= (eva_image_header['ENCLOSUR']           ,'Building containing Telescope')
            except:
                banzai_image_header['ENCID']   = ('UNKNOWN'          ,'ID code of the Enclosure')
                banzai_image_header['ENCLOSUR']= ('UNKNOWN'           ,'Building containing Telescope')
            banzai_image_header['TELID']   = (eva_image_header['TELID']           ,'ID code of the Telescope')
            banzai_image_header['TELESCOP']= (eva_image_header['TELESCOP']            ,'The Name of the Telescope')
            banzai_image_header['LATITUDE']= (float(eva_image_header['LATITUDE']) ,'[deg North] Telescope Latitude')
            banzai_image_header['LONGITUD']= (float(eva_image_header['LONGITUD'])  ,'[deg East] Telescope Longitude')
            banzai_image_header['HEIGHT']  = (float(eva_image_header['HEIGHT'])  ,'[m] Altitude of Telescope above sea level')

            # Calculate OBSGEO from EarthLocation
            obsgeo = EarthLocation.from_geodetic(eva_image_header['LONGITUD'], eva_image_header['LATITUDE'], eva_image_header['HEIGHT']).to_geocentric()
            banzai_image_header['OBSGEO-X']= (         float(str(obsgeo[0]).replace(' m','')) ,'[m] Cartesian X co-ord of telescope (WGS84)')
            banzai_image_header['OBSGEO-Y']= (         float(str(obsgeo[1]).replace(' m','')) ,'[m] Cartesian Y co-ord of telescope (WGS84)')
            banzai_image_header['OBSGEO-Z']= (         float(str(obsgeo[2]).replace(' m','')) ,'[m] Cartesian Z co-ord of telescope (WGS84)')

            banzai_image_header['OBSTYPE'] = (eva_image_header['OBSTYPE']            ,'Observation type')
            banzai_image_header['FRAMENUM']= (eva_image_header['FRAMENUM'] ,'Running frame number')
            banzai_image_header['MOLTYPE'] = ( 'EXPOSE  '           ,'Molecule type')
            banzai_image_header['MOLNUM']  = (                   1 ,'Molecule number')
            banzai_image_header['MOLFRNUM']= (                   1 ,'Exposure number within molecule')
            banzai_image_header['FRMTOTAL']= (                  1 ,'Total number of exposures within molecule')
            banzai_image_header['ORIGNAME']= ( eva_image_header['ORIGNAME'] ,'Fname written by ICS')
            banzai_image_header['OBSTELEM']= ( 'N/A     '           ,'Link to observation telemetry')
            banzai_image_header['TIMESYS'] = ( 'UTC     '           ,'Time system used')
            banzai_image_header['DATE']    = ( eva_image_header['DATE'] ,'[UTC] Date this FITS file was written')
            banzai_image_header['DATE-OBS']= ( eva_image_header['DATE-OBS'] ,'[UTC] Start date and time of the observation')
            banzai_image_header['DAY-OBS'] = ( eva_image_header['DAY-OBS']           ,'[UTC] Date at start of local observing night')

            # Calculate this
            temptime=Time(eva_image_header['DATE-OBS'],format='isot',scale='utc')
            banzai_image_header['UTSTART'] = ( str(temptime).split('T')[1]       ,'[UTC] The start time of the observation')
            temptime=temptime+ timedelta(seconds=float(eva_image_header['EXPTIME']))
            banzai_image_header['UTSTOP']  = ( str(temptime).split('T')[1]      ,'[UTC] The finish time of the observation')


            banzai_image_header['MJD-OBS'] = ( eva_image_header['MJD-OBS'] ,'[UTC days] Start date/time (Modified Julian Date)')
            banzai_image_header['EXPTIME'] = ( eva_image_header['EXPTIME'],'[s] Actual exposure length')
            if 'EFFEXPT' in eva_image_header:
                banzai_image_header['EFFEXPT'] = (
                    eva_image_header['EFFEXPT'],
                    '[s] Effective open shutter exposure time',
                )
            if 'EFFEXPN' in eva_image_header:
                banzai_image_header['EFFEXPN'] = (
                    eva_image_header['EFFEXPN'],
                    'Number of exposures contributing to the stack',
                )
            try:
                banzai_image_header['REQTIME'] = ( eva_image_header['REQTIME'] ,'[s] Requested exposure length')
            except:
                banzai_image_header['REQTIME'] = ( eva_image_header['EXPREQ'] ,'[s] Requested exposure length')


            # Convert this into BANZAI form
            # ACTUALLY THERE IS SOME NOT QUITE CLEAR INCOMPATIBILITY BETWEEN LCO AND PTR definitions, just put standard until this is sorted out.
            banzai_image_header['SUBREAD'] = (                    False ,'Is this a stacked (subread) image')
            banzai_image_header['NSUBREAD']= (                    1 ,'Number of stacked (subread) images')
            banzai_image_header['SEXPTIME']= (           eva_image_header['EXPTIME'] ,'[s] Subread exposure length')

            try:
                if eva_image_header['SMARTSTK'] == 'no' or not eva_image_header['SMARTSTK']:
                    banzai_image_header['SMARTSTK'] = (                    False ,'Is this part of an EVA smartstack')
                    banzai_image_header['SSTKLEN'] = (                     1  ,'Number of intended smartstack components')
                    banzai_image_header['SSTKNUM'] = (                    1 ,'ID of smartstack component')
                else:
                    banzai_image_header['SMARTSTK'] = (                    True ,'Is this part of an EVA smartstack')
                    banzai_image_header['SSTKLEN'] = (                     eva_image_header['SSTKLEN']  ,'Number of intended smartstack components')
                    banzai_image_header['SSTKNUM'] = (                    eva_image_header['SSTKNUM'] ,'ID of smartstack component')
                banzai_image_header['SSTKID'] = ( eva_image_header.get('SSTKID', eva_image_header['SMARTSTK']) ,'Smartstack identifier')
            except:
                banzai_image_header['SMARTSTK'] = (                    False ,'Is this part of an EVA smartstack')
                banzai_image_header['SSTKLEN'] = (                     1  ,'Number of intended smartstack components')
                banzai_image_header['SSTKNUM'] = (                    1 ,'ID of smartstack component')
                banzai_image_header['SSTKID'] = ( eva_image_header.get('SSTKID', eva_image_header.get('SMARTSTK','')) ,'Smartstack identifier')

            banzai_image_header['FILTER1'] = ( eva_image_header['FILTER']       ,'The first filter wheel filter type')
            banzai_image_header['FILTERI1']= ( 'N.I.Y.'        ,'The first filter wheel filter id')
            banzai_image_header['FILTER2'] = ( 'NOTPRESENT'         ,'The second filter wheel filter type')
            banzai_image_header['FILTERI2']= ( 'NOTPRESENT'         ,'The second filter wheel filter id')
            banzai_image_header['FILTER3'] = ( 'NOTPRESENT'         ,'The third filter wheel filter type')
            banzai_image_header['FILTERI3']= ( 'NOTPRESENT'         ,'The third filter wheel filter id')
            banzai_image_header['FILTER']  = ( eva_image_header['FILTER']       ,'Filter used')
            banzai_image_header['FWID']    = ( 'N.I.Y.'          ,'Filter Wheel ID')


            banzai_image_header['INSTRUME']= ( eva_image_header['INSTRUME']          ,'Instrument used')
            banzai_image_header['INSSTATE']= ( 'OKAY    '           ,'The instrument status')
            banzai_image_header['ICSVER']  = ( 'N.A.'   ,'Version number of the ICS software')


            banzai_image_header['CONFMODE']= ( 'full_frame'       ,'Camera mode configuration')
            banzai_image_header['CONFNAME']= ( 'N/A     '           ,'The instrument configuration used')


            banzai_image_header['DETECTOR']= ( eva_image_header['DETECTOR']         ,'Detector type')
            try:
                banzai_image_header['DETECTID']= ( eva_image_header['DETECTID']          ,'Detector serial number')
            except:
                banzai_image_header['DETECTID']= ( eva_image_header['CAMNAME']          ,'Detector serial number')

            # Normalise gain
            banzai_image_header['GAIN']    = ( 1.0 ,'[electrons/count] Pixel gain')
            banzai_image_header['RDNOISE'] = ( eva_image_header['RDNOISE'] * eva_image_header['GAIN'],'[electrons/pixel] Read noise')
            
            banzai_image_header['ORIGGAIN']    = ( eva_image_header['GAIN'] ,'[electrons/count] Original pixel gain')
            banzai_image_header['ORIGRDN'] = ( eva_image_header['RDNOISE'] ,'[ADU/pixel] Original Read noise')
            
            try:
                banzai_image_header['DARKCURR']= ( banzai_image_header['DARKCURR'] ,'[electrons/pixel/s @ 200K] Dark current')
            except:
                banzai_image_header['DARKCURR']= (            0.0100000 ,'[electrons/pixel/s @ 200K] Dark current')


            banzai_image_header['SATURATE']= ( eva_image_header['SATURATE'] * eva_image_header['GAIN'],'[ADU] Saturation level')
            banzai_image_header['MAXLIN']  = ( eva_image_header['MAXLIN'] * eva_image_header['GAIN'],'[ADU] Non-linearity level')
            banzai_image_header['RDSPEED'] = (          400.0000000 ,'[kpix/s] Readout speed used')
            banzai_image_header['DETSIZE'] = ('[1:' + str(eva_image_header['NAXIS1']) + ',1:' + str(eva_image_header['NAXIS2']) +']'    ,'[pixel] Detector size')
            banzai_image_header['AMPNAME'] = ('default '           ,'Amplifier name')


            ################ We don't do region of interest yet
            # banzai_image_header['CCDSEC']  = ( '[1:9600,1:6422]'    ,'[pixel] Region of CCD read')
            # banzai_image_header['CCDSUM']  = ( '1 1     '           ,'CCD on-chip summing/binning')
            # banzai_image_header['BIASSEC'] = ( 'N/A     '           ,'[binned pixel] Section of bias/overscan data')
            # banzai_image_header['DATASEC'] = ( '[1:2400,1:2400]'    ,'[binned pixel] Data section')
            # banzai_image_header['TRIMSEC'] = ( '[1:2400,1:2400]'    ,'[binned pixel] Section of useful data')
            # banzai_image_header['ROI']     = ( 'UNKNOWN '           ,'[binned pixel] Region of interest or MULTIPLE')
            # banzai_image_header['DETSEC']  = ( '[1:2400,1:2400]'    ,'[binned pixel] Section of useful data')
            banzai_image_header['CCDSEC']  = ( banzai_image_header['DETSIZE']    ,'[pixel] Region of CCD read')
            banzai_image_header['CCDSUM']  = ( '1 1     '           ,'CCD on-chip summing/binning')
            banzai_image_header['BIASSEC'] = ( 'N/A     '           ,'[binned pixel] Section of bias/overscan data')
            banzai_image_header['DATASEC'] = ( banzai_image_header['DETSIZE']    ,'[binned pixel] Data section')
            banzai_image_header['TRIMSEC'] = ( banzai_image_header['DETSIZE']    ,'[binned pixel] Section of useful data')
            banzai_image_header['ROI']     = ( 'UNKNOWN '           ,'[binned pixel] Region of interest or MULTIPLE')
            banzai_image_header['DETSEC']  = ( banzai_image_header['DETSIZE']    ,'[binned pixel] Section of useful data')

            # EVA is in um, BANZAI in m
            banzai_image_header['CCDXPIXE']= ( float(eva_image_header['CCDXPIXE'])/1000000 ,'[m] Size of pixels, in X')
            banzai_image_header['CCDYPIXE']= ( float(eva_image_header['CCDYPIXE'])/1000000 ,'[m] Size of pixels, in Y')

            banzai_image_header['PIXSCALE']= ( eva_image_header['PIXSCALE'] ,'[arcsec/pixel] Nominal pixel scale on sky')
            banzai_image_header['CCDSTEMP']= ( eva_image_header['CCDSTEMP'] ,'[deg C] CCD required temperature')
            banzai_image_header['CCDATEMP']= ( eva_image_header['CCDATEMP'] ,'[deg C] CCD actual temperature')
            banzai_image_header['CCDSESIG']= ( 'N/A     '           ,'[mK] CCD temp control servo error signal')
            banzai_image_header['DRKTCOEF']= (            0.0000000 ,'[1/deg C] Dark-scaling temperature coefficient')

            # Lots of currently irrelevant BANZAI stuff?
            banzai_image_header['TELMODE'] = ( 'AUTOMATIC'          ,'Telescope mode')
            banzai_image_header['TAGID']   = ( 'UNSPECIFIED'        ,'Time Allocation Group ID')
            banzai_image_header['USERID']  = ( eva_image_header['USERID']          ,'User ID')
            banzai_image_header['PROPID']  = ( eva_image_header['PROPID']   ,'Proposal ID')
            banzai_image_header['GROUPID'] = ( eva_image_header['GROUPID'] ,'Group ID')
            banzai_image_header['OBSID']   = ( eva_image_header['OBSID']        ,'Observation ID')
            banzai_image_header['OBSNOTE'] = ( 'UNSPECIFIED'        ,'Observation Note')

            # Info from the scheduler that may not be there if image not taken by the scheduler.
            try:
                banzai_image_header['SCHEDNAM']= ( eva_image_header['SCHEDNAM']          ,'Name of scheduler in control')
                banzai_image_header['TRACKNUM']= ( eva_image_header['TRACKNUM']             ,'Request DB tracking number')
                banzai_image_header['REQNUM']  = ( eva_image_header['REQNUM']             ,'Request DB request number')
                banzai_image_header['MOLUID']  = ( eva_image_header['MOLUID']           ,'Molecule unique ID')
                banzai_image_header['BLKTYPE'] = ( eva_image_header['BLKTYPE']            ,'Group type')
                banzai_image_header['BLKUID']  = ( eva_image_header['BLKUID']            ,'Group unique ID')
                banzai_image_header['BLKSDATE']= ( eva_image_header['BLKSDATE']   ,'[UTC] Block start date')
                banzai_image_header['BLKEDATE']= ( eva_image_header['BLKEDATE']   ,'[UTC] Block end date')
                banzai_image_header['BLKNOMEX']= ( eva_image_header['BLKNOMEX']  ,'[s] Block nominal exec time')
                banzai_image_header['BLKMNPH'] = ( eva_image_header['BLKMNPH']   ,'[(0-1)] Maximum lunar phase required')
                banzai_image_header['BLKMNDST']= ( eva_image_header['BLKMNDST']   ,'[deg] Minimum lunar distance required')
                banzai_image_header['BLKSEECO']= ( eva_image_header['BLKSEECO']             ,'Minimum seeing required')
                banzai_image_header['BLKTRNCO']= ( eva_image_header['BLKTRNCO']             ,'Minimum transparency required')
                banzai_image_header['BLKAIRCO']= ( eva_image_header['BLKAIRCO']             ,'Maximum airmass required')
                banzai_image_header['SCHEDSEE']= ( eva_image_header['SCHEDSEE']   ,'[arcsec] Estimated seeing when group scheduled')
                banzai_image_header['SCHEDTRN']= ( eva_image_header['SCHEDTRN']            ,'[(0-1)] Estimated transparency when group scheduled')
                banzai_image_header['TRIGGER'] = ( eva_image_header['TRIGGER']             ,'External trigger ID')
                banzai_image_header['MOSAIC']  = ( eva_image_header['MOSAIC']         ,'Mosaic Pattern Description')
                banzai_image_header['DITHER']  = ( eva_image_header['DITHER']          ,'Dither Pattern Description')
                banzai_image_header['OBRECIPE']= ( eva_image_header['OBRECIPE']            ,'Observing Recipes required/used')
                banzai_image_header['PCRECIPE']= ( eva_image_header['PCRECIPE']            ,'Processing Recipes required/used')
                banzai_image_header['PPRECIPE']= ( eva_image_header['PRRECIPE']            ,'Post-Processing Recipes required/used')
            except:
                banzai_image_header['SCHEDNAM']= ( 'COMMUNITY    '           ,'Name of scheduler in control')
                banzai_image_header['TRACKNUM']= ( 'N/A'           ,'Request DB tracking number')
                banzai_image_header['REQNUM']  = ( 'N/A'          ,'Request DB request number')
                banzai_image_header['MOLUID']  = ( 'N/A'         ,'Molecule unique ID')
                banzai_image_header['BLKTYPE'] = ( 'COMMUNITY    '           ,'Group type')
                banzai_image_header['BLKUID']  = ( 'N/A'         ,'Group unique ID')
                banzai_image_header['BLKSDATE']= ( 'N/A' ,'[UTC] Block start date')
                banzai_image_header['BLKEDATE']= ( 'N/A' ,'[UTC] Block end date')
                banzai_image_header['BLKNOMEX']= ( 'N/A' ,'[s] Block nominal exec time')
                banzai_image_header['BLKMNPH'] = ( 'N/A' ,'[(0-1)] Maximum lunar phase required')
                banzai_image_header['BLKMNDST']= ( 'N/A' ,'[deg] Minimum lunar distance required')
                banzai_image_header['BLKSEECO']= ( 'N/A     '           ,'Minimum seeing required')
                banzai_image_header['BLKTRNCO']= ( 'N/A     '           ,'Minimum transparency required')
                banzai_image_header['BLKAIRCO']= ( 'N/A'          ,'Maximum airmass required')
                banzai_image_header['SCHEDSEE']= ( 'N/A','[arcsec] Estimated seeing when group scheduled')
                banzai_image_header['SCHEDTRN']= ( 'N/A     '           ,'[(0-1)] Estimated transparency when group scheduled')
                banzai_image_header['TRIGGER'] = ( 'N/A     '           ,'External trigger ID')
                banzai_image_header['MOSAIC']  = ( 'UNSPECIFIED'        ,'Mosaic Pattern Description')
                banzai_image_header['DITHER']  = ( 'UNSPECIFIED'        ,'Dither Pattern Description')
                banzai_image_header['OBRECIPE']= ( 'N/A     '           ,'Observing Recipes required/used')
                banzai_image_header['PCRECIPE']= ( 'N/A     '           ,'Processing Recipes required/used')
                banzai_image_header['PPRECIPE']= ( 'N/A     '           ,'Post-Processing Recipes required/used')

            try:
                banzai_image_header['RA']      = ( eva_image_header['ORIGRA']       ,'[HH:MM:SS.sss] RA where telescope is pointing')
                banzai_image_header['DEC']     = ( eva_image_header['ORIGDEC']      ,'[sDD:MM:SS.ss] Dec where telescope is pointing')
            except:
                banzai_image_header['RA']      = ( eva_image_header['RA']       ,'[HH:MM:SS.sss] RA where telescope is pointing')
                banzai_image_header['DEC']     = ( eva_image_header['DEC']      ,'[sDD:MM:SS.ss] Dec where telescope is pointing')

            banzai_image_header['RADESYS'] = ( 'ICRS    '           ,'[[FK5,ICRS]] Fundamental coord. system of the o')
            try:
                banzai_image_header['LST']     = ( eva_image_header['LST']        ,'[HH:MM:SS.ss] LST at start of current observation')
            except:
                banzai_image_header['LST']     = ( 'N.Y.I.'        ,'[HH:MM:SS.ss] LST at start of current observation')
            banzai_image_header['CAT-RA']  = ( eva_image_header['CAT-RA']      ,'[HH:MM:SS.sss] Catalog RA of the object')
            banzai_image_header['CAT-DEC'] = ( eva_image_header['CAT-DEC']       ,'[sDD:MM:SS.ss] Catalog Dec of the object')
            banzai_image_header['CAT-EPOC']= (         2000.0000000 ,'[Year] Catalog epoch of the coordinates')
            banzai_image_header['OFST-RA'] = ( eva_image_header['OFST-RA']       ,'[HH:MM:SS.sss] Catalog RA plus pointing offsets')
            banzai_image_header['OFST-DEC']= ( eva_image_header['OFST-DEC']       ,'[sDD:MM:SS.ss] Catalog Dec plus pointing offsets')
            banzai_image_header['TPT-RA']  = ( eva_image_header['TPT-RA']       ,'[HH:MM:SS.sss] Telescope demand RA')
            banzai_image_header['TPT-DEC'] = ( eva_image_header['TPT-DEC']       ,'[sDD:MM:SS.ss] Telescope demand Dec')
            banzai_image_header['OBJECT']  = ( eva_image_header['OBJECT']         ,'Object name')
            try:
                banzai_image_header['SRCTYPE'] = ( eva_image_header['SRCTYPE']          ,'Source type')
            except:
                banzai_image_header['SRCTYPE'] = ( 'EXTRASOLAR'         ,'Source type')

            # The PM-RA stuff isn't implemented yet in Community Code
            try:
                banzai_image_header['PM-RA']   = (            eva_image_header['PM-RA'] ,'[sec/year] Proper motion in RA of the object')
                banzai_image_header['PM-DEC']  = (            eva_image_header['PM-DEC'] ,'[arcsec/year] Proper motion in Dec of the object')
                banzai_image_header['PARALLAX']= (            eva_image_header['PARALLAX'] ,'[arcsec] Parallax of the object')
                banzai_image_header['RADVEL']  = (            eva_image_header['RDVEL'] ,'[km/s] Radial velocity of the object')
                banzai_image_header['RATRACK'] = (            eva_image_header['RATRACK'] ,'[arcsec/s] Non-sidereal tracking in RA')
                banzai_image_header['DECTRACK']= (            eva_image_header['DECTRACK'] ,'[arcsec/s] Non-sidereal tracking in Dec')
                banzai_image_header['TRACFRAC']= (            eva_image_header['TRACFRAC'] ,'Fraction of ephemeris rate applied')
            except:
                banzai_image_header['PM-RA']   = (            0.0000000 ,'[sec/year] Proper motion in RA of the object')
                banzai_image_header['PM-DEC']  = (            0.0000000 ,'[arcsec/year] Proper motion in Dec of the object')
                banzai_image_header['PARALLAX']= (            0.0000000 ,'[arcsec] Parallax of the object')
                banzai_image_header['RADVEL']  = (            0.0000000 ,'[km/s] Radial velocity of the object')
                banzai_image_header['RATRACK'] = (            0.0000000 ,'[arcsec/s] Non-sidereal tracking in RA')
                banzai_image_header['DECTRACK']= (            0.0000000 ,'[arcsec/s] Non-sidereal tracking in Dec')
                banzai_image_header['TRACFRAC']= (            1.0000000 ,'Fraction of ephemeris rate applied')

            try:
                banzai_image_header['TELSTATE']= ( eva_image_header['TELSTATE']           ,'Current telescope status')
                banzai_image_header['ENGSTATE']= ( eva_image_header['ENGSTATE']           ,'Engineering override state')
                banzai_image_header['TCSSTATE']= ( eva_image_header['TCSSTATE']          ,'TCS state')
                banzai_image_header['TCSVER']  = ( eva_image_header['TCSVER']          ,'Version number of the TCS software')
            except:
                banzai_image_header['TELSTATE']= ( 'OKAY    '           ,'Current telescope status')
                banzai_image_header['ENGSTATE']= ( 'UNKNOWN '           ,'Engineering override state')
                banzai_image_header['TCSSTATE']= ( 'OKAY    '           ,'TCS state')
                banzai_image_header['TCSVER']  = ( 'COMMUNITY    '           ,'Version number of the TCS software')

            # try:
            #     banzai_image_header['TPNTMODL']= ( eva_image_header['TPNTMODL']     ,'Version number of the pointing model')
            #     banzai_image_header['UT1-UTC'] = ( eva_image_header['UT1-UTC'] ,'[s] UT1-UTC')
            #     banzai_image_header['POLARMOX']= ( eva_image_header['POLARMOX'] ,'[arcsec] Polar motion X')
            #     banzai_image_header['POLARMOY']= ( eva_image_header['POLARMOY'] ,'[arcsec] Polar motion Y')
            #     banzai_image_header['EOPSRC']  = ( eva_image_header['EOPSRC'] ,'Source of the EOP Values')

            #     banzai_image_header['ROLLERDR']= ( eva_image_header['ROLLERDR'],'[rad] Driven roller encoder angle')
            #     banzai_image_header['ROLLERND']= ( eva_image_header['ROLLERND'] ,'[rad] Non-driven roller encoder angle')
            #     banzai_image_header['AZDMD']   = ( eva_image_header['AZDMD'] ,'[deg] Azimuth axis demand')
            #     banzai_image_header['AZIMUTH'] = ( eva_image_header['AZIMUTH'] ,'[deg] Azimuth axis position')
            #     banzai_image_header['AZSTAT']  = ( eva_image_header['AZSTAT']           ,'Azimuth axis state')
            #     banzai_image_header['ALTDMD']  = ( eva_image_header['ALTDMD'] ,'[deg] Altitude axis demand')
            #     banzai_image_header['ALTITUDE']= ( eva_image_header['ALTITUDE'] ,'[deg] Altitude axis position')
            #     banzai_image_header['ALTSTAT'] = ( eva_image_header['ALTSTAT']           ,'Altitude axis state')
            #     banzai_image_header['ROTTYPE'] = ( eva_image_header['ROTTYPE']          ,'Selected image derotator')
            #     banzai_image_header['ROTMODE'] = ( eva_image_header['ROTMODE']           ,'Rotator mode')
            #     banzai_image_header['ROTDMD']  = ( eva_image_header['ROTDMD'] ,'[deg] Rotator axis demand')
            #     banzai_image_header['ROTANGLE']= ( eva_image_header['ROTANGLE']           ,'[deg] Rotator axis position')
            #     banzai_image_header['ROTSKYPA']= ( eva_image_header['ROTSKYPA'] ,'[deg] Rotator position angle')
            #     banzai_image_header['ROTSTAT'] = ( eva_image_header['ROTSTAT'] ,'Rotator axis state')
            #     banzai_image_header['AIRMASS'] = ( eva_image_header['AIRMASS'],'Effective mean airmass')
            #     banzai_image_header['AMSTART'] = ( eva_image_header['AMSTART'] ,'Airmass at start of observation')
            #     banzai_image_header['AMEND']   = ( eva_image_header['AMEND'] ,'Airmass at end of observation')
            #     banzai_image_header['ENC1STAT']= ( eva_image_header['ENC1STAT']          ,'Enclosure shutter 1 state')
            #     banzai_image_header['ENC2STAT']= ( eva_image_header['ENC2STAT']           ,'Enclosure shutter 2 state')
            #     banzai_image_header['ENCAZ']   = ( eva_image_header['ENCAZ'] ,'[deg] Enclosure azimuth')
            #     banzai_image_header['ENCWLIGT']= ( eva_image_header['ENCWLIGT']           ,'Enclosure white lights state')
            #     banzai_image_header['ENCRLIGT']= ( eva_image_header['ENCRLIGT']           ,'Enclosure red lights state')
            #     banzai_image_header['FSTATION']= ( eva_image_header['FSTATION'] ,'Focal station, e.g. Cassegrain, Nasmyth WS')
            #     banzai_image_header['FOLDSTAT']= ( eva_image_header['FOLDSTAT']          ,'Fold mirror state')
            #     banzai_image_header['FOLDPORT']= ( eva_image_header['FOLDPORT']           ,'Fold mirror port')
            #     banzai_image_header['FOLDPOSN']= ( eva_image_header['FOLDPOSN']          ,'[{mm,deg}] Fold mirror position (r, theta)')
            #     banzai_image_header['M1COVER'] = ( eva_image_header['M1COVER']           ,'M1 mirror cover state')
            #     banzai_image_header['M1HRTMN'] = ( eva_image_header['M1HRTMN']          ,'M1 Hartmann screen state')
            #     banzai_image_header['FOCDMD']  = ( eva_image_header['FOCDMD'] ,'[mm] Demanded focus position in focal plane')
            #     banzai_image_header['FOCPOSN'] = ( eva_image_header['FOCPOSN'] ,'[mm] Actual focus position in focal plane')
            #     banzai_image_header['FOCTELZP']= ( eva_image_header['FOCTELZP'] ,'[mm] Telescope default focus')
            #     banzai_image_header['FOCINOFF']= ( eva_image_header['FOCINOFF'],'[mm] Instrument focus offset')
            #     banzai_image_header['FOCTOFF'] = ( eva_image_header['FOCTOFF'] ,'[mm] Thermal correction value')
            #     banzai_image_header['FOCZOFF'] = ( eva_image_header['FOCZOFF'] ,'[mm] Zenith compression correction')
            #     banzai_image_header['FOCAFOFF']= ( eva_image_header['FOCAFOFF'] ,'[mm] Autofocus offset in focal plane')
            #     banzai_image_header['FOCOBOFF']= ( eva_image_header['FOCOBOFF'],'[mm] Observer focus offset/defocus in focal plane')
            #     banzai_image_header['FOCFLOFF']= ( eva_image_header['FOCFLOFF'] ,'[mm] Filter focus offset in focal plane')
            #     banzai_image_header['FOCSTAT'] = ( eva_image_header['FOCSTAT']           ,'Focus state')
            #     banzai_image_header['M2PITCH'] = ( eva_image_header['M2PITCH'] ,'[arcsec] M2 tilt about vertex in pitch direction')
            #     banzai_image_header['M2ROLL']  = ( eva_image_header['M2ROLL'] ,'[arcsec] M2 tilt about vertex in roll direction')
            #     banzai_image_header['AUXROLL'] = ( eva_image_header['AUXROLL'] ,'[arcsec] Auxiliary pointing corrections in roll')
            #     banzai_image_header['AUXPITCH']= ( eva_image_header['AUXPITCH'] ,'[arcsec] Auxiliary pointing corrections in pitch')
            # except:
            #     banzai_image_header['TPNTMODL']= ( 'N.Y.I.'     ,'Version number of the pointing model')
            #     banzai_image_header['UT1-UTC'] = (            0.0567600 ,'[s] UT1-UTC')
            #     banzai_image_header['POLARMOX']= (            0.0 ,'[arcsec] Polar motion X')
            #     banzai_image_header['POLARMOY']= (            0.0,'[arcsec] Polar motion Y')
            #     banzai_image_header['EOPSRC']  = ( 'IERS BULL. A 2024/09/19' ,'Source of the EOP Values')
            #     banzai_image_header['ROLLERDR']= (            0.0000000 ,'[rad] Driven roller encoder angle')
            #     banzai_image_header['ROLLERND']= (            0.0000000 ,'[rad] Non-driven roller encoder angle')
            #     banzai_image_header['AZDMD']   = (          eva_image_header['AZIMUTH'] ,'[deg] Azimuth axis demand')
            #     banzai_image_header['AZIMUTH'] = (          eva_image_header['AZIMUTH'] ,'[deg] Azimuth axis position')
            #     banzai_image_header['AZSTAT']  = ( 'OKAY    '           ,'Azimuth axis state')
            #     banzai_image_header['ALTDMD']  = (           eva_image_header['ALTITUDE'] ,'[deg] Altitude axis demand')
            #     banzai_image_header['ALTITUDE']= (           eva_image_header['ALTITUDE'] ,'[deg] Altitude axis position')
            #     banzai_image_header['ALTSTAT'] = ( 'OKAY    '           ,'Altitude axis state')
            #     banzai_image_header['ROTTYPE'] = ( 'NONE    '           ,'Selected image derotator')
            #     banzai_image_header['ROTMODE'] = ( 'FIXED   '           ,'Rotator mode')
            #     banzai_image_header['ROTDMD']  = (            0.0000000 ,'[deg] Rotator axis demand')
            #     banzai_image_header['ROTANGLE']= ( 'N/A     '           ,'[deg] Rotator axis position')
            #     banzai_image_header['ROTSKYPA']= (           0.0 ,'[deg] Rotator position angle')
            #     banzai_image_header['ROTSTAT'] = ('OFF     '           ,'Rotator axis state')
            #     banzai_image_header['AIRMASS'] = (           eva_image_header['AIRMASS'] ,'Effective mean airmass')
            #     banzai_image_header['AMSTART'] = (           eva_image_header['AIRMASS'] ,'Airmass at start of observation')
            #     banzai_image_header['AMEND']   = (           eva_image_header['AIRMASS'] ,'Airmass at end of observation')
            #     banzai_image_header['ENC1STAT']= ( 'OPEN    '           ,'Enclosure shutter 1 state')
            #     banzai_image_header['ENC2STAT']= ( 'OPEN    '           ,'Enclosure shutter 2 state')
            #     banzai_image_header['ENCAZ']   = (            0.0000000 ,'[deg] Enclosure azimuth')
            #     banzai_image_header['ENCWLIGT']= ( 'OFF     '           ,'Enclosure white lights state')
            #     banzai_image_header['ENCRLIGT']= ( 'OFF     '           ,'Enclosure red lights state')
            #     banzai_image_header['FSTATION']= ( 'UNKNOWN' ,'Focal station, e.g. Cassegrain, Nasmyth WS')
            #     banzai_image_header['FOLDSTAT']= ( 'N/A     '           ,'Fold mirror state')
            #     banzai_image_header['FOLDPORT']= ( '0       '           ,'Fold mirror port')
            #     banzai_image_header['FOLDPOSN']= ( 'N/A, N/A'           ,'[{mm,deg}] Fold mirror position (r, theta)')
            #     banzai_image_header['M1COVER'] = ( 'STOWED  '           ,'M1 mirror cover state')
            #     banzai_image_header['M1HRTMN'] = ( 'UNKNOWN '           ,'M1 Hartmann screen state')
            #     banzai_image_header['FOCDMD']  = (            0.0000000 ,'[arbitrary] Demanded focus position in focal plane')
            #     banzai_image_header['FOCPOSN'] = (            eva_image_header['FOCUSPOS'] ,'[arbitrary] Actual focus position in focal plane')
            #     banzai_image_header['FOCTELZP']= (            eva_image_header['FOCUSPOS'] ,'[arbitrary] Telescope default focus')
            #     banzai_image_header['FOCINOFF']= (            0.0000000 ,'[arbitrary] Instrument focus offset')
            #     banzai_image_header['FOCTOFF'] = (            0.00 ,'[arbitrary] Thermal correction value')
            #     banzai_image_header['FOCZOFF'] = (            0.00 ,'[arbitrary] Zenith compression correction')
            #     banzai_image_header['FOCAFOFF']= (            0.00 ,'[arbitrary] Autofocus offset in focal plane')
            #     banzai_image_header['FOCOBOFF']= (            0.0000000 ,'[arbitrary] Observer focus offset/defocus in focal plane')
            #     banzai_image_header['FOCFLOFF']= (            0.0 ,'[arbitrary] Filter focus offset in focal plane')
            #     banzai_image_header['FOCSTAT'] = ( 'HALTED  '           ,'Focus state')
            #     banzai_image_header['M2PITCH'] = (            0.0000000 ,'[arcsec] M2 tilt about vertex in pitch direction')
            #     banzai_image_header['M2ROLL']  = (            0.0000000 ,'[arcsec] M2 tilt about vertex in roll direction')
            #     banzai_image_header['AUXROLL'] = (            0.0000000 ,'[arcsec] Auxiliary pointing corrections in roll')
            #     banzai_image_header['AUXPITCH']= (            0.0000000 ,'[arcsec] Auxiliary pointing corrections in pitch')


            header_defs = {
                'TPNTMODL': ('TPNTMODL', eva_image_header.get('TPNTMODL', 'N.Y.I.'),             'Version number of the pointing model'),
                'UT1-UTC':  ('UT1-UTC',  eva_image_header.get('UT1-UTC',  0.0567600),           '[s] UT1-UTC'),
                'POLARMOX': ('POLARMOX', eva_image_header.get('POLARMOX', 0.0),                 '[arcsec] Polar motion X'),
                'POLARMOY': ('POLARMOY', eva_image_header.get('POLARMOY', 0.0),                 '[arcsec] Polar motion Y'),
                'EOPSRC':   ('EOPSRC',   eva_image_header.get('EOPSRC',   'IERS BULL. A 2024/09/19'), 'Source of the EOP Values'),
            
                'ROLLERDR': ('ROLLERDR', eva_image_header.get('ROLLERDR', 0.0),                 '[rad] Driven roller encoder angle'),
                'ROLLERND': ('ROLLERND', eva_image_header.get('ROLLERND', 0.0),                 '[rad] Non-driven roller encoder angle'),
            
                'AZDMD':    ('AZIMUTH',  eva_image_header.get('AZIMUTH', 0.0),                 '[deg] Azimuth axis demand'),
                'AZIMUTH':  ('AZIMUTH',  eva_image_header.get('AZIMUTH', 0.0),                 '[deg] Azimuth axis position'),
                'AZSTAT':   ('AZSTAT',   eva_image_header.get('AZSTAT',   'OKAY    '),         'Azimuth axis state'),
            
                'ALTDMD':   ('ALTITUDE', eva_image_header.get('ALTITUDE', 0.0),                '[deg] Altitude axis demand'),
                'ALTITUDE': ('ALTITUDE', eva_image_header.get('ALTITUDE', 0.0),                '[deg] Altitude axis position'),
                'ALTSTAT':  ('ALTSTAT',  eva_image_header.get('ALTSTAT',  'OKAY    '),         'Altitude axis state'),
            
                'ROTTYPE':  ('ROTTYPE',  eva_image_header.get('ROTTYPE',  'NONE    '),         'Selected image derotator'),
                'ROTMODE':  ('ROTMODE',  eva_image_header.get('ROTMODE',  'FIXED   '),         'Rotator mode'),
                'ROTDMD':   ('ROTDMD',   eva_image_header.get('ROTDMD',   0.0),                '[deg] Rotator axis demand'),
                'ROTANGLE': ('ROTANGLE', eva_image_header.get('ROTANGLE', 'N/A     '),         '[deg] Rotator axis position'),
                'ROTSKYPA': ('ROTSKYPA', eva_image_header.get('ROTSKYPA', 0.0),                '[deg] Rotator position angle'),
                'ROTSTAT':  ('ROTSTAT',  eva_image_header.get('ROTSTAT',  'OFF     '),         'Rotator axis state'),
            
                'AIRMASS':  ('AIRMASS',  eva_image_header.get('AIRMASS',  0.0),                'Effective mean airmass'),
                'AMSTART':  ('AIRMASS',  eva_image_header.get('AIRMASS',  0.0),                'Airmass at start of observation'),
                'AMEND':    ('AIRMASS',  eva_image_header.get('AIRMASS',  0.0),                'Airmass at end of observation'),
            
                'ENC1STAT': ('ENC1STAT', eva_image_header.get('ENC1STAT', 'OPEN    '),         'Enclosure shutter 1 state'),
                'ENC2STAT': ('ENC2STAT', eva_image_header.get('ENC2STAT', 'OPEN    '),         'Enclosure shutter 2 state'),
                'ENCAZ':    ('ENCAZ',    eva_image_header.get('ENCAZ',    0.0),                '[deg] Enclosure azimuth'),
                'ENCWLIGT': ('ENCWLIGT', eva_image_header.get('ENCWLIGT', 'OFF     '),         'Enclosure white lights state'),
                'ENCRLIGT': ('ENCRLIGT', eva_image_header.get('ENCRLIGT', 'OFF     '),         'Enclosure red lights state'),
            
                'FSTATION': ('FSTATION', eva_image_header.get('FSTATION', 'UNKNOWN'),         'Focal station, e.g. Cassegrain, Nasmyth WS'),
                'FOLDSTAT': ('FOLDSTAT', eva_image_header.get('FOLDSTAT', 'N/A     '),         'Fold mirror state'),
                'FOLDPORT': ('FOLDPORT', eva_image_header.get('FOLDPORT', '0       '),         'Fold mirror port'),
                'FOLDPOSN': ('FOLDPOSN', eva_image_header.get('FOLDPOSN', 'N/A, N/A'),         '[{mm,deg}] Fold mirror position (r, theta)'),
            
                'M1COVER':  ('M1COVER',  eva_image_header.get('M1COVER',  'STOWED  '),         'M1 mirror cover state'),
                'M1HRTMN':  ('M1HRTMN',  eva_image_header.get('M1HRTMN',  'UNKNOWN '),         'M1 Hartmann screen state'),
            
                'FOCDMD':   ('FOCDMD',   eva_image_header.get('FOCDMD',   0.0),                '[arbitrary] Demanded focus position in focal plane'),
                'FOCPOSN':  ('FOCUSPOS', eva_image_header.get('FOCUSPOS', 0.0),                '[arbitrary] Actual focus position in focal plane'),
                'FOCTELZP': ('FOCTELZP', eva_image_header.get('FOCTELZP', eva_image_header.get('FOCUSPOS', 0.0)), '[arbitrary] Telescope default focus'),
                'FOCINOFF': ('FOCINOFF', eva_image_header.get('FOCINOFF', 0.0),               '[arbitrary] Instrument focus offset'),
                'FOCTOFF':  ('FOCTOFF',  eva_image_header.get('FOCTOFF',  0.00),              '[arbitrary] Thermal correction value'),
                'FOCZOFF':  ('FOCZOFF',  eva_image_header.get('FOCZOFF',  0.00),              '[arbitrary] Zenith compression correction'),
                'FOCAFOFF': ('FOCAFOFF', eva_image_header.get('FOCAFOFF', 0.00),              '[arbitrary] Autofocus offset in focal plane'),
                'FOCOBOFF': ('FOCOBOFF', eva_image_header.get('FOCOBOFF', 0.0),               '[arbitrary] Observer focus offset/defocus in focal plane'),
                'FOCFLOFF': ('FOCFLOFF', eva_image_header.get('FOCFLOFF', 0.0),               '[arbitrary] Filter focus offset in focal plane'),
                'FOCSTAT':  ('FOCSTAT',  eva_image_header.get('FOCSTAT',  'HALTED  '),        'Focus state'),
            
                'M2PITCH':  ('M2PITCH',  eva_image_header.get('M2PITCH', 0.0),               '[arcsec] M2 tilt about vertex in pitch direction'),
                'M2ROLL':   ('M2ROLL',   eva_image_header.get('M2ROLL',  0.0),               '[arcsec] M2 tilt about vertex in roll direction'),
            
                'AUXROLL':  ('AUXROLL',  eva_image_header.get('AUXROLL', 0.0),               '[arcsec] Auxiliary pointing corrections in roll'),
                'AUXPITCH': ('AUXPITCH', eva_image_header.get('AUXPITCH',0.0),               '[arcsec] Auxiliary pointing corrections in pitch'),
            }
            
            # then your loop:
            for hdr_key, (eva_key, default, comment) in header_defs.items():
                value = eva_image_header.get(eva_key, default)
                banzai_image_header[hdr_key] = (value, comment)

            ########### WCS COPY HERE
            try:
                banzai_image_header['WCSAXES']  = ( eva_image_header['WCSAXES']           ,'Number of WCSAXES')
            except:
                pass

            try:
                banzai_image_header['CTYPE1']  = ( eva_image_header['CTYPE1']           ,'Type of WCS Projection')
                banzai_image_header['CTYPE2']  = ( eva_image_header['CTYPE2']         ,'Type of WCS Projection')
                banzai_image_header['CRPIX1']  = ( eva_image_header['CRPIX1'] ,'[pixel] Coordinate of reference point (axis 1)')
                banzai_image_header['CRPIX2']  = ( eva_image_header['CRPIX2'] ,'[pixel] Coordinate of reference point (axis 2)')
                banzai_image_header['CRVAL1']  = ( eva_image_header['CRVAL1'] ,'[deg] RA at the reference pixel')
                banzai_image_header['CRVAL2']  = ( eva_image_header['CRVAL2'] ,'[deg] Dec at the reference pixel')
                banzai_image_header['CUNIT1']  = ( eva_image_header['CUNIT1']           ,'Units of RA')
                banzai_image_header['CUNIT2']  = ( eva_image_header['CUNIT2']          ,'Units of Dec')

                ####### The PTR WCS can be significantly different than the LCO one,
                ####### and can have various different WCS types and polynomial scales, so we need to check what is there and plonk it in
                potential_wcs_keywords=['WCSAXES','EQUINOX','LONPOLE','LATPOLE','MJDREF','CROTA1','CROTA2','CD1_1','CD1_2','CD2_1','CD2_2','IMAGEW','IMAGEH','A_ORDER','A_0_0','A_0_1','A_0_2','A_1_0','A_1_1','A_2_0','B_ORDER','B_0_0','B_0_1','B_0_2','B_1_0','B_1_1','B_2_0','AP_ORDER','AP_0_0','AP_0_1','AP_0_2','AP_1_0','AP_1_1','AP_2_0','BP_ORDER','BP_0_0','BP_0_1','BP_0_2','BP_1_0','BP_1_1','BP_2_0','A_0_3','A_0_4','A_1_2','A_1_3','A_2_1','A_2_2','A_3_0','A_3_1','A_4_0','B_0_3','B_0_4','B_1_2','B_1_3','B_2_1','B_2_2','B_3_0','B_3_1','B_4_0','AP_0_3','AP_0_4','AP_1_2','AP_1_3','AP_2_1','AP_2_2','AP_3_0','AP_3_1','AP_4_0','BP_0_3','BP_0_4','BP_1_2','BP_1_3','BP_2_1','BP_2_2','BP_3_0','BP_3_1','BP_4_0','PC1_1', 'PC1_2','PC2_1','PC2_2','CDELT1','CDELT2','PV1_1','PV1_2','PV1_3','PV1_4','PV1_5','PV1_6','PV1_7','PV1_8','PV1_9','PV2_0','PV2_1','PV2_2','PV2_3','PV2_4','PV2_5','PV2_6','PV2_7','PV2_8','PV2_9']

                for temp_entry in potential_wcs_keywords:
                    try:
                        banzai_image_header[temp_entry]   = (  eva_image_header[temp_entry]  ,'WCS CD transformation matrix')
                    except:
                        pass

            except:
                pass



            try:
                banzai_image_header['WMSSTATE']= ( eva_image_header['WMSSTATE']          ,'WMS system state')
                banzai_image_header['WMSHUMID']= ( eva_image_header['WMSHUMID'] ,'[%] Current percentage humidity')
                banzai_image_header['WMSTEMP'] = ( eva_image_header['WMSTEMP'] ,'[deg C] External temperature')
                banzai_image_header['WMSPRES'] = ( eva_image_header['WMSPRES'] ,'[mbar] Atmospheric pressure')
                banzai_image_header['WINDSPEE']= ( eva_image_header['WINDSPEE'] ,'[km/h] Windspeed')
                banzai_image_header['WINDDIR'] = ( eva_image_header['WINDDIR'] ,'[deg E of N] Wind direction')
                banzai_image_header['WMSRAIN'] = ( eva_image_header['WMSRAIN']           ,'Rain alert')
                banzai_image_header['WMSMOIST']= ( eva_image_header['WMSMOIST'],'[mV] Moisture level')
                banzai_image_header['WMSDEWPT']= ( eva_image_header['WMSDEWPT'],'[deg C] Dewpoint')
                banzai_image_header['WMSCLOUD']= ( eva_image_header['WMSCLOUD'] ,'[deg C] Boltwood sky temperature')
                banzai_image_header['WMSSKYBR']= ( eva_image_header['WMSSKYBR'] ,'[mag/arcsec^2] Measured sky brightness')
                banzai_image_header['SKYMAG']  = ( eva_image_header['SKYMAG'],'[mag/arcsec^2] Computed (expected) sky brightness')
                banzai_image_header['TUBETEMP']= ( eva_image_header['TUBETEMP'],'[deg C] Temperature of the telescope tube')
                banzai_image_header['M1TEMP']  = ( eva_image_header['M1TEMP'],'[deg C] Primary mirror temperature')
                try:
                    banzai_image_header['FOCTEMP'] = ( eva_image_header['FOCTEMP'],'[deg C] Focus temperature')
                except:
                    banzai_image_header['FOCTEMP'] = ( eva_image_header['FOCUSTMP'],'[deg C] Focus temperature')
                banzai_image_header['ISSTEMP'] = ( eva_image_header['ISSTEMP']           ,'[deg C] ISS temperature')
                banzai_image_header['REFPRES'] = ( eva_image_header['REFPRES'] ,'[mbar] Pressure used in refraction calculation')
                banzai_image_header['REFTEMP'] = ( eva_image_header['REFTEMP'] ,'[deg C] Temperature used in refraction calculat')
                banzai_image_header['REFHUMID']= ( eva_image_header['REFHUMID'] ,'[%] Humidity used in refraction calculation')
                banzai_image_header['AGSTATE'] = ( eva_image_header['AGSTATE'] ,'Autoguider software state')
                banzai_image_header['AGCAM']   = ( eva_image_header['AGCAM']           ,'Camera used for autoguiding')
                banzai_image_header['AGLCKFRC']= ( eva_image_header['AGLCKFRC'] ,'[%] Fraction of time AG locked')
                banzai_image_header['AGMODE']  = ( eva_image_header['AGMODE']           ,'Autoguider mode')
                banzai_image_header['AGRA']    = ( eva_image_header['AGRA'] ,'[deg] RA of guide star')
                banzai_image_header['AGDEC']   = ( eva_image_header['AGDEC'] ,'[deg] Dec of guide star')
                banzai_image_header['AGGMAG']  = ( eva_image_header['AGGMAG']          ,'[mag] Autoguider guide star mag')
                banzai_image_header['AGFWHM']  = ( eva_image_header['AGFHWM'],'[arcsec] Autoguider FWHM')
                banzai_image_header['AGNSRC']  = ( eva_image_header['AGNSRC'] ,'Number of potential guide stars')
                banzai_image_header['AGDX']    = ( eva_image_header['AGDX'] ,'[pixel] Autoguider correction in X')
                banzai_image_header['AGDY']    = ( eva_image_header['AGDY'] ,'[pixel] Autoguider correction in Y')
                banzai_image_header['AGMIRDMD']= ( eva_image_header['AGMIRDMD']          ,'[mm] Autoguider mirror demand')
                banzai_image_header['AGMIRPOS']= ( eva_image_header['AGMIRPOS']           ,'Autoguider mirror position')
                banzai_image_header['AGMIRST'] = ( eva_image_header['AGMIRST']           ,'Autoguider mirror state')
                banzai_image_header['AGFOCDMD']= ( eva_image_header['AGFOCDMD']         ,'[mm] Autoguider focus demand')
                banzai_image_header['AGFOCUS'] = ( eva_image_header['AGFOCUS']           ,'[mm] Autoguider focus position')
                banzai_image_header['AGFOCOFF']= ( eva_image_header['AGFOCOFF'] ,'[mm] Autoguider relative focus offset')
                banzai_image_header['AGFOCST'] = ( eva_image_header['AGFOCST']           ,'Autoguider focus state')
                banzai_image_header['AGFILTER']= ( eva_image_header['AGFILTER']       ,'Autoguider filter')
                banzai_image_header['AGFILTID']= ( eva_image_header['AGFILTID']      ,'Autoguider filter id')
                banzai_image_header['AGFILST'] = ( eva_image_header['AGFILST']          ,'Autoguider filter state')
                banzai_image_header['MOONSTAT']= ( eva_image_header['MOONSTAT']          ,'[{UP, DOWN}] Moon position at obs start')
                banzai_image_header['MOONFRAC']= ( eva_image_header['MOONFRAC'] ,'[(0 - 1)] Lunar Illuminated Fraction')
                banzai_image_header['MOONDIST']= ( eva_image_header['MOONDIST'] ,'[deg] Lunar distance from target')
                banzai_image_header['MOONALT'] = ( eva_image_header['MOONALT'] ,'[deg] Lunar altitude')
                banzai_image_header['SUNDIST'] = ( eva_image_header['SUNDIST'] ,'[deg] Solar distance from target')
                banzai_image_header['SUNALT']  = ( eva_image_header['SUNALT'] ,'[deg] Solar altitude')
                banzai_image_header['EXTNAME'] = ( 'SCI     '           ,'Extension name')
                banzai_image_header['EXTVER']  = (                    1 ,'Extension instance')
                #banzai_image_header['CHECKSUM']= ( eva_image_header['CUNIT2']  ,'HDU checksum updated 2024-09-23T22:02:38')
                #banzai_image_header['DATASUM'] = ( eva_image_header['CUNIT2']        ,'data unit checksum updated 2024-09-23T22:02:38')
                banzai_image_header['L1IDMASK']= ( eva_image_header['L1IDMASK'] ,'Id. of mask file u')
                banzai_image_header['L1IDRDN'] = ( eva_image_header['L1IDRDN'] ,'Id. of readn')
                banzai_image_header['SATFRAC'] = ( eva_image_header['SATFRAC'],'Fraction of Pixels that are Saturated')
                banzai_image_header['L1STATOV']= ( eva_image_header['L1STATOV']          ,'Status flag for overscan correction')
                banzai_image_header['OVERSCAN']= ( eva_image_header['OVERSCAN'] )
                banzai_image_header['OVERSCN1']= ( eva_image_header['OVERSCN1']           ,'Overscan value that was subtracted')
                banzai_image_header['BIASLVL'] = ( eva_image_header['BIASLVL'],'Bias level that was removed after overscan')
                banzai_image_header['L1IDBIAS']= ( eva_image_header['L1IDBIAS'],'ID of bias')
                banzai_image_header['L1STATBI']= ( eva_image_header['L1STATBI'] ,'Status flag for bias frame correction')
                banzai_image_header['L1IDDARK']= ( eva_image_header['L1IDDARK'] ,'ID of dark')
                banzai_image_header['L1STATDA']= ( eva_image_header['L1STATDA'],'Status flag for dark frame correction')
                banzai_image_header['DRKTSCAL']= ( eva_image_header['DRKTSCAL'] ,'Temperature scaling factor applied to dark imag')
                banzai_image_header['L1IDFLAT']= ( eva_image_header['L1IDFLAT'] ,'ID of flat frame')
                banzai_image_header['L1STATFL']= ( eva_image_header['L1STATFL'] ,'Status flag for flat field correction')
                banzai_image_header['L1MEAN']=   ( eva_image_header['L1MEAN'],'[counts] Sigma clipped mean of frame background')
                banzai_image_header['L1MEDIAN']= ( eva_image_header['L1MEDIAN'] ,'[counts] Median of frame background')
                banzai_image_header['L1SIGMA'] = ( eva_image_header['L1SIGMA'] ,'[counts] Robust std dev of frame background')
                banzai_image_header['L1FWHM']  = ( eva_image_header['L1FWHM'] ,'[arcsec] Frame FWHM in arcsec')
                banzai_image_header['L1FWTM']  = ( eva_image_header['L1FWTM'] ,'Ratio of FWHM to Full-Width Tenth Max')
                banzai_image_header['L1ELLIP'] = ( eva_image_header['L1ELLIP'],'Mean image ellipticity (1-B/A)')
                banzai_image_header['L1ELLIPA']= ( eva_image_header['L1ELLIPA'] ,'[deg] PA of mean image ellipticity')

                if 'CRVAL1' in eva_image_header.keys():
                    banzai_image_header['WCSERR']  = ( 0 ,'Error status of WCS fit. 0 for no error')
                else:
                    banzai_image_header['WCSERR']  = ( 1 ,'Error status of WCS fit. 0 for no error')
                banzai_image_header['PNTOFST'] = ( eva_image_header['PNTOFST'] ,'[arcsec] offset of requested and solved center')
                if 'EVA-' in file:
                    banzai_image_header['RLEVEL']  = ( 92 ,'Reduction level')
                else:
                    banzai_image_header['RLEVEL']  = ( 93 ,'Reduction level')
                banzai_image_header['PIPEVER'] = ( 'EVA  '           ,'Pipeline version')
                banzai_image_header['L1PUBDAT']= ( eva_image_header['L1PUBDAT'] ,'[UTC] Date the frame becomes public')
                #banzai_image_header['ZHECKSUM']= ( eva_image_header['CUNIT2']  ,'HDU checksum updated 2024-09-23T22:02:38')
                #banzai_image_header['ZDATASUM']= (eva_image_header['CUNIT2']        ,'data unit checksum updated 2024-09-23T22:02:38')
            except:
                banzai_image_header['WMSSTATE']= ( 'OKAY    '           ,'WMS system state')
                banzai_image_header['WMSHUMID']= (           'UNKNOWN '  ,'[%] Current percentage humidity')
                banzai_image_header['WMSTEMP'] = (           'UNKNOWN '  ,'[deg C] External temperature')
                banzai_image_header['WMSPRES'] = (          'UNKNOWN '  ,'[mbar] Atmospheric pressure')
                banzai_image_header['WINDSPEE']= (           'UNKNOWN ' ,'[km/h] Windspeed')
                banzai_image_header['WINDDIR'] = (        'UNKNOWN '  ,'[deg E of N] Wind direction')
                banzai_image_header['WMSRAIN'] = ( 'CLEAR   '           ,'Rain alert')
                banzai_image_header['WMSMOIST']= (         'UNKNOWN ' ,'[mV] Moisture level')
                banzai_image_header['WMSDEWPT']= (         'UNKNOWN '  ,'[deg C] Dewpoint')
                banzai_image_header['WMSCLOUD']= (          'UNKNOWN '  ,'[deg C] Boltwood sky temperature')
                banzai_image_header['WMSSKYBR']= (           'UNKNOWN '  ,'[mag/arcsec^2] Measured sky brightness')
                banzai_image_header['SKYMAG']  = (           'UNKNOWN '  ,'[mag/arcsec^2] Computed (expected) sky brightne')
                banzai_image_header['TUBETEMP']= (           'UNKNOWN '  ,'[deg C] Temperature of the telescope tube')
                banzai_image_header['M1TEMP']  = ( 'UNKNOWN '           ,'[deg C] Primary mirror temperature')
                try:
                    banzai_image_header['FOCTEMP'] = ( eva_image_header['FOCTEMP'],'[deg C] Focus temperature')
                except:
                    banzai_image_header['FOCTEMP'] = ( eva_image_header['FOCUSTMP'],'[deg C] Focus temperature')
                banzai_image_header['ISSTEMP'] = ( 'UNKNOWN '           ,'[deg C] ISS temperature')
                banzai_image_header['REFPRES'] = (     'UNKNOWN '  ,'[mbar] Pressure used in refraction calculation')
                banzai_image_header['REFTEMP'] = (      'UNKNOWN '  ,'[deg C] Temperature used in refraction calculat')
                banzai_image_header['REFHUMID']= (       'UNKNOWN '  ,'[%] Humidity used in refraction calculation')
                banzai_image_header['AGSTATE'] = ( 'OFF' ,'Autoguider software state')
                banzai_image_header['AGCAM']   = ( 'No_autoguider    '           ,'Camera used for autoguiding')
                banzai_image_header['AGLCKFRC']= (               0 ,'[%] Fraction of time AG locked')
                banzai_image_header['AGMODE']  = ( 'OFF   '           ,'Autoguider mode')
                banzai_image_header['AGRA']    = (            0 ,'[deg] RA of guide star')
                banzai_image_header['AGDEC']   = (            0 ,'[deg] Dec of guide star')
                banzai_image_header['AGGMAG']  = ( 'UNKNOWN '           ,'[mag] Autoguider guide star mag')
                banzai_image_header['AGFWHM']  = (            0,'[arcsec] Autoguider FWHM')
                banzai_image_header['AGNSRC']  = (                    0 ,'Number of potential guide stars')
                banzai_image_header['AGDX']    = (            0.0000000 ,'[pixel] Autoguider correction in X')
                banzai_image_header['AGDY']    = (            0.0000000 ,'[pixel] Autoguider correction in Y')
                banzai_image_header['AGMIRDMD']= ( 'UNKNOWN '           ,'[mm] Autoguider mirror demand')
                banzai_image_header['AGMIRPOS']= ( 'N/A, N/A'           ,'Autoguider mirror position')
                banzai_image_header['AGMIRST'] = ( 'N/A     '           ,'Autoguider mirror state')
                banzai_image_header['AGFOCDMD']= ( 'UNKNOWN '           ,'[mm] Autoguider focus demand')
                banzai_image_header['AGFOCUS'] = ( 'UNKNOWN '           ,'[mm] Autoguider focus position')
                banzai_image_header['AGFOCOFF']= (            0.0000000 ,'[mm] Autoguider relative focus offset')
                banzai_image_header['AGFOCST'] = ( 'UNKNOWN '           ,'Autoguider focus state')
                banzai_image_header['AGFILTER']= ( 'air'       ,'Autoguider filter')
                banzai_image_header['AGFILTID']= ( 'None'        ,'Autoguider filter id')
                banzai_image_header['AGFILST'] = ( 'N/A '           ,'Autoguider filter state')
                try:
                    banzai_image_header['MOONSTAT']= ( eva_image_header['MOONSTAT']           ,'[{UP, DOWN}] Moon position at obs start')
                    banzai_image_header['MOONFRAC']= ( eva_image_header['MOONFRAC'] ,'[(0 - 1)] Lunar Illuminated Fraction')
                    banzai_image_header['MOONDIST']= ( eva_image_header['MOONDIST'] ,'[deg] Lunar distance from target')
                    banzai_image_header['MOONALT'] = ( eva_image_header['MOONALT'] ,'[deg] Lunar altitude')
                    banzai_image_header['SUNDIST'] = ( eva_image_header['SUNDIST'] ,'[deg] Solar distance from target')
                    banzai_image_header['SUNALT']  = ( eva_image_header['SUNALT'] ,'[deg] Solar altitude')
                except:
                    banzai_image_header['MOONSTAT']= ( 'UNKNOWN'           ,'[{UP, DOWN}] Moon position at obs start')
                    banzai_image_header['MOONFRAC']= ( 'UNKNOWN'  ,'[(0 - 1)] Lunar Illuminated Fraction')
                    banzai_image_header['MOONDIST']= ( 'UNKNOWN' ,'[deg] Lunar distance from target')
                    banzai_image_header['MOONALT'] = ( 'UNKNOWN'  ,'[deg] Lunar altitude')
                    banzai_image_header['SUNDIST'] = ( 'UNKNOWN' ,'[deg] Solar distance from target')
                    banzai_image_header['SUNALT']  = ( 'UNKNOWN'  ,'[deg] Solar altitude')

                banzai_image_header['EXTNAME'] = ( 'SCI     '           ,'Extension name')
                banzai_image_header['EXTVER']  = (                    1 ,'Extension instance')
                #banzai_image_header['CHECKSUM']= ( 'WAcaX7ZSWAbYW5ZY'   ,'HDU checksum updated 2024-09-23T22:02:38')
                #banzai_image_header['DATASUM'] = ( '3136908556'         ,'data unit checksum updated 2024-09-23T22:02:38')
                banzai_image_header['L1IDMASK']= ( 'EVA' ,'Id. of mask file u')
                banzai_image_header['L1IDRDN'] = ( 'EVA' ,'Id. of readnise')
                banzai_image_header['SATFRAC'] = (                  0.0 ,'Fraction of Pixels that are Saturated')
                banzai_image_header['L1STATOV']= ( '0       '           ,'Status flag for overscan correction')
                banzai_image_header['OVERSCAN']= (                  0.0 )
                banzai_image_header['OVERSCN1']= ( '0.00    '           ,'Overscan value that was subtracted')
                banzai_image_header['BIASLVL'] = (   0.0 ,'Bias level that was removed after overscan')
                banzai_image_header['L1IDBIAS']= ( 'EVA' ,'ID of bias')
                banzai_image_header['L1STATBI']= (                    1 ,'Status flag for bias frame correction')
                banzai_image_header['L1IDDARK']= ( 'EVA' ,'ID of dark')
                banzai_image_header['L1STATDA']= (                    1 ,'Status flag for dark frame correction')
                banzai_image_header['DRKTSCAL']= (                  1.0 ,'Temperature scaling factor applied to dark imag')
                banzai_image_header['L1IDFLAT']= ( 'EVA' ,'ID of flat frame')
                banzai_image_header['L1STATFL']= (                    1 ,'Status flag for flat field correction')
                try:
                    banzai_image_header['L1MEAN']=   ( eva_image_header['L1MEAN']  ,'[counts] Sigma clipped mean of frame background')
                    banzai_image_header['L1MEDIAN']= ( eva_image_header['L1MEDIAN']  ,'[counts] Median of frame background')
                    banzai_image_header['L1SIGMA'] = ( eva_image_header['L1SIGMA']  ,'[counts] Robust std dev of frame background')
                    banzai_image_header['L1FWHM']  = ( eva_image_header['FWHMasec'] ,'[arcsec] Frame FWHM in arcsec')
                    banzai_image_header['L1FWTM']  = ( eva_image_header['L1FWTM']  ,'Ratio of FWHM to Full-Width Tenth Max')
                banzai_image_header['L1ELLIP'] = ( eva_image_header['L1ELLIP']  ,'Mean image ellipticity (1-B/A)')
                banzai_image_header['L1ELLIPA']= ( eva_image_header['L1ELLIPA']  ,'[deg] PA of mean image ellipticity')
                banzai_image_header['IMGMIN']= ( eva_image_header['IMGMIN'] ,'Minimum Value of Image Array')
                banzai_image_header['IMGMAX']= ( eva_image_header['IMGMAX'] ,'Maximum Value of Image Array')
                banzai_image_header['IMGMEAN']= ( eva_image_header['IMGMEAN'] ,'Mean Value of Image Array')
                banzai_image_header['IMGMODE']= ( eva_image_header['IMGMODE'] ,'Mode Value of Image Array')
                banzai_image_header['IMGMED']= ( eva_image_header['IMGMED'] ,'Median Value of Image Array')
                banzai_image_header['IMGMAD']= ( eva_image_header['IMGMAD'] ,'Median Absolute Deviation of Image Array')
                banzai_image_header['IMGSTDEV']= ( eva_image_header['IMGSTDEV'] ,'Standard Deviation of Image Array')
                    if 'CRVAL1' in eva_image_header.keys():
                        banzai_image_header['WCSERR']  = ( 0 ,'Error status of WCS fit. 0 for no error')
                    else:
                        banzai_image_header['WCSERR']  = ( 1 ,'Error status of WCS fit. 0 for no error')
                    banzai_image_header['PNTOFST'] = (   eva_image_header['PNTOFST'] ,'[arcsec] offset of requested and solved center')
                except:
                    banzai_image_header['L1MEAN']=   ( 0.0 ,'[counts] Sigma clipped mean of frame background')
                    banzai_image_header['L1MEDIAN']= (    0.0 ,'[counts] Median of frame background')
                    banzai_image_header['L1SIGMA'] = (   0.0 ,'[counts] Robust std dev of frame background')
                    try:
                        banzai_image_header['L1FWHM']  = (   eva_image_header['FWHMasec'] ,'[arcsec] Frame FWHM in arcsec')
                    except:
                        banzai_image_header['L1FWHM']  = (   'nan' ,'[arcsec] Frame FWHM in arcsec')

                    banzai_image_header['L1FWTM']  = (   'UNKNOWN' ,'Ratio of FWHM to Full-Width Tenth Max')
                    banzai_image_header['L1ELLIP'] = (  0.0 ,'Mean image ellipticity (1-B/A)')
                    banzai_image_header['L1ELLIPA']= (   0.0 ,'[deg] PA of mean image ellipticity')
                    banzai_image_header['IMGMIN']=   ( 0.0 ,'Minimum Value of Image Array')
                    banzai_image_header['IMGMAX']=   ( 0.0 ,'Maximum Value of Image Array')
                    banzai_image_header['IMGMEAN']=  ( 0.0 ,'Mean Value of Image Array')
                    banzai_image_header['IMGMODE']=  ( 0.0 ,'Mode Value of Image Array')
                    banzai_image_header['IMGMED']=   ( 0.0 ,'Median Value of Image Array')
                    banzai_image_header['IMGMAD']=   ( 0.0 ,'Median Absolute Deviation of Image Array')
                    banzai_image_header['IMGSTDEV']= ( 0.0 ,'Standard Deviation of Image Array')
                    if 'CRVAL1' in eva_image_header.keys():
                        banzai_image_header['WCSERR']  = ( 0 ,'Error status of WCS fit. 0 for no error')
                    else:
                        banzai_image_header['WCSERR']  = ( 1 ,'Error status of WCS fit. 0 for no error')
                    banzai_image_header['PNTOFST'] = (   0.0 ,'[arcsec] offset of requested and solved center')
                if 'EVA-' in file:
                    banzai_image_header['RLEVEL']  = ( 92 ,'Reduction level')
                else:
                    banzai_image_header['RLEVEL']  = ( 93 ,'Reduction level')
                banzai_image_header['PIPEVER'] = ( 'EVA  '           ,'Pipeline version')
                banzai_image_header['L1PUBDAT']= ( '2024-09-23T22:02:16.963' ,'[UTC] Date the frame becomes public')
                #banzai_image_header['ZHECKSUM']= ( 'WAcaX7ZSWAbYW5ZY'   ,'HDU checksum updated 2024-09-23T22:02:38')
                #banzai_image_header['ZDATASUM']= ( '3136908556'         ,'data unit checksum updated 2024-09-23T22:02:38')

        except:
            logging.info(traceback.format_exc())


        # This routine creates a BANZAI-esque file with the main SCI image and header in it
        # Astropy expects the header to contain valid NAXIS information.  If
        # the header is incomplete (e.g. when NAXIS==0), CompImageHDU raises a
        # ZeroDivisionError internally while setting up compression tiles.  In
        # that case, the image is effectively unusable so skip creating the
        # BANZAI file for this frame.
        try:
            hdufz = fits.CompImageHDU(np.array(banzai_image, dtype=np.float32),
                                      banzai_image_header)
        except ZeroDivisionError:
            logging.info(
                "Skipping BZESK creation for %s due to invalid header (zero tile size)",
                file,
            )
            return

        bzesque_file.append(hdufz)

        # Now add in the source catalogue
        #
        # First convert astropy Table to fits (this function straight from BANZAI - 'table_to_fits')
        table_hdu = fits.BinTableHDU(sources)
        # Put in the description keywords
        for k in table_hdu.header.keys():
            if 'TTYPE' in k:
                column_name = table_hdu.header[k].lower()
                description = sources[column_name].description
                table_hdu.header[k] = (column_name.upper(), description)
                # Get the value of n in TTYPE
                n = k[5:]
                table_hdu.header['TCOMM{0}'.format(n)] = description

        table_hdu.header['EXTNAME'] = 'CAT'

        bzesque_file.append(table_hdu)

        # If EVA is in the filename, it is not a smartstack
        if 'EVA-' in file:
            if telescope == 'lco':
                origfilename=banzai_image_header['ORIGNAME']
                try:
                    standin_bpm_array=np.load(basedirectory+'/lcobpms/'+origfilename.replace('.fits.fz','.npy'))
                    # For single images, we actually crop them for wcs flatness
                    cropvalue_h=int(standin_bpm_array.shape[1]*0.05) + 20
                    cropvalue_w=int(standin_bpm_array.shape[0]*0.05) + 20
                    standin_bpm_array = standin_bpm_array[cropvalue_w:-cropvalue_w,cropvalue_h:-cropvalue_h]  
                    standin_bpm_array=fits.CompImageHDU(standin_bpm_array)
                except:
                    logging.info ("failed bpm for bzesk " + str(origfilename))
                    standin_bpm_array= fits.CompImageHDU(np.zeros(banzai_image.shape))
                
                
            else:                            
                # Create a BPM array (zeros if failed to open calibration)
                try:
                    standin_bpm_array=np.load(calibration_directory + '/' + banzai_image_header['INSTRUME'] + '/' + banzai_image_header['TELID'] +'_' + banzai_image_header['INSTRUME'] + '_badpixelmask_bin1.npy').astype(int)
                    # For single images, we actually crop them for wcs flatness
                    cropvalue_h=int(standin_bpm_array.shape[1]*0.05) + 20
                    cropvalue_w=int(standin_bpm_array.shape[0]*0.05) + 20
                    standin_bpm_array = standin_bpm_array[cropvalue_w:-cropvalue_w,cropvalue_h:-cropvalue_h]
                    # Then bung it in a compimagehdu
                    standin_bpm_array=fits.CompImageHDU(standin_bpm_array)                    
                except:
                    logging.info ("Failed to get BPM array")
                    logging.info(traceback.format_exc())
                    standin_bpm_array= fits.CompImageHDU(np.zeros(banzai_image.shape))
        else:
            # If it is a smartstack, then by definition there are no bad pixels as they will have been dithered and medianed out
            standin_bpm_array= fits.CompImageHDU(np.zeros(banzai_image.shape))

        standin_bpm_array.header['EXTNAME'] = 'BPM'

        bzesque_file.append(standin_bpm_array)

        # If EVA is in the filename, it is not a smartstack
        if 'EVA-' in file:
            if telescope == 'lco':
                origfilename=banzai_image_header['ORIGNAME']
                try:
                    standin_err_array=np.load(basedirectory+'/lcoerrays/'+origfilename.replace('.fits.fz','.npy'))
                    # For single images, we actually crop them for wcs flatness
                    cropvalue_h=int(standin_err_array.shape[1]*0.05) + 20
                    cropvalue_w=int(standin_err_array.shape[0]*0.05) + 20
                    standin_err_array = standin_err_array[cropvalue_w:-cropvalue_w,cropvalue_h:-cropvalue_h]
                    standin_err_array= fits.CompImageHDU(standin_err_array)
                except:
                    logging.info ("Failed to get LCO VARIANCE ARRAY array"  + str(origfilename))
                    #logging.info(traceback.format_exc())
                    standin_err_array=np.empty(banzai_image.shape).fill(1.0)
                    standin_err_array= fits.CompImageHDU(standin_err_array)
            else:
                # Create an ERR array (1 if failed to open calibration)
                try:
                    standin_err_array=np.load(calibration_directory + '/' + banzai_image_header['INSTRUME'] + '/' + banzai_image_header['TELID'] +'_' + banzai_image_header['INSTRUME'] + '_readnoise_variance_adu.npy')
                    
                    # For single images, we actually crop them for wcs flatness
                    cropvalue_h=int(standin_err_array.shape[1]*0.05) + 20
                    cropvalue_w=int(standin_err_array.shape[0]*0.05) + 20
                    standin_err_array = standin_err_array[cropvalue_w:-cropvalue_w,cropvalue_h:-cropvalue_h]     
                    
                    # create the err array. 
                    # Just incorporating readnoise for the moment, this is most of our error. Next to incoporate is dark noise
                    standin_err_array= fits.CompImageHDU(np.sqrt((np.abs(banzai_image) + standin_err_array*banzai_image_header['GAIN'])))
                    
                except:
                    logging.info ("Failed to get VARIANCE array")
                    logging.info(traceback.format_exc())
                    standin_err_array=np.empty(banzai_image.shape).fill(1.0)
                    standin_err_array= fits.CompImageHDU(standin_err_array)
        else:
            #If it is a smartstack, then a reprojected variance array was created during the smartstack process.
            logging.info ("ready to load the smartstack variance array")

            standin_err_array=np.array(fits.open(file.replace('SmSTACK-','variance_SmSTACK-'))[0].data)
            standin_err_array= fits.CompImageHDU(standin_err_array)
            
        standin_err_array.header['EXTNAME'] = 'ERR'
        bzesque_file.append(standin_err_array)

        output_bzesk_filename=file.replace('EVA-','BZESK-').replace('SmSTACK-','BZESKSmSTACK-').replace('.fits','.fits.fz')
        try:
            bzesque_file.writeto(output_bzesk_filename, overwrite=True)
        except:
            logging.info("Failed to make bzesk file: " + str(output_bzesk_filename))


def make_quickanalysis_file(file):
    """Create a quickanalysis JSON from photometry files."""
    temp_header = fits.getheader(file, memmap=False)
    if temp_header.get('EXPTIME', 0) < 1:
        logging.info("Not making quickanalysis file as it isn't a file to be ingested")
        return

    psxfile = file.replace('outputdirectory', 'photometry').replace('.fits', '.psx')\
                    .replace('EVA-', 'psxphot-').replace('SmSTACK-', 'psxphotSmSTACK-')\
                    .replace('LoSTACK-', 'psxphotLoSTACK-')
    sekfile = file.replace('outputdirectory', 'photometry').replace('.fits', '.sek')\
                    .replace('EVA-', 'sekphot-').replace('SmSTACK-', 'sekphotSmSTACK-')\
                    .replace('LoSTACK-', 'sekphotLoSTACK-')

    source_info = []
    src_path = None
    if os.path.exists(psxfile):
        src_path = psxfile
    elif os.path.exists(sekfile):
        src_path = sekfile

    rmax = int(max(1, 3 * float(temp_header.get('FWHMpix', temp_header.get('FWHM', 3)))))
    try:
        image_data = fits.getdata(file, memmap=False)
    except Exception:
        image_data = None

    if src_path:
        with open(src_path, 'r') as f:
            header_read = False
            for line in f:
                if not header_read:
                    header_read = True
                    continue
                if line.strip() and not line.startswith('#'):
                    raw = line.strip()
                    profile = []
                    fit_params = None
                    try:
                        parts = [float(x) for x in raw.split(',')]
                        xpix, ypix = parts[2], parts[3]
                        if image_data is not None:
                            profile = _radial_profile(image_data, (xpix, ypix), rmax)
                            fit_params = _fit_moffat(profile)
                    except Exception:
                        pass
                    source_info.append({'raw': raw, 'profile': profile, 'moffat_fit': fit_params})

    qa_data = {'source_list': source_info}

    if image_data is not None:
        try:
            rows, cols = image_data.shape
            mid_row = rows // 2
            mid_col = cols // 2

            slices = {
                'row': image_data[mid_row, :].astype(float).tolist(),
                'column': image_data[:, mid_col].astype(float).tolist(),
                'diag1': np.diag(image_data).astype(float).tolist(),
                'diag2': np.diag(np.fliplr(image_data)).astype(float).tolist()
            }

            grid = 10
            box_rows = rows // grid
            box_cols = cols // grid
            box_means = []
            for i in range(grid):
                row_vals = []
                for j in range(grid):
                    box = image_data[i*box_rows:(i+1)*box_rows,
                                       j*box_cols:(j+1)*box_cols]
                    row_vals.append(float(bn.nanmean(box)))
                box_means.append(row_vals)

            flat = image_data.ravel()[~np.isnan(image_data.ravel())].astype(int)
            unique, counts = np.unique(flat, return_counts=True)
            histogram = [[int(u), int(c)] for u, c in zip(unique, counts)]

            qa_data.update({
                'slices': slices,
                'box_means': box_means,
                'histogram': histogram
            })
        except Exception:
            pass

    dest = file.replace('outputdirectory', 'quickanalysis')\
              .replace('EVA-', 'quickanalysis-')\
              .replace('SmSTACK-', 'quickanalysisSmSTACK-')\
              .replace('LoSTACK-', 'quickanalysisLoSTACK-')\
              .replace('.fits', '.json')

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + '.temp', 'w') as fp:
        json.dump(qa_data, fp, indent=4)
    os.rename(dest + '.temp', dest)
