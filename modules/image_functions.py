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
from astropy.convolution import Gaussian2DKernel#, interpolate_replace_nans
import bottleneck as bn
from photutils.detection import DAOStarFinder
import copy
from astropy.stats import mad_std
from photutils.background import Background2D, MedianBackground
from astroscrappy import detect_cosmics
import warnings
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
import time

kernel = Gaussian2DKernel(x_stddev=2,y_stddev=2)

def sigma_clip_mad(data, sigma=2.5, maxiters=10):
    """
    Perform sigma clipping using MAD as a robust standard deviation estimate.
    
    Parameters:
        data (np.ndarray): Input array.
        sigma (float): Sigma threshold for clipping.
        maxiters (int): Maximum number of iterations.
    
    Returns:
        np.ndarray: Array with values outside the sigma range replaced by np.nan.
    """
    clipped_data = data.copy()  # Copy the data to avoid modifying the original array
    
    for iter in range(maxiters):

        if iter < (maxiters-1):
            # Compute the mean and standard deviation, ignoring NaN values
            median = bn.nanmedian(clipped_data)
            std = bn.nanstd(clipped_data)
            
            # Identify the mask of outliers
            mask = np.abs(clipped_data - median) > sigma * std
        else:
            # Compute the median of the current data
            median = bn.nanmedian(clipped_data)
            # Compute the MAD and scale it to approximate standard deviation
            mad = bn.nanmedian(np.abs(clipped_data - median))
            mad_std = mad * 1.4826
            
            # Identify the mask of outliers
            mask = np.abs(clipped_data - median) > sigma * mad_std
        
        # If no more values are being clipped, break the loop
        if not np.any(mask):
            break
        
        # Replace outliers with np.nan
        clipped_data[mask] = np.nan
    
    return clipped_data

def mid_stretch_jpeg(data):
    """
    Apply a midtones stretch to the image data using the PixInsight method.

    Args:
        data (np.array): The original image data array.

    Returns:
        np.array: The stretched image data.
    """

    target_bkg = 0.25
    shadows_clip = -1.25

    # Normalize data
    try:
        data = data / np.max(data)
    except ZeroDivisionError:
        pass  # Avoids division by zero if the image is flat

    # Compute average deviation from the median
    median = np.median(data.ravel())
    avg_dev = np.mean(np.abs(data - median))
    c0 = np.clip(median + (shadows_clip * avg_dev), 0, 1)

    # Midtones Transfer Function (MTF)
    def apply_mtf(x, m):
        """Applies the Midtones Transfer Function to an array."""
        shape = x.shape
        x = x.ravel()

        zeros = x == 0
        halfs = x == m
        ones = x == 1
        others = ~(zeros | halfs | ones)

        x[zeros] = 0
        x[halfs] = 0.5
        x[ones] = 1
        x[others] = (m - 1) * x[others] / ((((2 * m) - 1) * x[others]) - m)

        return x.reshape(shape)

    # Initial stretch
    x = median - c0
    m = apply_mtf(x, target_bkg)

    # Clip everything below the shadows clipping point
    data[data < c0] = 0
    above = data >= c0

    # Apply the stretch to the remaining pixels
    x = (data[above] - c0) / (1 - c0)
    data[above] = apply_mtf(x, m)

    return data

def detect_and_mask_stars(imagedata=None,fwhm=12.0,noise_level=10, pixscale=None):
    
    # Detect stars using DAOStarFinder
    daofind = DAOStarFinder(fwhm=fwhm, threshold=5 * noise_level)
    stars = daofind(imagedata)       
    
    if stars is None:     
        #No stars detected, skipping star removal
        image_star_masked=copy.deepcopy(imagedata)
    else:
        fwhm = 2.355 / np.sqrt(np.abs(stars['sharpness']))
        stars['fwhm'] = fwhm  # Add the FWHM as a new column
               
        # Create an empty mask with the same shape as the image
        star_mask = np.zeros_like(imagedata, dtype=bool)
        
        # Mask stars based on their centroids and FWHM
        stars_masked=0
        for star in stars:
            
            if star['fwhm'] > (1.2/pixscale):
            
                x, y = int(star['xcentroid']), int(star['ycentroid'])
                radius = int(star['fwhm'] * 2)  # Mask area 2x the FWHM
                
                y_min, y_max = max(0, y - radius), min(imagedata.shape[0], y + radius)
                x_min, x_max = max(0, x - radius), min(imagedata.shape[1], x + radius)
                star_mask[y_min:y_max, x_min:x_max] = True
                stars_masked=stars_masked+1
                
        image_star_masked=copy.deepcopy(imagedata)
        image_star_masked[star_mask] = np.nan
    
    return image_star_masked


def mask_cosmics(imagedata, saturate=65000, gain=1.0, rdnoise=10.0, imageMode=200, pixscale=None, telescopename='other'):    
    
    # Mask NaN and Inf values
    nanmask = np.isfinite(imagedata)
    # Remove stars from the image
    # Estimate the noise level (standard deviation of the background)
    noise_level = mad_std(imagedata[nanmask])
    image_star_masked=detect_and_mask_stars(imagedata=imagedata,fwhm=12.0,noise_level=noise_level, pixscale=pixscale)
    
    # Need to remove the background
    bkg_estimator = MedianBackground()
    bkg = Background2D(image_star_masked, box_size=50, filter_size=3, bkg_estimator=bkg_estimator)
    background = bkg.background
    image_star_masked=image_star_masked-background
        
    # Make sure that there are only positive values in the image for cosmic ray subtraction
    # Step 1: Determine the offset
    min_val = np.min(image_star_masked)
    if min_val < 0:
        offset = abs(min_val) + 100
    else:
        offset = 100  # No adjustment needed if all values are already positive
    
    # Step 2: Apply the offset
    image_star_masked = image_star_masked + offset
    
    if telescopename == 'lco':
        # Do cosmic rays
        mask, _ = detect_cosmics(image_star_masked, cleantype='meanmask', gain=gain, readnoise=rdnoise, sepmed=True, sigclip = 4.5 , sigfrac = 0.3 , objlim = 5.0 , niter=4, satlevel=saturate, verbose=True)
        imagedata[mask]=np.nan
    else:
        # Do cosmic rays        
        iteration=0.0
        while True:
            mask, _ = detect_cosmics(image_star_masked, cleantype='meanmask', gain=gain, readnoise=rdnoise, sepmed=True, sigclip = 4.5 + iteration , sigfrac = 0.3 , objlim = 5.0 +iteration, niter=2, satlevel=saturate, verbose=True)
            
            print ("Image size " + str(image_star_masked.size)) 
            print ("Cosmic ray pixels : " + str(np.sum(mask)))
            fraction_of_pixels=np.sum(mask) / np.sum(image_star_masked.size)
            print ("Fraction : " + str(fraction_of_pixels))
            if fraction_of_pixels < 0.05:
                imagedata[mask]=np.nan
                break
            if iteration==5.0:
                break
            
            iteration=iteration+0.5               
            
    return imagedata

def find_nan_crop_limits(imagedata):
    """
    Find the cropping limits of a 2D array by removing rows and columns filled with NaNs.

    This function determines the first and last rows and columns that contain valid (non-NaN) data 
    in a 2D array. It can be used to crop out edges filled entirely with NaNs.

    Args:
        imagedata (numpy.ndarray): 2D array of image data.

    Returns:
        tuple: (firstrow, lastrow, firstcol, lastcol) representing the indices of the 
               first and last rows and columns that contain valid data.
    """
    nans=np.isnan(imagedata)
    nancols=np.all(nans,axis=0)
    nanrows=np.all(nans,axis=1)
    firstcol=nancols.argmin()
    firstrow=nanrows.argmin()
    lastcol= len(nancols) - nancols[::-1].argmin()
    lastrow= len(nanrows) - nanrows[::-1].argmin()
    return firstrow, lastrow, firstcol, lastcol

def thresh(imagedata, saturate=65000):
    """
    Apply thresholding to an image array.

    This function processes a 2D array of image data by:
    - Removing excessively high and low pixel values by setting them to NaN.
    - Identifying the most common pixel value (mode) in the image.
    - Finding a reasonable lower cutoff value based on the mode and removing values below it.

    Args:
        imagedata (numpy.ndarray): 2D array of image data.
        saturate (int, optional): Saturation limit for the image. Defaults to 65000.

    Returns:
        tuple:
            - numpy.ndarray: Processed image data with thresholded values set to NaN.
            - int: Mode of the image data.
    """

    #Thresholding image
    highCut=saturate*1.5 # arbitrarily high value where it makes no sense to have a pixel this brgith

    # Cutting off the top and the bottom of the image
    imagedata[imagedata > highCut] = np.nan
    imagedata[imagedata < -10000] = np.nan # Surely -10000 is a reasonable floor value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        int_array_flattened=imagedata.ravel()[~np.isnan(imagedata.ravel())].astype(int)

    unique,counts=np.unique(int_array_flattened, return_counts=True)
    
    m=counts.argmax()
    imageMode=unique[m]
    histogramdata=np.column_stack([unique,counts]).astype(np.int32)

    # Find the value just (more than 16 below) below the mode that is missing from the array
    zeroValueArray = histogramdata[histogramdata[:, 0] < imageMode][:, 0]
    counter = 0
    while True:
        counter += 1
        if all((imageMode - (counter + i)) not in zeroValueArray for i in range(17)):
            zeroValue = imageMode - counter
            break

    imagedata[imagedata < zeroValue] = np.nan

    return imagedata, imageMode

def deviation_from_surroundings(data, window_size=20, weight_type="gaussian"):
    """
    Computes the deviation of each entry from its surrounding ±window_size pixels, 
    weighted more heavily to nearby pixels.

    Parameters:
        data (np.ndarray): The 1D input array.
        window_size (int): The range around each pixel to consider (default is 20).
        weight_type (str): Type of weighting ('gaussian' or 'triangular').

    Returns:
        np.ndarray: The array of deviations.
    """
    # Create weights
    if weight_type == "gaussian":
        sigma = window_size / 2.0
        weights = np.exp(-0.5 * (np.arange(-window_size, window_size + 1) / sigma) ** 2)
    elif weight_type == "triangular":
        weights = 1 - (np.abs(np.arange(-window_size, window_size + 1)) / (window_size + 1))
    else:
        raise ValueError("Unsupported weight_type. Use 'gaussian' or 'triangular'.")

    # Normalize weights to sum to 1
    weights /= weights.sum()

    # Convolve the data with the weights to get the weighted moving average
    padded_data = np.pad(data, (window_size, window_size), mode="reflect")
    weighted_avg = np.convolve(padded_data, weights, mode="valid")

    # Calculate deviations
    deviations = data - weighted_avg

    return deviations

# Linear interpolation
def linear_interpolate(arr):
    nans = np.isnan(arr)
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr

def debanding (bandeddata):

    # Store the current nans as a mask to reapply later
    nan_mask=copy.deepcopy(np.isnan(bandeddata))    

    ysize=bandeddata.shape[1]

    sigma_clipped_array=copy.deepcopy(bandeddata)        
    sigma_clipped_array = sigma_clip_mad(sigma_clipped_array, sigma=2.5, maxiters=4)
    
    # Do rows
    rows_median = bn.nanmedian(sigma_clipped_array,axis=1)    
    
    rows_deviations=deviation_from_surroundings(rows_median, window_size=20, weight_type="gaussian")    
    
    # Get the middle 100 pixels
    # mid_start = (len(rows_deviations) - 200) // 2  # Starting index of the middle 100 pixels
    # mid_end = mid_start + 200          # Ending index (exclusive)
    # middle_100_pixels = rows_deviations[mid_start:mid_end]
    # plot_1d_array(middle_100_pixels) 
    
    #remove nans
    rows_deviations=linear_interpolate(rows_deviations)

    row_debanded_image=bandeddata-np.tile(rows_deviations[:,None],(1,ysize))
    row_debanded_image= np.subtract(bandeddata,rows_deviations[:,None])
    
    # Then run this on columns
    # sigma_clipped_array=copy.deepcopy(row_debanded_image)
    # sigma_clipped_array = sigma_clip_mad(sigma_clipped_array, sigma=2.5, maxiters=4)
    columns_median = bn.nanmedian(sigma_clipped_array,axis=0)        
    columns_deviations=deviation_from_surroundings(columns_median, window_size=20, weight_type="gaussian")
    
    #remove nans
    columns_deviations=linear_interpolate(columns_deviations)
    
    both_debanded_image= row_debanded_image-columns_deviations[None,:]

    #Reapply the original nans after debanding
    both_debanded_image[nan_mask] = np.nan

    return both_debanded_image

def calculate_image_fwhm(imagedata, header):
    # Remove NaNs and replace with median value
    image_median = np.nanmedian(imagedata)
    imagedata = np.nan_to_num(imagedata, nan=image_median)

    # Crop image to central 1 degree x 1 degree if larger
    pixscale = float(header['PIXSCALE'])
    fx, fy = imagedata.shape
    fx_degrees = (fx * pixscale) / 3600
    fy_degrees = (fy * pixscale) / 3600

    if fx_degrees > 1.0:
        crop_x = int((fx - (fx / fx_degrees)) / 2)
        imagedata = imagedata[crop_x:-crop_x, :]
    if fy_degrees > 1.0:
        crop_y = int((fy - (fy / fy_degrees)) / 2)
        imagedata = imagedata[:, crop_y:-crop_y]

    # Estimate background statistics
    mean, median, std = sigma_clipped_stats(imagedata, sigma=3.0)
    
    # Use DAOStarFinder to detect stars
    threshold = max(3 * std, 200 * pixscale)
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold)
    sources = daofind(imagedata - median)

    if sources is None or len(sources) == 0:
        header["FWHM"] = (-99, 'FWHM in pixels')
        header["FWHMasec"] = (-99, 'FWHM in arcseconds')
        header["NSTARS"] = (0, 'Number of star-like sources in image')
        return header

    # Fit Gaussian to brightest stars to measure FWHM
    fwhm_list = []
    for source in sources[:min(len(sources), 200)]:
        x, y = source['xcentroid'], source['ycentroid']
        cutout = imagedata[int(y)-5:int(y)+5, int(x)-5:int(x)+5]
        
        if cutout.shape != (10, 10):
            continue
        
        # Fit a 2D Gaussian to the star
        y, x = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
        gaussian_init = models.Gaussian2D(amplitude=cutout.max(), x_mean=5, y_mean=5)
        fitter = fitting.LevMarLSQFitter()
        fitted_gaussian = fitter(gaussian_init, x, y, cutout)

        # FWHM from the fitted Gaussian (in pixels)
        fwhm_x = 2.355 * abs(fitted_gaussian.x_stddev.value)
        fwhm_y = 2.355 * abs(fitted_gaussian.y_stddev.value)
        fwhm = np.mean([fwhm_x, fwhm_y])
        fwhm_list.append(fwhm)

    if len(fwhm_list) > 0:
        fwhm_median = np.median(fwhm_list)
        fwhm_arcsec = fwhm_median * pixscale
        fwhm_std = np.std(fwhm_list) * pixscale

        header["FWHM"] = (fwhm_median, 'FWHM in pixels')
        header["FWHMasec"] = (fwhm_arcsec, 'FWHM in arcseconds')
        header["FWHMstd"] = (fwhm_std, 'FWHM standard deviation in arcseconds')
        header["NSTARS"] = (len(fwhm_list), 'Number of star-like sources in image')
    else:
        header["FWHM"] = (-99, 'FWHM in pixels')
        header["FWHMasec"] = (-99, 'FWHM in arcseconds')
        header["FWHMstd"] = (-99, 'FWHM standard deviation in arcseconds')
        header["NSTARS"] = (0, 'Number of star-like sources in image')

    return header

def cyclic_next_door_neighbour_interpolate_nans(imagedata):
    
    """
    Fills NaN values in a 2D array using neighboring values. 
    Cycles through the array, attempting to fill NaNs based on the number of available non-NaN neighbors, 
    starting with cells surrounded by four valid neighbors and working down to one. 
    Stops when no more changes occur or after a timeout limit of 300 seconds.
    
    Args:
        imagedata (np.ndarray): 2D array containing NaN values to be filled.
    
    Returns:
        np.ndarray: Array with NaN values interpolated where possible.
    """
    
    timeout_limit = 300
    timeout_timer = time.time()
    
    x_size, y_size = imagedata.shape

    for interp_limit in [4, 3, 2, 1]:
        while np.isnan(imagedata).sum() > 0:
            if (time.time() - timeout_timer) > timeout_limit:
                print("Timed out!")
                return imagedata

            nan_coords = np.argwhere(np.isnan(imagedata))
            updated = False

            for x, y in nan_coords:
                neighbors = []
                
                # Collect neighboring values if within bounds and not NaN
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < x_size and 0 <= ny < y_size and not np.isnan(imagedata[nx, ny]):
                        neighbors.append(imagedata[nx, ny])
                
                if len(neighbors) >= interp_limit:
                    imagedata[x, y] = bn.nanmean(neighbors)
                    updated = True

            if not updated:  # If no changes, break the loop
                break

    return imagedata