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
import logging

def crop_images_for_wcs_flatness(file):

    imagedata=np.load('workingdirectory/' +file)

    # We are actually going to crop the image
    # So we need to adjust the size here also.
    # Even though we don't adjust it in this section.

    cropvalue_h=int(imagedata.shape[1]*0.05)
    cropvalue_w=int(imagedata.shape[0]*0.05)

    imagedata = imagedata[cropvalue_w:-cropvalue_w,cropvalue_h:-cropvalue_h]

    np.save('workingdirectory/' +file, imagedata)
    return cropvalue_h,cropvalue_w

##### Crop single images to get rid of regions with probably dodgy wcs values
##### For both individual images, but also so strange stretches don't emerge in stacks

def multiprocess_crop_images_for_flatness(header):

    cropvalue_h,cropvalue_w=crop_images_for_wcs_flatness(header['ORIGNAME'].replace('.fits.fz','.npy').replace('.fits','.npy'))

    if any("CRPIX1" in s for s in header.keys()):
        header['CRPIX1']=header['CRPIX1']-cropvalue_h
        header['CRPIX2']=header['CRPIX2']-cropvalue_w
    header['NAXIS1']=header['NAXIS1']-(2*cropvalue_h)
    header['NAXIS2']=header['NAXIS2']-(2*cropvalue_w)
    if any("IMAGEH" in s for s in header.keys()):
        header['IMAGEH']=header['IMAGEH']-(2*cropvalue_h)
        header['IMAGEW']=header['IMAGEW']-(2*cropvalue_w)

    return header

            
def get_source_spread_at_given_source_depths(catalogue, source_starting_depth, intended_length):
    
    """
    Calculates the spread of sources in a given catalogue after sorting and trimming based on source flux.

    Args:
        catalogue (np.ndarray): A 2D array where each row represents a source, and columns represent:
            - Column 0: Right Ascension (RA)
            - Column 1: Declination (Dec)
            - Column 2: flux
        source_starting_depth (int): Number of top sources (based on flux) to exclude from the analysis.
        intended_length (int): Target number of sources to retain after trimming.

    Returns:
        tuple:
            - float: Standard deviation of the spread of sources across grid cells, representing how uniformly the sources are distributed.
            - np.ndarray: Trimmed and sorted catalogue of sources.

    Process:
        1. Determines the upper and lower bounds for RA and Dec.
        2. Sorts the catalogue by source strength (descending) and removes the top `source_starting_depth` sources.
        3. Trims the catalogue to the specified `intended_length`.
        4. Divides the sky area into a 4x4 grid and assigns each source to a grid cell based on its RA and Dec.
        5. Computes the proportion of sources in each grid cell.
        6. Calculates and returns the standard deviation of these proportions to measure the spread of sources.
    """

    upperlimit_ra=max(catalogue[:,0])
    lowerlimit_ra=min(catalogue[:,0])

    upperlimit_dec=max(catalogue[:,1])
    lowerlimit_dec=min(catalogue[:,1])

    # Sort catalogue
    catalogue=catalogue[catalogue[:,2].argsort()[::-1]]
    catalogue=catalogue[source_starting_depth:, :]

    while len(catalogue) > intended_length:
        catalogue = np.delete(catalogue, [-1], 0)

    # Separate out into grid
    N_of_boxes=4

    grid_holder={}
    for i in range (N_of_boxes*N_of_boxes):
        grid_holder[i] = []

    ra_boxsize=(upperlimit_ra - lowerlimit_ra) / N_of_boxes
    dec_boxsize=(upperlimit_dec - lowerlimit_dec) / N_of_boxes

    for sourceline in catalogue:
        counter=0
        for i in range(3):
            for t in range(3):

                ra_left=lowerlimit_ra + (i*ra_boxsize)
                ra_right=lowerlimit_ra + ((i+1)*ra_boxsize)

                dec_left=lowerlimit_dec + (t*dec_boxsize)
                dec_right=lowerlimit_dec + ((t+1)*dec_boxsize)

                if ra_left <= sourceline[0] <= ra_right:
                    if dec_left <= sourceline[1] <= dec_right:
                        grid_holder[counter].append(sourceline)


                counter=counter+1

    spread_in_proportions=[]
    for counter in range(N_of_boxes*N_of_boxes):
        spread_in_proportions.append(len(grid_holder[counter])/len(catalogue))


    return np.std(np.array(spread_in_proportions)), catalogue

def get_source_spread_through_catalogue(source_catalogue):
    """
    Measures the spread of sources through a catalogue at different depths and returns the spread values and catalogues.

    Args:
        source_catalogue (np.ndarray): A 2D array where each row represents a source, with columns representing:
            - Column 0: Right Ascension (RA)
            - Column 1: Declination (Dec)
            - Column 2: Source flux

    Returns:
        tuple:
            - list: A list of [starting depth, spread value] pairs, where the spread value is the standard deviation of the source distribution across grid cells.
            - list: A list of trimmed catalogues at different depths.

    Process:
        1. Determines the intended sample size based on the total number of sources:
            - >200 sources → use 200 sources.
            - >100 sources → use 50 sources.
            - >50 sources → use 25 sources.
            - ≤50 sources → run with the full catalogue and return default values.
        2. Iteratively processes the catalogue in increments of 10 sources:
            - Calls `get_source_spread_at_given_source_depths()` to compute the spread at each depth.
            - Stores the resulting spread and catalogue.
        3. Returns the list of spread values and the corresponding catalogues.
    """


    full_length=len(source_catalogue)

    if full_length > 200:
        intended_length=200
    elif full_length > 100:
        intended_length=50
    elif full_length > 50:
        intended_length=25
    else:
        logging.info ("run with full catalogue")
        return [[0,1]], [source_catalogue]

    spreads=[]
    catalogues=[]

    test_counter=0
    while test_counter+intended_length < full_length:

        stdev_across_boxes, actual_catalogue=get_source_spread_at_given_source_depths(source_catalogue, test_counter, intended_length)
        spreads.append([test_counter, stdev_across_boxes])
        catalogues.append(actual_catalogue)

        test_counter=test_counter+10

    return spreads, catalogues