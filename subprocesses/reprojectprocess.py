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

import pickle
import sys
from astropy.io import fits
from datetime import datetime
from reproject import reproject_adaptive, reproject_interp #reproject_exact
import numpy
import copy

input_reproject_info=pickle.load(sys.stdin.buffer)

file=input_reproject_info[0]
counter=input_reproject_info[1]
wcs_out=input_reproject_info[2]
shape_out=input_reproject_info[3]
readnoise_variance_file=input_reproject_info[4]
bias_file=input_reproject_info[5]
biasdark_file=input_reproject_info[6]
lco=input_reproject_info[7]

# First reproject the actual image

print(datetime.now().strftime("%H:%M:%S"))
print("Reprojecting Image Array: " + file)
hdufirst = fits.open(file)[0]   # This opens the fits file .....    print (hdufirst.data) # <----- this is how you get the data out of the opened fits file

# Get rid of astropy overhead
image_to_reproject=numpy.array(hdufirst.data)
image_header=copy.deepcopy(hdufirst.header)
del hdufirst

array = reproject_adaptive((image_to_reproject,image_header), wcs_out, shape_out=shape_out, return_footprint=False, kernel='gaussian', conserve_flux=True, boundary_mode='strict') # This reprojects the single image into a larger image called "array"

#array = reproject_adaptive(image_to_reproject, wcs_out, shape_out=shape_out, return_footprint=False, kernel='gaussian', conserve_flux=True, boundary_mode='strict') # This reprojects the single image into a larger image called "array"
#array = (reproject_exact(hdufirst, wcs_out, shape_out=shape_out, return_footprint=False, parallel=True))
#array = reproject_exact((image_to_reproject,image_header), wcs_out, shape_out=shape_out, return_footprint=False)
tempfilename=file.replace('.fits','.'+str(counter)+'.reproject.npy')
numpy.save(file.replace('.fits','.'+str(counter)+'.reproject.npy'),array)
print ("Saved as " + str(tempfilename))
print(datetime.now().strftime("%H:%M:%S"))
#PLDrive = numpy.memmap('tempfile'+str(i), dtype='float32', mode= 'w+', shape = shape_out) # Make a file to put the array in - note we are using shape_out from the previous command

del array # Remove them once we are done
#del hdufirst # Remove them once we are done


if not readnoise_variance_file  == 'none':

    # Then reproject the variance array
    
    print(datetime.now().strftime("%H:%M:%S"))
    print("Reprojecting Variance Array: " + file)
    #hdufirst = fits.open(variance_file)[0]   # This opens the fits file .....    print (hdufirst.data) # <----- this is how you get the data out of the opened fits file
    
    # Get rid of astropy overhead
    if lco:
        image_to_reproject=numpy.load(readnoise_variance_file) 
    
    else:
        image_to_reproject=numpy.load(readnoise_variance_file) + numpy.load(biasdark_file) - numpy.load(bias_file)
    #image_header=copy.deepcopy(hdufirst.header)
    #del hdufirst
    
    array = reproject_interp((image_to_reproject,image_header), wcs_out, shape_out=shape_out, return_footprint=False)#, kernel='gaussian', conserve_flux=True, boundary_mode='strict') # This reprojects the single image into a larger image called "array"
    
    #array = reproject_adaptive(image_to_reproject, wcs_out, shape_out=shape_out, return_footprint=False, kernel='gaussian', conserve_flux=True, boundary_mode='strict') # This reprojects the single image into a larger image called "array"
    #array = (reproject_exact(hdufirst, wcs_out, shape_out=shape_out, return_footprint=False, parallel=True))
    #array = reproject_exact((image_to_reproject,image_header), wcs_out, shape_out=shape_out, return_footprint=False)
    tempfilename=file.replace('.fits','.'+str(counter)+'.reprojectedvariance.npy')
    numpy.save(file.replace('.fits','.'+str(counter)+'.reprojectedvariance.npy'),array)
    print ("Saved as " + str(tempfilename))
    print(datetime.now().strftime("%H:%M:%S"))
    #PLDrive = numpy.memmap('tempfile'+str(i), dtype='float32', mode= 'w+', shape = shape_out) # Make a file to put the array in - note we are using shape_out from the previous command
    
    del array # Remove them once we are done
    #del hdufirst # Remove them once we are done