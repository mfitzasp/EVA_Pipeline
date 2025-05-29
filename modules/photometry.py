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
import os
import subprocess
import math
import shutil
import copy

def run_source_extractor(file, codedir):

    prihdr = fits.open(file)[0].header
    saturlevel=prihdr['SATURATE']
    gain=prihdr['GAIN']
    try:
        seeingfwhm=float(prihdr['FWHMASEC'])
        if seeingfwhm < 1:
           seeingfwhm=2.5
    except:
        seeingfwhm=2.5
    photapertures=max(5.0/float(prihdr['PIXSCALE']),3)

    tempdir=file.replace('.','d')
    try:
        os.makedirs(tempdir, mode=0o777)
    except:
        pass

    # Calculate broadly minimum area from the seeing and the pixelscale
    fwhmpixels= seeingfwhm / float(prihdr['PIXSCALE'])
    psfarea= math.pi * pow((fwhmpixels /2),2)
    minarea= 1.0 * psfarea    
    backsize= 4 * fwhmpixels
    
    print ('source-extractor', file, '-c', os.path.expanduser(codedir) +'/photometryparams/default.sexfull',
     '-CATALOG_NAME', str(tempdir + '/test.cat'), '-SATUR_LEVEL', str(saturlevel), '-GAIN', str(gain), '-BACKPHOTO_TYPE','LOCAL', '-BACK_SIZE', str(backsize), '-BACK_FILTERSIZE',str(4), '-DETECT_MINAREA', str(minarea),'-DETECT_THRESH', str(2.5), '-ANALYSIS_THRESH',str(2.5),
     '-SEEING_FWHM', str(seeingfwhm), '-PHOT_APERTURES', str(photapertures), '-FILTER_NAME', str(os.path.expanduser(codedir) +'/photometryparams/sourceex_convs/gauss_2.0_5x5.conv'))

    tempprocess = subprocess.Popen(
        ['source-extractor', file, '-c', os.path.expanduser(codedir) +'/photometryparams/default.sexfull',
         '-CATALOG_NAME', str(tempdir + '/test.cat'), '-SATUR_LEVEL', str(saturlevel), '-GAIN', str(gain), '-BACKPHOTO_TYPE','LOCAL', '-BACK_SIZE', str(backsize), '-BACK_FILTERSIZE',str(4), '-DETECT_MINAREA', str(minarea),'-DETECT_THRESH', str(2.5), '-ANALYSIS_THRESH',str(2.5),
         '-SEEING_FWHM', str(seeingfwhm), '-PHOT_APERTURES', str(photapertures), '-FILTER_NAME', str(os.path.expanduser(codedir) +'/photometryparams/sourceex_convs/gauss_2.0_5x5.conv'),'-PARAMETERS_NAME',  str(os.path.expanduser(codedir) +'/photometryparams/default.paramsek')], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, bufsize=0)
    tempprocess.wait()

    shutil.copy(str(tempdir+'/test.cat'), file.replace('EVA-','sekphot-').replace('SmSTACK-','sekphotSmSTACK-').replace('LoSTACK-','sekphotLoSTACK-').replace('.fits','.fullsek'))

    with open(file.replace('EVA-','seaphot-').replace('SmSTACK-','seaphotSmSTACK-').replace('LoSTACK-','seaphotLoSTACK-').replace('.fits','.sea'),'w') as q:
        with open(file.replace('EVA-','sekphot-').replace('SmSTACK-','sekphotSmSTACK-').replace('LoSTACK-','sekphotLoSTACK-').replace('.fits','.sek'),'w') as p:
            with open(str(tempdir+'/test.cat'), 'r') as f:
                p.write('RA,Dec,Xpix,Ypix,Counts,Counts_err,Ra_err,Dec_err' + '\n')
                q.write('RA,Dec,Xpix,Ypix,Counts,Counts_err,Ra_err,Dec_err' + '\n')
                for line in f:
                    if not '#' in line:
                        
                        splitline=line.split(' ')
                        splitline=[i for i in splitline if i]
                        
                        # First check it rises above S/N = 30
                        if float(splitline[7])/float(splitline[8]) > 10:
                            # Check that is plausibly sized
                            if float(splitline[27]) > 0.5*fwhmpixels:

                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
        
        
                                del splitline[0]
                                splitline[9]=splitline[9].replace('\n','')
        
                                sealine=copy.deepcopy(splitline)
                                del sealine[6]
                                del sealine[6]
                                sealine=','.join(sealine)
                                q.write(sealine + '\n')
        
                                sekline=copy.deepcopy(splitline)
                                del sekline[4]
                                del sekline[4]
                                sekline=','.join(sekline)
                                p.write(sekline + '\n')

def run_pre_psfex(file, codedir):

    prihdr = fits.open(file)[0].header
    saturlevel=prihdr['SATURATE']
    gain=prihdr['GAIN']
    try:
        seeingfwhm=float(prihdr['FWHMASEC'])
        if seeingfwhm < 1:
           seeingfwhm=2.5
    except:
        seeingfwhm=2.5
    photapertures=max(5.0/float(prihdr['PIXSCALE']),3)
    tempdir=file.replace('.','d')
    
    # Calculate broadly minimum area from the seeing and the pixelscale
    fwhmpixels= seeingfwhm / float(prihdr['PIXSCALE'])
    psfarea= math.pi * pow((fwhmpixels /2),2)
    minarea= 1.0 * psfarea    
    backsize= 4 * fwhmpixels

    tempprocess=subprocess.Popen(['source-extractor' , file ,'-c',os.path.expanduser(codedir) +'/photometryparams/default.sexfull','-CATALOG_NAME',str(tempdir+'/psf.cat'),'-CATALOG_TYPE','FITS_LDAC','-SATUR_LEVEL', str(saturlevel) , '-DETECT_THRESH', str(2.5), '-ANALYSIS_THRESH',str(2.5),'-BACKPHOTO_TYPE','LOCAL', '-BACK_SIZE', str(backsize), '-BACK_FILTERSIZE',str(4), '-DETECT_MINAREA', str(minarea), '-GAIN',str(gain),'-SEEING_FWHM',str(seeingfwhm),'-PHOT_APERTURES', str(photapertures),'-FILTER_NAME', str(os.path.expanduser(codedir) +'/photometryparams/sourceex_convs/gauss_2.0_5x5.conv'),'-PARAMETERS_NAME',  str(os.path.expanduser(codedir) +'/photometryparams/default.paramprepsx')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
    tempprocess.wait()

    tempprocess=subprocess.Popen(['psfex', str(tempdir)+'/psf.cat','-CHECKPLOT_DEV','NULL','-CHECKIMAGE_TYPE','NONE','-PSF_DIR',str(tempdir),'-PARAMETERS_NAME',  str(os.path.expanduser(codedir) +'/photometryparams/default.psfex')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
    tempprocess.wait()

def run_actual_psfex(file, codedir):

    prihdr = fits.open(file)[0].header
    saturlevel=prihdr['SATURATE']
    gain=prihdr['GAIN']
    photapertures=max(3.0/float(prihdr['PIXSCALE']),3)
    
    try:
        seeingfwhm=float(prihdr['FWHMASEC'])
        if seeingfwhm < 1:
           seeingfwhm=2.5
    except:
        seeingfwhm=2.5
    
    # Calculate broadly minimum area from the seeing and the pixelscale
    fwhmpixels= seeingfwhm / float(prihdr['PIXSCALE'])
    psfarea= math.pi * pow((fwhmpixels /2),2)
    minarea= 1.0 * psfarea
    backsize= 4 * fwhmpixels
    
    tempdir=file.replace('.','d')
    
    tempprocess=subprocess.Popen(['source-extractor','-PSF_NAME',str(tempdir)+'/psf.psf', file ,'-c',os.path.expanduser(codedir) +'/photometryparams/default.sexfull','-CATALOG_NAME',str(tempdir+'/psf.cat'),'-CATALOG_TYPE','ASCII', '-BACKPHOTO_TYPE','LOCAL', '-BACK_SIZE', str(backsize), '-BACK_FILTERSIZE',str(4), '-DETECT_THRESH', str(3), '-ANALYSIS_THRESH',str(3), '-DETECT_MINAREA', str(minarea),'-SATUR_LEVEL', str(saturlevel) ,'-GAIN',str(gain),'-PHOT_APERTURES', str(photapertures),'-FILTER_NAME', str(os.path.expanduser(codedir) +'/photometryparams/sourceex_convs/gauss_2.0_5x5.conv'),'-PARAMETERS_NAME',  str(os.path.expanduser(codedir) +'/photometryparams/default.paramactualpsx')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
    tempprocess.wait()
    
    shutil.copy(str(tempdir+'/psf.cat'), file.replace('EVA-','psxphot-').replace('SmSTACK-','psxphotSmSTACK-').replace('LoSTACK-','psxphotLoSTACK-').replace('.fits','.fullpsx'))

    with open(file.replace('EVA-','psxphot-').replace('SmSTACK-','psxphotSmSTACK-').replace('LoSTACK-','psxphotLoSTACK-').replace('.fits','.psx'),'w') as p:
        with open(str(tempdir+'/psf.cat'), 'r') as f:
            p.write('RA,Dec,Xpix,Ypix,Counts,Counts_err,Ra_err,Dec_err' + '\n')
            for line in f:
                if not '#' in line:
                    splitline=line.split(' ')
                    splitline=[i for i in splitline if i]
                    del splitline[0]
                    if float(splitline[4]) > 0 and float(splitline[5]) > 0:                        
                        
                        # First check it rises above S/N = 30
                        if float(splitline[4])/float(splitline[5]) > 10:
                            # Check that is plausibly sized
                            if float(splitline[24]) > 0.5*fwhmpixels:
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
                                del splitline[-1]
        
                                splitline=','.join(splitline)
                                p.write(splitline + '\n')
    try:
        shutil.rmtree(tempdir)
        shutil.rmtree('outputdirectory/' + tempdir)
    except:
        pass
    

