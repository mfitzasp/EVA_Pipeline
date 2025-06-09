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
import shutil
import json
import traceback
import logging

def archive_preparer(file, largedataset_output_folder, shortexposure_output_folder, ingestion_output_folder, local_output_folder, local_copy, ingester_copy):

    try:
        queue_add=[]
    
        tempheader = fits.open(file)
        if ('BZESK' in file) or ('.fits.fz' in file):
            tempheader = tempheader[1].header
        else:
            tempheader = tempheader[0].header
        headerdict = {}
        for entry in tempheader.keys():
            if not 'COMMENT' in entry:
                headerdict[entry] = tempheader[entry]
                
        if local_copy:
    
            dayobs=str(tempheader['DAY-OBS'])
            origname=str(tempheader['ORIGNAME'])
            try:
                seq_number='_' + str(origname.split('-')[3])
            except:
                seq_number='_unknown'
        
            if not os.path.exists(local_output_folder):
                os.makedirs(local_output_folder , mode=0o777)
        
            if not os.path.exists(local_output_folder + '/' + dayobs ):
                os.makedirs(local_output_folder + '/' + dayobs, mode=0o777)
        
            # Copy the fits over
            if not os.path.exists(local_output_folder + '/' + dayobs + '/fits'):
                os.makedirs(local_output_folder + '/' + dayobs + '/fits', mode=0o777)
            shutil.copy(file, local_output_folder + '/' + dayobs + '/fits/' + file.split('/')[-1].replace('.fits',seq_number+'.fits'))
    
        # Need to replace this hack by number of images requested by LCO. LCO does not have nexpreq header
        try:    
            tempnreq=tempheader['NEXPREQ']
        except:
            tempnreq=1
    
        if tempnreq > 20:
            relevant_archive_folder=largedataset_output_folder + '/' + str(tempheader['BATCHCDE'])
        elif tempheader['EXPTIME'] < 1 and not "jupiter" in str(tempheader['OBJECT']).lower() and not "saturn" in str(tempheader['OBJECT']).lower() and not "mars" in str(tempheader['OBJECT']).lower() and not "lcogt" in str(tempheader['ORIGIN']).lower():
            relevant_archive_folder=shortexposure_output_folder
        else:        
            relevant_archive_folder=ingestion_output_folder   
        
        if not os.path.exists(relevant_archive_folder):
            os.makedirs(relevant_archive_folder , mode=0o777)
    
        print (file)
    
        try:
                    # For FITS files, we are only going to upload the BZESK file to the archive, even though we do upload dataproducts from the component files    
            if ('BZESK-' in file) and "lcogt" in str(tempheader['ORIGIN']).lower():
              logging.info ("not reuploading an individual BZESK for an original LCO file: " + str(file))  
             
            elif ('EVA-' in file) and "lcogt" in str(tempheader['ORIGIN']).lower():
              logging.info ("not reuploading an individual EVA for an original LCO file: " + str(file))  
            
            elif ('BZESK' in file) and tempheader['EXPTIME'] >= 1:# or ('no' in str(tempheader['SMARTSTK'])) or ('False' in str(tempheader['SMARTSTK'])):
                try:
                    shutil.copy(file, relevant_archive_folder) 
                    landing_filename=relevant_archive_folder +'/' + file.split('/')[-1]
                    with open(landing_filename+'.tempjson','w') as tempfile:
                        json.dump(headerdict, tempfile, indent=4)
                    #np.save(landing_filename+'.temppickle', headerdict, allow_pickle=True)
                    os.rename (landing_filename+'.tempjson', landing_filename+'.json')
                except:
                    logging.info(traceback.format_exc())
            elif ('BZESK' not in file) and tempheader['EXPTIME'] < 1:
                try:
                    shutil.copy(file, relevant_archive_folder) 
                    landing_filename=relevant_archive_folder +'/' + file.split('/')[-1]
                    with open(landing_filename+'.tempjson','w') as tempfile:
                        json.dump(headerdict, tempfile, indent=4)
                    #np.save(landing_filename+'.temppickle', headerdict, allow_pickle=True)
                    os.rename (landing_filename+'.tempjson', landing_filename+'.json')
                except:
                    logging.info(traceback.format_exc())
                
                                 
        except:
            logging.info ("FAILED BSK ARCHIVE PREP")
            logging.info (str(tempheader['SMARTSTK']))
    
        # Upload source extractor aperture (sek) photometry
        tempfilename = file.replace('outputdirectory', 'photometry').replace('.fits', '.sek').replace('EVA-',
                                                                                                      'sekphot-').replace(
            'SmSTACK-', 'sekphotSmSTACK-').replace('LoSTACK-', 'sekphotLoSTACK-')
                                                                                                      
        if os.path.exists(tempfilename):
            
            if local_copy:
                if not os.path.exists(local_output_folder + '/' + dayobs + '/photometry'):
                    os.makedirs(local_output_folder + '/' + dayobs + '/photometry', mode=0o777)
                shutil.copy(tempfilename, local_output_folder + '/' + dayobs + '/photometry/' + tempfilename.split('/')[-1].replace('.sek',seq_number+'.sek'))
            
            headerdict['RLEVEL'] = 82
            try:
                if os.path.exists(tempfilename) and os.path.getsize(tempfilename) > 0:
                    shutil.copy(tempfilename, relevant_archive_folder)        
                    landing_filename=relevant_archive_folder +'/' + tempfilename.split('/')[-1]
                    with open(landing_filename+'.tempjson','w') as tempfile:
                        json.dump(headerdict, tempfile, indent=4)
                    #np.save(landing_filename+'.temppickle', headerdict, allow_pickle=True)
                    os.rename (landing_filename+'.tempjson', landing_filename+'.json')
            except:
                logging.info(traceback.format_exc())
    
        # Upload source extractor aperture (psx) photometry
        tempfilename = file.replace('outputdirectory', 'photometry').replace('.fits', '.psx').replace('EVA-',
                                                                                                      'psxphot-').replace(
            'SmSTACK-', 'psxphotSmSTACK-').replace('LoSTACK-', 'psxphotLoSTACK-')
        if os.path.exists(tempfilename):
            
            if local_copy:
                if not os.path.exists(local_output_folder + '/' + dayobs + '/photometry'):
                    os.makedirs(local_output_folder + '/' + dayobs + '/photometry', mode=0o777)
                shutil.copy(tempfilename, local_output_folder + '/' + dayobs + '/photometry/' + tempfilename.split('/')[-1].replace('.psx',seq_number+'.psx'))
            
            headerdict['RLEVEL'] = 83
            try:
                if os.path.exists(tempfilename) and os.path.getsize(tempfilename) > 0:
                    shutil.copy(tempfilename, relevant_archive_folder)        
                    landing_filename=relevant_archive_folder +'/' + tempfilename.split('/')[-1]
                    with open(landing_filename+'.tempjson','w') as tempfile:
                        json.dump(headerdict, tempfile, indent=4)
                    #np.save(landing_filename+'.temppickle', headerdict, allow_pickle=True)
                    os.rename (landing_filename+'.tempjson', landing_filename+'.json')
            except:
                logging.info(traceback.format_exc())
    
        
        # Upload png
        tempfilename = file.replace('outputdirectory', 'previews').replace('.fits', '.png').replace('EVA-',
                                                                                                    'png-').replace(
            'SmSTACK-', 'pngSmSTACK-')
                                                                                                        
        if os.path.exists(tempfilename):
            
            if local_copy:
                if not os.path.exists(local_output_folder + '/' + dayobs + '/pngs'):
                    os.makedirs(local_output_folder + '/' + dayobs + '/pngs', mode=0o777)
                shutil.copy(tempfilename, local_output_folder + '/' + dayobs + '/pngs/' + tempfilename.split('/')[-1].replace('.png',seq_number+'.png'))
            
            headerdict['RLEVEL'] = 75
            try:
                shutil.copy(tempfilename, relevant_archive_folder)        
                landing_filename=relevant_archive_folder +'/' + tempfilename.split('/')[-1]
                with open(landing_filename+'.tempjson','w') as tempfile:
                    json.dump(headerdict, tempfile, indent=4)
                os.rename (landing_filename+'.tempjson', landing_filename+'.json')
            except:
                logging.info(traceback.format_exc())       
    
        # Upload small jpeg
        tempfilename = file.replace('outputdirectory', 'smalljpgs').replace('.fits', '.jpg').replace('EVA-',
                                                                                                     'smalljpg-').replace(
            'SmSTACK-', 'smalljpgSmSTACK-')
                                                                                                         
        if os.path.exists(tempfilename):
            
            if local_copy:
                if not os.path.exists(local_output_folder + '/' + dayobs + '/previews'):
                    os.makedirs(local_output_folder + '/' + dayobs + '/previews', mode=0o777)
                shutil.copy(tempfilename, local_output_folder + '/' + dayobs + '/previews/' + tempfilename.split('/')[-1].replace('.jpg',seq_number+'.jpg'))
            
            headerdict['RLEVEL'] = 74
            try:
                shutil.copy(tempfilename, relevant_archive_folder)        
                landing_filename=relevant_archive_folder +'/' + tempfilename.split('/')[-1]
                with open(landing_filename+'.tempjson','w') as tempfile:
                    json.dump(headerdict, tempfile, indent=4)
                os.rename (landing_filename+'.tempjson', landing_filename+'.json')
            except:
                logging.info(traceback.format_exc())
       
    
        # Upload thumbnail jpeg
        tempfilename = file.replace('outputdirectory', 'thumbnails').replace('.fits', '.jpg').replace('EVA-',
                                                                                                      'thumbnail-').replace(
            'SmSTACK-', 'thumbnailSmSTACK-').replace('LoSTACK-', 'thumbnailLoSTACK-')
        if os.path.exists(tempfilename):
            
            if local_copy:
                if not os.path.exists(local_output_folder + '/' + dayobs + '/thumbnails'):
                    os.makedirs(local_output_folder + '/' + dayobs + '/thumbnails', mode=0o777)
                shutil.copy(tempfilename, local_output_folder + '/' + dayobs + '/thumbnails/' + tempfilename.split('/')[-1].replace('.jpg',seq_number+'.jpg'))
            
            headerdict['RLEVEL'] = 73
            try:
                shutil.copy(tempfilename, relevant_archive_folder)        
                landing_filename=relevant_archive_folder +'/' + tempfilename.split('/')[-1]
                with open(landing_filename+'.tempjson','w') as tempfile:
                    json.dump(headerdict, tempfile, indent=4)
                os.rename (landing_filename+'.tempjson', landing_filename+'.json')
            except:
                logging.info(traceback.format_exc())
                
        return queue_add
    
    except:
        logging.info("Failed to archive file: " + str(file))
