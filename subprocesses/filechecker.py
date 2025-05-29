from astropy.io import fits
import sys

file =  sys.argv[1]

with fits.open(file, ignore_missing_end=True) as hdul:
    try:
        if '.fits.fz' in file:
            tempo=hdul[1].data
            temph=hdul[1].header
        else:
            tempo=hdul[0].data
            temph=hdul[0].header
        
        del tempo
        hdul.close()
        del hdul
        print ("safely opened file: " + file)
    except:
        try:            
            #os.remove(file)
            print ("Corrupt File: " + file)
        except:
            print ("Corrupt and couldn't delete " + file)