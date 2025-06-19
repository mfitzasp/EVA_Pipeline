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

import copy
from astropy import coordinates as coord
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
import time
import logging

while True:
    try:
        from barycorrpy import utc_tdb
        from astropy.utils.iers import conf as iers_conf
        iers_conf.iers_auto_url = 'https://astroconda.org/aux/astropy_mirror/iers_a_1/finals2000A.all'
        iers_conf.auto_max_age = None
    except:
        logging.info ("barycorrpy failed at import. Waiting a moment....")
        time.sleep(30)
        continue
    else:
        break

def freshen_header(header):
    
    keyKeep=['SIMPLE','NAXIS', 'NAXIS1', 'NAXIS2', 'BITPIX','BUNIT','CCDXPIXE','CCDYPIXE','EXPTIME','EXPOSURE','EFFEXPT','EFFEXPN','EXPREQ','REQTIME','EXPREQSE','COOLERON','CCDSTEMP','CCDATEMP','CCDHUMID','CCDPRESS','OBSTYPE','IMAGETYP','XBINING','YBINING','OBSID','TELID','PTRTEL','TELESCOP','BLKUID','DETECTOR','DETECTID','CAMNAME','CMOSCAM','OSCCAM','OCSMONO','OSCMATCH','OSCSEP','DATE','INSTRUME','CAMNAME','ORIGIN','FILTER','OBJECT','SRCTYPE','SCHEDNAM','TRACKNUM','REQNUM','MOLUID','BLKTYPE','BLKUID','BLKSDATE','BLKEDATE','BLKNOMEX','BLKMNPH','BLKMNDST','BLKSEECO','BLKTRNCO','BLKAIRCO','SCHEDSEE','SCHEDTRN','TRIGGER','MOSAIC','DITHER','OBRECIPE','PCRECIPE','PPRECIPE','RA', 'DEC','RADESYS','LST','CAT-RA','CAT-DEC','OFST-RA','OFST-DEC','TPT-RA','TPT-DEC','PM-RA','PM-DEC','PARALLAX','RADVEL','RATRACK','DECTRACK','TRACFRAC','RAHRS','RADEG','DECDEG','RA-HMS','DEC-DMS','GAIN','RDNOISE','SATLEVEL','SATURATE','FULLWELL','MAXLIN','AZIMUTH', 'ALTITUDE','AIRMASS','PIERSIDE','DAY-OBS','DATE-OBS','MJD-OBS','L1PUBDAT','BZERO','BSCALE','PEDESTAL','LONGITUD','LATITUDE','HEIGHT','OBSERVER','OPERATOR','OBSNOTE','USERID','USERNAME','PROPID','SITEID','REQNUM','FRAMENUM','SUBSTACK','SUBEXPT','SUBMIDTS','SMARTSTK','SSTKID','SSTKNUM','SSTKLEN','MJD-MID','JD','JD-START','JD-MID','BJD','TDB','BJD-TDB','BJD_TDB','BJD-MID','TDB-MID','HJD','TELSTATE','ENGSTATE','TCSSTATE','TCSVER','TPNTMODL','UT1-UTC','POLARMOX','POLARMOY','EOPSRC','ROLLERDR','ROLLERND','AZDMD','AZSTAT','ALTDMD','ALTSTAT','ROTTYPE','ROTMODE','ROTDMD','ROTANGLE','ROTSKYPA','ROTSTAT','AMSTART','AMEND','ENC1STAT','ENC2STAT','ENCAZ','ENCWLIGT','ENCRLIGT','FSTATION','FOLDSTAT','FOLDPORT','FOLDPOSN','M1COVER','M1HRTMN','FOCDMD','FOCPOSN', 'FOCUSPOS','FOCUSTMP','FOCTELZP','FOCINOFF','FOCTOFF','FOCZOFF','FOCAFOFF','FOCOBOFF','FOCFLOFF','FOCSTAT','M2PITCH','M2ROLL','AUXROLL','AUXPITCH','WCSAXES','CTYPE1','CTYPE2','EQUINOX','LONPOLE','LATPOLE','CRVAL1','CRVAL2','CRPIX1','CRPIX2','CUNIT1','CUNIT2','MJDREF','CROTA1','CROTA2','CD1_1','CD1_2','CD2_1','CD2_2','IMAGEW','IMAGEH','A_ORDER','A_0_0','A_0_1','A_0_2','A_1_0','A_1_1','A_2_0','B_ORDER','B_0_0','B_0_1','B_0_2','B_1_0','B_1_1','B_2_0','AP_ORDER','AP_0_0','AP_0_1','AP_0_2','AP_1_0','AP_1_1','AP_2_0','BP_ORDER','BP_0_0','BP_0_1','BP_0_2','BP_1_0','BP_1_1','BP_2_0','A_0_3','A_0_4','A_1_2','A_1_3','A_2_1','A_2_2','A_3_0','A_3_1','A_4_0','B_0_3','B_0_4','B_1_2','B_1_3','B_2_1','B_2_2','B_3_0','B_3_1','B_4_0','AP_0_3','AP_0_4','AP_1_2','AP_1_3','AP_2_1','AP_2_2','AP_3_0','AP_3_1','AP_4_0','BP_0_3','BP_0_4','BP_1_2','BP_1_3','BP_2_1','BP_2_2','BP_3_0','BP_3_1','BP_4_0','PC1_1', 'PC1_2','PC2_1','PC2_2','CDELT1','CDELT2','PV1_1','PV1_2','PV1_3','PV1_4','PV1_5','PV1_6','PV1_7','PV1_8','PV1_9','PV2_0','PV2_1','PV2_2','PV2_3','PV2_4','PV2_5','PV2_6','PV2_7','PV2_8','PV2_9','GROUPID','MOSAIC','PANE','PIXSCALE','DRZPIXSC','DOCOSMIC','DOSNP','DODBND','ASRA','SKYLEVEL','FWHM','FWHMPIX','FWHMASEC','FWHMSTD','NSTARS','ASDEC','ORIGRA','ORIGDEC','ORIGNAME','WMSSTATE','WMSHUMID','WMSTEMP','WMSPRES','WINDSPEE','WINDDIR','WMSRAIN','WMSMOIST','WMSDEWPT','WMSCLOUD','WMSSKYBR','SKYMAG','TUBETEMP','M1TEMP','FOCTEMP','ISSTEMP','REFPRES','REFTEMP','REFHUMID','AGSTATE','AGCAM','AGLCKFRC','AGMODE','AGRA','AGDEC','AGGMAG','AGFWHM','AGNSRC','AGDX','AGDY','AGMIRDMD','AGMIRPOS','AGMIRST','AGFOCDMD','AGFOCUS','AGFOCOFF','AGFOCST','AGFILTER','AGFILTID','AGFILST','MOONSTAT','MOONFRAC','MOONDIST','MOONALT','SUNDIST','SUNALT','EXTNAME','EXTVER','CHECKSUM','DATASUM','L1IDMASK','L1IDRDN','SATFRAC','L1STATOV','OVERSCAN','OVERSCN1','BIASLVL','L1IDBIAS','L1STATBI','L1IDDARK','L1STATDA','DRKTSCAL','L1IDFLAT','L1STATFL','L1MEAN','L1MEDIAN','L1SIGMA','L1FWHM','L1FWTM','L1ELLIP','L1ELLIPA','WCSERR','PNTOFST','RLEVEL','PIPEVER','L1PUBDAT','ZHECKSUM','ZDATASUM']

    keylist=(list(header.keys()))
    counter=0
    for entry in keylist:
        if not entry in keyKeep:
            commentline= ('OLDHEADER '+str(keylist[counter])+' '+str(header[entry]))
            header.remove(entry)
            try:
                header.set('COMMENT', commentline)
            except:
                pass
        counter=counter+1
    return header


def value_add_header(header, telescope):
    """
    Process and standardize a FITS header.

    This function updates and corrects various fields in the header of an astronomical FITS file:
    - Sets default values for missing fields (e.g., `PEDESTAL`, `EXPTIME`, `USERID`, etc.)
    - Converts RA and DEC to decimal degrees.
    - Standardizes filter names using a mapping dictionary.
    - Computes mid-exposure time, Julian Date (JD), and Barycentric Julian Date (BJD).
    - Fixes site information (longitude, latitude, elevation).
    - Cleans object and instrument names.
    - Handles instrument-specific fields like `SMARTSTK`.
    
    Args:
        header (dict): FITS header to be updated.
    
    Returns:
        dict: Updated FITS header.
    """

    header['BZERO']=0
    header['BSCALE']=1
    if not any("PEDESTAL" in s for s in header.keys()):
        header['PEDESTAL']=0
    if (header['AIRMASS'] < 1):
        header['AIRMASS']=99.999
    elif (header['AIRMASS'] > 5):
        header['AIRMASS']=99.999
    if not any("EXPTIME" in s for s in header.keys()):
        header['EXPTIME']=header['EXPOSURE']
    if not any("EXPTIME" in s for s in header.keys()):
        header['EXPOSURE']=header['EXPTIME']
    if not any("USERID" in s for s in header.keys()):
        header['USERID']='Unknown'
    # Check that object field has a name
    if any("OBJECT" in s for s in header.keys()):
        if (header['OBJECT']==''):
            #logging.info "uneiwo"
            header['OBJECT']='Unspecified'
    if not any("OBJECT" in s for s in header.keys()):
        header['OBJECT']='Unspecified'

    if not any("SUBSTACK" in s for s in header.keys()):
        header['SUBSTACK']='F'

    ## Worst comes to worst... Fake it!
    if not any("GAIN" in s for s in header.keys()):
        header['GAIN']=1.0

    if not any("RDNOISE" in s for s in header.keys()):
        header['RDNOISE']=10.0

    header['OBJECT']=str(header['OBJECT'].replace('(','').replace(')',''))
    header['OBJECT']=str(header['OBJECT'].replace(' ',''))
    header['OBJECT']=str(header['OBJECT'].replace('1of2','').replace('2of2','').replace('1of3','').replace('2of3','').replace('3of3',''))

    # Get rid of dodgy bits of camera name
    header['INSTRUME']=str(header['INSTRUME'].replace('(1)','').replace('(','').replace(')',''))

    if not any("GROUPID" in s for s in header.keys()):
        header['GROUPID']='None'
    if not any("PROPID" in s for s in header.keys()):
        header['PROPID']='UNKNOWN'

    if any("DATE-OBS" in s for s in header.keys()):
        header['DATE'] = ( header['DATE-OBS'], 'Date of Start of Observation')

    # If we have RA or DEC in HMS or DMS, then bung it in those variables.
    if ':' in header['RA'] or 'm' in header['RA']:
        header['RA-HMS']=copy.deepcopy(header['RA'])
    if ':' in header['DEC'] or 'm' in header['DEC']:
        header['DEC-DMS']=copy.deepcopy(header['DEC'])

    # Get RA into decimal degrees
    header['RA']=str(header['RA']).replace(' ',':')
    if ':' in str(header['RA']):
        pointing = coord.SkyCoord(header['RA'], header['DEC'], unit=(u.hourangle, u.deg))
    else:
        pointing = coord.SkyCoord(header['RA'], header['DEC'], unit=(u.deg, u.deg))
    header['RA']=pointing.ra.degree
    header['DEC']=pointing.dec.degree

    header['RAHRS']=float(pointing.ra.degree) / 15

    if not any("LONGITUD" in s for s in header.keys()):
        if any("LONG-OBS" in s for s in header.keys()):
            header['LONGITUD']=float(header['LONG-OBS'])
        if any("SITELONG" in s for s in header.keys()):
            tempHold=header['SITELONG']
            if "W" in tempHold:
               tempHold=('-'+tempHold).replace('W','')
               tempHold=str(tempHold).replace('d','X').replace('m','X').replace('s','').replace(' ','').split('X')
               decLong=float(tempHold[0])+(float(tempHold[1])/60)+(float(tempHold[2])/3600)
            elif "E" in tempHold:
               tempHold=(tempHold).replace('E','')
               tempHold=str(tempHold).replace('d','X').replace('m','X').replace('s','').replace(' ','').split('X')
               decLong=float(tempHold[0])+(float(tempHold[1])/60)+(float(tempHold[2])/3600)
            else:
                decLong=header['SITELONG']

            header['LONGITUD']=decLong

    if not any("LATITUDE" in s for s in header.keys()):
        if any("LAT-OBS" in s for s in header.keys()):
            header['LATITUDE']=float(header['LAT-OBS'])
        if any("SITELAT" in s for s in header.keys()):
            tempHold=header['SITELAT']
            if "S" in tempHold:
               tempHold=('-'+tempHold).replace('S','')
               tempHold=str(tempHold).replace('d','X').replace('m','X').replace('s','').replace(' ','').split('X')
               decLat=float(tempHold[0])+(float(tempHold[1])/60)+(float(tempHold[2])/3600)
            elif "N" in tempHold:
               tempHold=(tempHold).replace('N','')
               tempHold=str(tempHold).replace('d','X').replace('m','X').replace('s','').replace(' ','').split('X')
               decLat=float(tempHold[0])+(float(tempHold[1])/60)+(float(tempHold[2])/3600)
            else:
               decLat=header['SITELAT']

            header['LATITUDE']=decLat
           
    if not any("ORIGIN" in s for s in header.keys()):
        header['ORIGIN'] = 'Unknown'

    # When the data comes from LCOGT, standardise the observation ID early
    if header.get('ORIGIN', '').lower() == 'lcogt':
        header['OBSID'] = 'LCO'
    
    if not any("HEIGHT" in s for s in header.keys()):
        if any("ALT-OBS" in s for s in header.keys()):
            header['HEIGHT']=float(header['ALT-OBS'])
        if any("SITEELEV" in s for s in header.keys()):
            header['HEIGHT']=header['SITEELEV']

    # Fix up longitude and latitude
    if (set(['LONGITUD']).intersection(set(header.keys())) == set(['LONGITUD'])):
        header['LONGITUD']=(str(header['LONGITUD'])).replace(' ',':')
        if ':' in header['LONGITUD']:
            header['LONGITUD'] = sum(float(x) / 60 ** n for n, x in enumerate(header['LONGITUD'][:-1].split(':')))
    if (set(['LATITUDE']).intersection(set(header.keys())) == set(['LATITUDE'])):
        header['LATITUDE']=(str(header['LATITUDE'])).replace(' ',':')
        if ':' in header['LATITUDE']:
            header['LATITUDE'] = sum(float(x) / 60 ** n for n, x in enumerate(header['LATITUDE'][:-1].split(':')))

    if not any("LONGITUD" in s for s in header.keys()):
        header['LONGITUD']='UNKNOWN'

    if not any("LATITUDE" in s for s in header.keys()):
        header['LATITUDE']='UNKNOWN'

    if not any("HEIGHT" in s for s in header.keys()):
        header['HEIGHT']='UNKNOWN'

    #GET AN MJD IN THERE
    if any("MJD-OBS" in s for s in header.keys()):
        mJD = float(header['MJD-OBS'])
        header['MJD-OBS'] = mJD
    elif any("MJD" in s for s in header.keys()):
        mJD = float(header['MJD'])
        header['MJD-OBS'] = mJD
    elif any("TSTART" in s for s in header.keys()):
        bJD = (float(header['TSTART'])+float(header['TSTOP']))/2
        logging.info (bJD)
        header['BJD'] = bJD
        header['TDB'] = bJD
    elif any("JD" in s for s in header.keys()):
        mJD = float(header['JD'] - 2400000.5)
        header['MJD-OBS'] = mJD
        header['MJD'] = mJD
    elif any("DATE-OBS" in s for s in header.keys()):
        dateconv=Time(header['DATE-OBS'],format='isot',scale='utc')
        header['MJD-OBS'] = dateconv.mjd

    # Calculate mid-exposure time
    if any("MJD-OBS" in s for s in header.keys()):
        mJD = float(header['MJD-OBS'])
        expTimeCorr=float(header['EXPTIME'])/172800.0
        header['MJD-MID'] = mJD + expTimeCorr

        t = Time(mJD, format='mjd',scale='utc')

        header['JD']=t.jd
        header['JD-MID']=header['JD']+expTimeCorr

    ## HJD and BJD
    if (header['RA'] == 'UNKNOWN'):
        header['BJD']='UNKNOWN'
        header['TDB']='UNKNOWN'
        header['BJD_TDB']='UNKNOWN'
        header['HJD']='UNKNOWN'
        header['BJD-MID']='UNKNOWN'
        header['BJD-TDB']='UNKNOWN'
        header['TDB-MID']='UNKNOWN'

    elif any("MJD-OBS" in s for s in header.keys()):

        if (header['RA'] != 'UNKNOWN'):
            if ':' in str(header['RA']):
                pointing = coord.SkyCoord(header['RA'], header['DEC'], unit=(u.hourangle, u.deg))
            else:
                pointing = coord.SkyCoord(header['RA'], header['DEC'], unit=(u.deg, u.deg))
        location = EarthLocation.from_geodetic(header['LONGITUD'], header['LATITUDE'], header['HEIGHT'])

        t = Time(mJD, format='mjd', scale='utc', location=location)


        tdbholder= (utc_tdb.JDUTC_to_BJDTDB(t.jd, ra=float(pointing.ra.degree), dec=float(pointing.dec.degree), lat=float(header['LATITUDE']), longi=float(header['LONGITUD']), alt=float(header['HEIGHT']), leap_update=True))

        header['BJD'] = (tdbholder[0][0], 'BJD-TDB at start of exposure')
        header['TDB'] = (tdbholder[0][0], 'BJD-TDB at start of exposure')
        header['BJD_TDB'] = (tdbholder[0][0], 'BJD-TDB at start of exposure')
        header['BJD-TDB'] = (tdbholder[0][0], 'BJD-TDB at start of exposure')
        header['BJD-MID'] = (tdbholder[0][0]+expTimeCorr, 'BJD-TDB at middle of exposure')
        header['TDB-MID'] = (tdbholder[0][0]+expTimeCorr, 'BJD-TDB at middle of exposure')


    #Standardise filter headers
    if any("FILTER" in s for s in header.keys()):

        filter_map = {
        'H-Alpha': 'ha',
        'OIII': 'o3',
        'SII': 's2',
        'I - Sloan': 'ip', 'I-Sloan': 'ip', 'I - sloan': 'ip', 'Isloan': 'ip',
        'Luminance': 'L', 'Lum': 'L',
        "u'": 'up', "g'": 'gp', "r'": 'rp', "i'": 'ip', "z'": 'zs',
        'gprime': 'gp', 'rprime': 'rp', 'iprime': 'ip', 'zprime': 'zs', 'uprime': 'up',
        'Z2sloan': 'zs', 'Z2 - sloan': 'zs', 'R2 - sloan': 'rp', 'G2 - sloan': 'gp',
        'R2sloan': 'rp', 'G2sloan': 'gp',
        'Ic': 'I', 'B - JC': 'B', 'Ic - JC': 'I', 'UV - JC': 'up', 'V - JC': 'V', 'UV': 'up'
        }
    
        header['FILTER'] = filter_map.get(header['FILTER'], header['FILTER'])
        
        if any("SITENAME" in s for s in header.keys()):
            if (header['SITENAME'] == 'BARO'):
                if (header['FILTER'] == 'UV'):
                    header['FILTER'] = 'up'

    # Just check NAXIS is correct... sometimes it oddly isn't....
    header['NAXIS'] = 2

    # Preserve any original SMARTSTK identifier for reference
    if 'SMARTSTK' in header and str(header['SMARTSTK']).lower() not in ['no', 'false', '0', 'none', '']:
        header['SSTKID'] = header['SMARTSTK']

    # For OSCs we need to break up the smartstacks into their respective individual bands
    # So we incorporate the bayer filter code into the smartstack code
    if telescope != 'lco':
        if header['OSCCAM'] == True and not header['SMARTSTK'] == 'no':
            header['SMARTSTK']=header['SMARTSTK'] + str(header['FILTER']).split('_')[-1]
            logging.info (header['FILTER'])
            logging.info (header['SMARTSTK'])
    else:
        header['SMARTSTK']='lco'+header['REQNUM'] + str(header['FILTER']).split('_')[-1]
        logging.info (header['FILTER'])
        logging.info (header['SMARTSTK'])
        
        # include a couple of header items not included in lco
        header['EXPREQ'] =header['REQTIME'] 
        header['NEXPREQ'] = 1 # This actually needs to be replacement so we don't re-ingest LCO fits files from exoplanet runs... but we don't ingest EVA for LCO anyway. 

    # Put the comments into the header
    header.comments['BJD'] = 'BJD-TDB at start of exposure'
    header.comments['TDB'] = 'BJD-TDB at start of exposure'
    header.comments['BJD_TDB'] =  'BJD-TDB at start of exposure'
    header.comments['BJD-TDB'] = 'BJD-TDB at start of exposure'
    header.comments['BJD-MID'] = 'BJD-TDB at middle of exposure'
    header.comments['TDB-MID'] = 'BJD-TDB at middle of exposure'

    return header


def human_namer(header):
    """
    Generate a standardized filename from a FITS header.

    This function processes the metadata in a FITS header to create a human-readable and 
    standardized filename. It sanitizes and replaces characters in key fields, formats 
    RA and DEC values, and structures the filename based on the observation details.

    Key operations:
    - Cleans and replaces unwanted characters in `OBJECT`, `TELESCOP`, `FILTER`, and other fields.
    - Handles missing or unspecified values by assigning defaults.
    - Formats `RA` and `DEC` values if `OBJECT` is unspecified.
    - Builds a filename in the format:
        ```
        EVA-<object>_<filter1>_<filter2>_<filter3>_<MJD>_<date>_<airmass>_<exptime>_<instrument>_<userid>_<telescope>.fits
        ```
    - Replaces specific patterns using a mapping dictionary for consistency.
    - Assigns a processing level of 95 (`RLEVEL`).

    Args:
        header (dict): FITS header containing metadata fields.

    Returns:
        str: Generated filename based on header data.
    """


    headerkeys = header.keys()
    if any("OBJECT" in s for s in headerkeys):
        objectTemp=header['OBJECT'].replace('-','d').replace('+','p').replace('.','d').replace(' ','').replace('_','').replace('=','e').replace('(','').replace(')','').replace('<','').replace('>','').replace('/','').replace(':','c').replace('*','star')
    else:
        objectTemp="UNKNOWN"

    try:
        telescopTemp=header['TELESCOP'].replace('-','d').replace('+','p').replace('.','d').replace(' ','').replace('_','').replace('=','e').replace('(','').replace(')','').replace('<','').replace('>','').replace('/','').replace(':','c').replace('*','star')
    except:
        telescopTemp="UNKNOWN"

    if any("PTRTEL" in s for s in headerkeys):
        telescopTemp=header['TELID']

    if any("IMAGETYP" in s for s in headerkeys):
        if (header['IMAGETYP'] == 'Dark Frame'):
            objectTemp='DARK'

    if any("IMAGETYP" in s for s in headerkeys):
        if (header['IMAGETYP'] == 'Bias Frame'):
            objectTemp='BIAS'

    if any("IMAGETYP" in s for s in headerkeys):
        if (header['IMAGETYP'] == 'Flat Field'):
            objectTemp='FLAT'

    filterTwo='NOTPRESENT'
    filterThree='NOTPRESENT'

    if 'FILTER1' in header:
        filterOne=(header['FILTER1']).replace('_','')
        filterTwo=(header['FILTER2']).replace('_','')
        filterThree=(header['FILTER3']).replace('_','')
    elif any("FILTER" in s for s in headerkeys):

        filterOne=(header['FILTER'].replace('_','').replace('SDSS-','').replace('-011','').replace('RP','rp').replace('IP','ip').replace('-','d').replace('JC','').replace('Isloan','ip').replace('ISloan','ip').replace("(",'').replace(")",''))
        filterTwo='NOTPRESENT'
        filterThree='NOTPRESENT'

    elif any("SITENAME" in s for s in headerkeys):
        if header['SITENAME'] == 'BARO':
            filterOne=header['FILTER'].replace('_','').replace(' ','').replace('/','').replace('-','').replace('JC','').replace('Isloan','ip').replace('ISloan','ip')

    else:
        filterOne=header['FILTER'].replace('_','').replace(' ','').replace('/','').replace('-','')


    expTime=(str("{:.4f}").format(header['EXPTIME'])).replace('.','d')
    dateObs=(header['DATE'].replace('-','d').replace(':','d').replace('.','d'))
    airMass=(str("{:.4f}").format(float(header['AIRMASS']))).replace('.','a')
    instruMe=(header['INSTRUME']).replace(' ','').replace('/','').replace('-','')
    userId=(header['USERID']).replace('googledoauth','').replace('@gmail.com','').replace('@sacredheart.wa.edu.au','').replace(' ','').replace('.','d').replace('_','').replace('-','d').replace(':','d').replace('.com','').replace('@','at').replace('.','d').replace('|','').replace('ossatncc2dvicdedudau','').replace('+','')


    if header['OBJECT'].lower() == "unspecified" or header['OBJECT'] == "empty" or ("ra" in header['OBJECT'] and "dec" in header['OBJECT']):
        RAtemp = float(header['RA']) / 15
        DECtemp = float(header['DEC'])
        RAstring = f'{RAtemp:.1f}'.replace('.','h')
        DECstring = f'{DECtemp:.1f}'.replace('-','n').replace('.','d')
        #RAstring = RAtemp.replace('.','h').replace(':','')
        #DECstring = DECtemp.replace('-','n').replace('.','d').replace(':','')
        header['OBJECT'] = RAstring + "ra" + DECstring + "dec"
        objectTemp = header['OBJECT']
        header['OBJSPECF']= "no"

    if (header['BJD'] == 'UNKNOWN'):
        newName='EVA-'+objectTemp + "_" + filterOne + "_" +filterTwo+ "_" + filterThree + "_" + "_UNKNOWN_" + "_" + dateObs + "_" + airMass + "_"+ expTime + "_"+ instruMe + "_" + userId + "_"+ telescopTemp +".fits"
    else:
        mjdObs = '{0:.8f}'.format(header['BJD-MID'])
        newName='EVA-'+objectTemp + "_" + filterOne + "_" +filterTwo+ "_" + filterThree + "_" + str(mjdObs).replace('.','d') + "_" + dateObs + "_" + airMass + "_" + expTime + "_" + instruMe + "_" + userId + "_" + telescopTemp +".fits"

    newName=newName.replace(' ','').replace("(",'').replace(")",'')

    replacements = {
        '_NOTPRESENT_NOTPRESENT_': '_',
        '_NP_NP_': '_',
        '_air_air_air_': '_air_',
        '_air_air_V_': '_V_',
        '_air_air_B_': '_B_',
        '_air_air_R_': '_R_',
        '_air_air_I_': '_I_',
        '_air_air_U_': '_U_',
        '_air_V_air_': '_V_',
        '_air_B_air_': '_B_',
        '_air_R_air_': '_R_',
        '_air_I_air_': '_I_',
        '_air_U_air_': '_U_',
        '_SDSSdU_air_air_': '_up_',
        '_air_SDSSdU_air_': '_up_',
        '_SDSSdG_air_air_': '_gp_',
        '_air_SDSSdG_air_': '_gp_',
        '_SDSSdR_air_air_': '_rp_',
        '_air_SDSSdR_air_': '_rp_',
        '_SDSSdI_air_air_': '_ip_',
        '_air_SDSSdI_air_': '_ip_',
        '_PandStarrsdZ_air_air_': '_zs_',
        '_air_PandStarrsdZ_air_': '_zs_',
        '_air_w_air_': '_w_',
        '_air_ip_air_': '_ip_',
        '_air_gp_air_': '_gp_',
        '_air_rp_air_': '_rp_',
        '_air_up_air_': '_up_',
        '_air_zs_air_': '_zs_',
        '_V_V_V': '_V',
        '_B_B_B': '_B',
        '_R_R_R': '_R',
        '_U_U_U': '_U',
        '_I_I_I': '_I',
        '_u_u_u': '_up',
        '_g_g_g': '_gp',
        '_r_r_r': '_rp',
        '_i_i_i': '_ip',
        '_z_z_z': '_zs',
        '_HdAlpha_air_air': '_HAlpha',
        '_air_air_HdAlpha_': '_HAlpha_',
        '_air_air_OIII_': '_OIII_',
        '_OIII_air_air_': '_OIII_',
    }
    
    for old, new in replacements.items():
        newName = newName.replace(old, new)


    header['RLEVEL'] = 95

    return newName