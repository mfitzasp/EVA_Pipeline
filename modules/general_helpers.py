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
import random
import time
import psutil
import subprocess
from datetime import datetime, timedelta
import re
import os
from pathlib import Path
import shutil
import sys
import traceback
import urllib.request
from astropy.io import fits
from astropy import coordinates as coord
import numpy as np
import copy
import astropy.units as u
import json
import platform
import logging

# def load_config(pipeid, local_or_online):
#     with open(f"configs/{pipeid}.json") as f:
#         cfg = json.load(f)

#     if platform.system() == "Linux" and  os.path.exists("~/.bash_profile"):
#         # inject environment variables from bash_profile
#         with open(os.path.expanduser("~/.bash_profile")) as f:
#             for line in f:
#                 if line.startswith("export "):
#                     k, v = line[len("export "):].strip().split("=", 1)
#                     os.environ[k] = v

#     return {
#         **cfg,
#         "homedir": os.path.expanduser("~"),
#         # …
#     }

def load_config(pipeid, local_or_online):
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path  = os.path.join(project_root, "configs", f"{pipeid}.json")
    
    # with open(f"configs/{pipeid}.json") as f:
    #     cfg = json.load(f)
    with open(config_path) as f:
        cfg = json.load(f)

    if platform.system() == "Linux" and  os.path.exists("~/.bash_profile"):
        # inject environment variables from bash_profile
        with open(os.path.expanduser("~/.bash_profile")) as f:
            for line in f:
                if line.startswith("export "):
                    k, v = line[len("export "):].strip().split("=", 1)
                    os.environ[k] = v

    return {
        **cfg,
        "homedir": os.path.expanduser("~"),
        # …
    }

def cleanup_and_exit(homedir, basedirectory, original_token_file=None, remove_token=False):
    """
    Changes to the home directory, cleans up the temporary directory, 
    and optionally removes a token file.

    Parameters:
        homedir (str): Path to the home directory to switch to.
        basedirectory (str): Path to the base directory to remove.
        original_token_file (str, optional): Path to the token file to be removed.
        remove_token (bool): Whether to remove the token file (default is False).

    Returns:
        None
    """
    # Optionally remove token file
    if remove_token and original_token_file:
        try:
            os.remove(original_token_file)
            logging.info(f"Token file '{original_token_file}' removed.")
        except Exception as e:
            logging.info(f"Failed to remove token file: {e}")

    # Change to home directory
    try:
        os.chdir(homedir)
        logging.info("Directory changed to" + str( homedir))
    except Exception as e:
        logging.info(f"Failed to change directory: {e}")

    # Remove base directory
    try:
        shutil.rmtree(basedirectory)
        logging.info("Directory removed:" + str( basedirectory))
    except Exception:
        logging.info(traceback.format_exc())

    # Exit the program
    sys.exit()

def test_connect(host='http://google.com'):
    """
    Tests internet connectivity by attempting to open a connection to a specified URL.

    Parameters:
        host (str): URL to test the connection against. Defaults to 'http://google.com'.

    Returns:
        bool: 
            - True if connection to the specified host succeeds.
            - False if the connection attempt fails (e.g., due to no internet connection or unreachable host).

    Notes:
        - Uses urllib from Python's standard library.
        - Does not differentiate between types of exceptions; returns False for any connection issue.
    """
    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except:
        return False

def _is_float(value):
    """Check if value can be converted to a float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
    
def de_fz_file(infopacket, base):
    """
    Processes and converts a `.fits.fz` file into a `.npy` file by extracting image data and updating the header.

    Args:
        infopacket (tuple): A tuple containing:
            - file (str): Path to the `.fits.fz` file.
            - location (str): Location type, which could be 'local' or 'generic'.

    Returns:
        astropy.io.fits.Header or None: 
            - Returns the updated FITS header after processing.
            - Returns None if an error occurs during processing.

    Process:
        1. **Open and Verify FITS File**:
            - Opens the FITS file, attempts to fix any structural issues.
            - Extracts image data and header from the first or second extension.

        2. **Apply LCO Bad Pixel Mask (if available)**:
            - If a bad pixel mask is found, masks out bad pixels as `NaN`. This is the usual approach for LCO BANZAI images

        3. **Set Sentinel Values to NaN**:
            - Converts certain placeholder pixel values (e.g., -252) to `NaN`. This is the usual approach for Community Code images.

        4. **Update Header Keywords**:
            - Adjusts header values like `NAXIS`, `SATLEVEL`, `DOCOSMIC`, etc.
            - Converts RA and DEC to degrees if needed.

        5. **Clean Up WCS Keys**:
            - Removes problematic WCS-related header keys that may interfere with stacking.

        6. **Set Default Values for Gain and Read Noise**:
            - Assigns reasonable default values if not available in the header.

        7. **Save Processed Data**:
            - Saves the processed image as a `.npy` file using key header values in the filename.

        8. **Remove Original File**:
            - If the file is not local or generic, deletes the original FITS file.

    Exceptions:
        - Prints an error message and stack trace if an exception occurs during processing.

    Notes:
        - The function is designed to handle issues with corrupted or incomplete FITS files.
        - Header cleanup ensures compatibility with further astrometric processing.
    """

    try:

        (file,location) = infopacket
        
        # Make a backup if a reduction fails and we need to test something.
        orig_dir = Path(base) / 'originalimages'
        orig_dir.mkdir(parents=True, exist_ok=True)
        # build source path (if `file` isn’t already an absolute path)
        if location == 'local':
            src = file
        else:
            src = file if os.path.isabs(file) else os.path.join(location, file)        
        # destination path: same basename, into originalimages
        dst = os.path.join(orig_dir, os.path.basename(src))        
        # copy (use copy2 to preserve timestamps/permissions too)
        shutil.copy2(src, dst)
        
        

        if not header_is_valid(file):
            return None

        hdul = fits.open(file, ignore_missing_end=True)
        hdul.verify('fix')
        try:
            imagtemp=np.asarray(hdul[1].data)
        except:
            imagtemp=np.asarray(hdul[0].data)

        try:
            tempheader=hdul[1].header
        except:
            tempheader=hdul[0].header

        # Verify that PIXSCALE, GAIN, and RDNOISE are present and valid
        for key in ['PIXSCALE', 'GAIN', 'RDNOISE']:
            if key not in tempheader:
                logging.info(f"{file} missing {key} in header. Rejecting early.")
                return None

            if not _is_float(tempheader[key]):
                logging.info(f"{file} has invalid {key} value. Rejecting early.")
                return None

            if key in ['GAIN', 'RDNOISE'] and float(tempheader[key]) > 500:
                logging.info(
                    f"{file} {key} value {tempheader[key]} out of range. Rejecting early."
                )
                return None

        # IF LCO, then get the bpm out and apply it.
        try:
            bpm=np.asarray(hdul[3].data)
            bpm[bpm > 0] = 1
            bpm=bpm.astype('bool')
            imagtemp[bpm] = np.nan
            logging.info ("applied LCO bpm")
            
            lcoerr = Path(base) / 'lcoerrays'
            lcoerr.mkdir(parents=True, exist_ok=True)
            np.save(lcoerr / file.split('/')[-1].replace('.fits.fz','.npy'), np.asarray(hdul[4].data))

            lcobpm = Path(base) / 'lcobpms'
            lcobpm.mkdir(parents=True, exist_ok=True)
            np.save(lcobpm / file.split('/')[-1].replace('.fits.fz','.npy'), np.asarray(hdul[3].data))
            
        except:
            logging.info ("Did not apply an LCO bpm")
            
        # If community code, apply the sentinel value
        # to return nans to the array.
        imagtemp[(imagtemp > -252) & (imagtemp < -251)] = np.nan
        
        tempheader['NAXIS'] = 2
        tempheader['NAXIS1'] = imagtemp.shape[0]
        tempheader['NAXIS2'] = imagtemp.shape[1]

        tempheader['BZERO'] = 0
        tempheader['BSCALE'] = 1
        tempheader['ORIGNAME'] = str(file.split('/')[-1])
        logging.info ("de-fz-ed file" + str(file))

        if not any("SATLEVEL" in s for s in tempheader.keys()):
            if any("SATURATE" in s for s in tempheader.keys()):
                tempheader['SATLEVEL']=tempheader['SATURATE']

        if not any("SATURATE" in s for s in tempheader.keys()):
            if any("SATLEVEL" in s for s in tempheader.keys()):
                tempheader['SATURATE']=tempheader['SATLEVEL']

        if not any("DOCOSMIC" in s for s in tempheader.keys()):
            if any("PIXSCALE" in s for s in tempheader.keys()):
                if float(tempheader['PIXSCALE']) < 0.6:
                    tempheader['DOCOSMIC']=True
                else:
                    tempheader['DOCOSMIC']=False
            else:
                tempheader['DOCOSMIC']=False

        if not any("RAdeg" in s for s in tempheader.keys()) or not any("DECdeg" in s for s in tempheader.keys()):
            if ':' in str(tempheader['RA']):
                pointing = coord.SkyCoord(tempheader['RA'], tempheader['DEC'], unit=(u.hourangle, u.deg))
            else:
                pointing = coord.SkyCoord(tempheader['RA'], tempheader['DEC'], unit=(u.deg, u.deg))
            tempheader['RAdeg']=float(pointing.ra.degree)
            tempheader['DECdeg']=float(pointing.dec.degree)

        # This long list gets rid of wacky WCS keys that can stay over and mess up the WCS in the stack.
        wcskeydelete=['WCSRFCAT','WCSIMCAT','WWCSNREF','WCSMATCH','WCCATTYP','WCNTERMS','WCSRDRES','WCSDELRA','WCSDELDE','WCSERR','CTYPE1','CTYPE2','CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2','QV1_0','QV1_1','QV1_7','QV1_9','QV1_17','QV1_21','QV1_31','QV1_37','QV2_0','QV2_1','QV2_7','QV2_9','QV2_17','QV2_21','QV2_31','QV2_37','WCSAXES','EQUINOX','LONPOLE','LATPOLE','CUNIT1','CUNIT2','IMAGEW','IMAGEH','A_ORDER','A_0_0','A_0_1','A_0_2','A_1_0','A_1_1','A_1_2','A_2_0','B_ORDER','B_0_0','B_0_1','B_0_2','B_1_0','B_1_1','B_1_2','B_2_0','BP_ORDER','BP_0_0','BP_0_1','BP_0_2','BP_1_0','BP_1_1','BP_1_2','BP_2_0','AP_ORDER','AP_0_0','AP_0_1','AP_0_2','AP_1_0','AP_1_1','AP_1_2','AP_2_0','CDELT1','CDELT2','PC1_1','PC1_2','PC2_1','PC2_2','MJDREF','A_0_3','A_0_4','A_1_3','A_2_1','A_2_2','A_3_0','A_3_1','A_4_0','B_0_3','B_0_4','B_1_3','B_2_1','B_2_2','B_3_0','B_3_1','B_4_0','AP_0_3','AP_0_4','AP_1_3','AP_2_1','AP_2_2','AP_3_0','AP_3_1','AP_4_0','BP_0_3','BP_0_4','BP_1_3','BP_2_1','BP_2_2','BP_3_0','BP_3_1','BP_4_0','CRPIX1','CRPIX2','CUNIT1','CUNIT2','CTYPE1','CTYPE2','CRVAL1','CRVAL2','LONPOLE','LATPOLE','RADESYS','EQUINOX']
        for wcskey in wcskeydelete:
            if wcskey in list(tempheader.keys()):
                del tempheader[wcskey]

        if not any("SUBSTACK" in s for s in tempheader.keys()):
            tempheader['SUBSTACK']='F'
        
        # Make strings for readnoise and gain
        # Before permanent commissioning, these can be unknown
        # If unknown set to a standard value        
        # 3.0 is a reasonable read_noise
        try:
            readnoise_string=round(float(tempheader['RDNOISE']),2)
        except:
            readnoise_string=3.0
        
        # We just set the gain to 1 to process the file as is. 
        try:
            gain_string=round(float(tempheader['GAIN']),2)
        except:
            gain_string=1.0    
        
        np.save(Path(base) / (
            'EXP' + str(round(tempheader['EXPTIME'],3)) +
            'EXPDOCOSMIC' + str(tempheader['DOCOSMIC']) +
            'DOCOSMICFILTER' + tempheader['FILTER'].split('_')[0] +
            'FILTERSATURATE' + str(int(tempheader['SATURATE'])) + 'SATURATE' +
            'RAdeg' + str(round(tempheader['RAdeg'],3)) +
            'RAdegDECdeg' + str(round(tempheader['DECdeg'],3)) +
            'DECdegGAIN' + str(gain_string) + 'GAINRDNOISE' +
            str(readnoise_string) + 'RDNOISEIMAGEW' +
            str(tempheader['NAXIS1']) + 'IMAGEWIMAGEH' +
            str(tempheader['NAXIS2']) + 'IMAGEHPIXSCALE' +
            str(tempheader['PIXSCALE']) + 'PIXSCALE' +
            file.split('/')[-1].replace('.fits.fz','.npy').replace('.fits','.npy')
        ), imagtemp)
        final_header=copy.deepcopy(tempheader)

        if location != 'local' and location != 'generic':
            try:
                os.remove(file)
            except:
                logging.info ("Couldn't delete " + file)

        return final_header

    except:
        logging.info("***********" + file + " had a problem de-fz-ing")
        logging.info(traceback.format_exc())
        return None
    
def archive_downloader(frame, session):

    with open(frame['filename'], 'wb') as f:
      f.write(session.get(frame['url']).content)

    return str(frame['filename']) + " downloaded."


def wait_for_file(filepath, altpath=None, timeout=6 * 60 * 60, interval=60):
    """Wait for a file to appear on disk.

    Parameters
    ----------
    filepath : str or Path
        Path of the primary file to wait for.
    altpath : str or Path, optional
        Alternative path that also satisfies the wait condition. If ``None``,
        only ``filepath`` is considered.
    timeout : int, optional
        Maximum time in seconds to wait. Defaults to six hours.
    interval : int, optional
        Sleep duration between existence checks in seconds. Defaults to sixty
        seconds.

    Returns
    -------
    bool
        ``True`` if either file exists within the timeout period, ``False``
        otherwise.
    """

    start = time.time()
    p = Path(filepath)
    alt = Path(altpath) if altpath else None
    targets = f"{p}" if not alt else f"{p} or {alt}"

    while not p.exists() and not (alt and alt.exists()):
        if time.time() - start > timeout:
            return False
        logging.info(f"Waiting for file {targets} to appear")
        time.sleep(interval)

    return True

def token_is_older_than(token_name, days=30):
    """Return ``True`` if the date embedded in ``token_name`` is older than ``days``."""

    m = re.search(r"(\d{8})", token_name)
    if not m:
        return False

    try:
        tdate = datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return False

    return (datetime.utcnow().date() - tdate) > timedelta(days=days)

def move_token_to_failed(token_file, output_root=None, token_root=None):
    """Move ``token_file`` from the ``tokens`` directory into ``failedtokens``."""

    tpath = Path(token_file)
    if token_root:
        tpath = Path(token_root) / 'tokens' / tpath.name
    dest_dir = Path(output_root) / 'failedtokens' if output_root else None
    moved = False
    if dest_dir:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tpath), dest_dir / tpath.name)
            logging.info(f"Token file '{tpath}' moved to '{dest_dir}'.")
            moved = True
        except Exception as e:
            logging.info(f"Failed to move token file '{tpath}': {e}")
    if not moved:
        try:
            tpath.unlink()
            logging.info(f"Token file '{tpath}' deleted.")
        except Exception as e:
            logging.info(f"Failed to delete token file '{tpath}': {e}")

def move_token_to_successful(token_file, output_root=None, token_root=None):
    """Move ``token_file`` from the ``tokens`` directory into ``successfultokens``."""

    tpath = Path(token_file)
    if token_root:
        tpath = Path(token_root) / 'tokens' / tpath.name
    dest_dir = Path(output_root) / 'successfultokens' if output_root else None
    moved = False
    if dest_dir:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tpath), dest_dir / tpath.name)
            logging.info(f"Token file '{tpath}' moved to '{dest_dir}'.")
            moved = True
        except Exception as e:
            logging.info(f"Failed to move token file '{tpath}': {e}")
    if not moved:
        try:
            tpath.unlink()
            logging.info(f"Token file '{tpath}' deleted.")
        except Exception as e:
            logging.info(f"Failed to delete token file '{tpath}': {e}")

def wait_for_diskspace(directory="/", threshold=0.75, interval=5, timeout=3 * 60 * 60):
    """
    Waits until the used disk space in a specified directory falls below a threshold,
    with a timeout to exit after a certain period.

    Parameters:
        directory (str): The directory whose disk usage will be checked (default is "/").
        threshold (float): The maximum allowed disk usage ratio (default is 0.75).
        interval (int): The number of seconds to wait between checks (default is 5).
        timeout (int): The maximum number of seconds to wait before exiting (default is 3 hours).

    Returns:
        None
    """
    start_time = time.time()

    while True:
        total, used, free = shutil.disk_usage(directory)

        logging.info(f"Waiting for diskspace to clear in '{directory}' before smartstacks.")
        logging.info("Total: %d GiB" % (total // (2 ** 30)))
        logging.info("Used: %d GiB" % (used // (2 ** 30)))
        logging.info("Free: %d GiB" % (free // (2 ** 30)))

        if (used / total) < threshold:
            break
        
        if time.time() - start_time > timeout:
            logging.info("Timeout reached — exiting.")
            sys.exit(1)
        
        time.sleep(interval)

def check_that_file_opens(file, script_path='filechecker.py', python_command='/usr/bin/python3'):
    """
    Checks whether a given file can be safely opened by invoking an external script.

    Parameters:
        file (str):
            The path to the file that needs to be checked.
        script_path (str, optional):
            Path to the ``filechecker.py`` script. Defaults to ``'filechecker.py'``.
        python_command (str, optional):
            Python executable used to run the script. Defaults to ``'/usr/bin/python3'``.

    Returns:
        str: 
            - Output from the external script if the file is safely opened.
            - 'failed' if the file cannot be opened, resulting in deletion of the file.

    Side effects:
        - Adds the filename to `list_of_nice_files` if the file opens successfully.
        - Deletes the file from the filesystem if it's determined to be unsafe or corrupted.

    Exceptions handled:
        subprocess.CalledProcessError: Raised when the external script encounters an error.
                                      Prints an error message and attempts to delete the problematic file.
    """
    try:
        output = subprocess.check_output([python_command, script_path, file], text=True)
        if 'safely opened' in output:
            logging.info (output)
            return output
    except subprocess.CalledProcessError as e:
        logging.info (file + " is a dodgy file")
        logging.info (e.returncode)
        logging.info (e.output)
        try:
            os.remove(file)
        except:
            pass
        return 'failed'

def header_is_valid(file):
    """Quickly check whether a FITS header can be parsed.

    Parameters
    ----------
    file : str
        Path to the FITS file.

    Returns
    -------
    bool
        ``True`` if the header is readable, ``False`` otherwise.
    """
    try:
        fits.getheader(file, ignore_missing_end=True)
        return True
    except Exception as exc:
        logging.info(f"{file} header check failed: {exc}")
        return False

def hard_drive_activity(drive):
    output = subprocess.check_output(["iostat","-y" ,"5","1","|","grep",drive])
    output = str(output).split(drive)[-1].split(' ')
    while("" in output):
         output.remove("")
    logging.info ("Hard Drive activity at " + str(drive) + " : " + str(float(output[2])))
    return float(output[2])


def wait_for_resources(memory_fraction=70, cpu_fraction=50, wait_for_harddrive=False, workdrive=None, max_wait=7200):
    """Block until system resources drop below thresholds or a timeout occurs."""

    # Randomised wait up to ``max_wait`` seconds helps spread out processes that
    # start at the same time while ensuring we eventually continue.
    random_timeout_period = random.randint(1800, max_wait)

    start_time = time.time()
    vert_timer = start_time

    def log_wait(mem, cpu, hd=None):
        if time.time() - vert_timer > 30:
            msg = f"Waiting: Mem: {mem} CPU: {cpu}"
            if hd is not None:
                msg += f" HD: {hd}"
            msg += f" {datetime.now()}"
            logging.info(msg)
            return time.time()
        return vert_timer

    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        hard_drive_usage = None
        if wait_for_harddrive and workdrive:
            hard_drive_usage = hard_drive_activity(workdrive)

        if ((memory_usage <= memory_fraction) and
            (cpu_usage <= cpu_fraction) and
            (hard_drive_usage is None or hard_drive_usage <= 10000)):
            break

        if time.time() - start_time >= random_timeout_period:
            logging.info("Resource wait timeout reached; continuing execution.")
            break

        vert_timer = log_wait(memory_usage, cpu_usage, hard_drive_usage)
        time.sleep(1)
