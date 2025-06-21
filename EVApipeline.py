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

import logging

# silence matplotlib’s debug noise
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

import argparse
import os
import glob
import math
import shutil
import pickle
import json
import queue
import warnings
from pathlib import Path
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
import traceback

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.utils.exceptions import AstropyUserWarning
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# suppress Astropy warnings
warnings.filterwarnings('ignore', category=AstropyUserWarning)
# limit thread usage
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# helper modules
from modules.general_helpers import (
    wait_for_resources,
    check_that_file_opens,
    cleanup_and_exit,
    wait_for_diskspace,
    load_config,
    archive_downloader, de_fz_file,
    wait_for_file,
    token_is_older_than,
    move_token_to_failed,
    move_token_to_successful
)
from modules.platesolving import multiprocess_crop_images_for_flatness
from modules.archive import archive_preparer
from modules.astrometry import process_preastrom, process_lco_preastrom, run_astrometry_net
from modules.header_functions import freshen_header, value_add_header, human_namer
from modules.final_image import (
    make_banzai_file_out_of_EVA,
    multiprocess_final_image_construction_smartstack,
    multiprocess_final_image_construction_single_image,
    multiprocess_preview_images,
    make_quickanalysis_file
)
from functools import partial
from modules.smart_stack import smart_stack
from modules.photometry import run_source_extractor, run_pre_psfex, run_actual_psfex

# load environment
load_dotenv('.env')


def parse_args():
    p = argparse.ArgumentParser(description='EVA Pipeline')
    p.add_argument('rundate')
    p.add_argument('telescope')
    p.add_argument('mode', choices=['local','online','generic'])
    p.add_argument('camera')
    p.add_argument('filters')
    p.add_argument('osc')
    p.add_argument('pipeid')
    return p.parse_args()

def setup_logging():
    # Reset any existing logging configuration to allow print statements to pass through
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Configure logging; use DEBUG to capture all logs, and force to override existing handlers
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        force=True
    )

def init_config(args):
    cfg = load_config(args.pipeid, args.mode)
    cfg['filters'] = args.filters.split(',')
    return cfg

# Convenience wrapper to apply configured thresholds
def wait_for_resources_cfg(cfg, **kwargs):
    """Call :func:`wait_for_resources` using thresholds from ``cfg``."""
    wait_for_resources(
        memory_fraction=cfg.get('memory_threshold', 70),
        cpu_fraction=cfg.get('cpu_threshold', 50),
        **kwargs,
    )

def prepare_dirs(cfg, args):
    if args.mode == 'online':
        base = Path(cfg['working_directory']) / args.telescope / args.rundate
        base.mkdir(parents=True, exist_ok=True)
    else:
        base = Path.cwd()
    
    for d in [cfg['memmappath'], cfg['local_output_folder'], cfg['shortexposure_output_folder'],
              cfg['largedataset_output_folder'], cfg['ingestion_output_folder'], base/'Targets']:
        Path(d).mkdir(parents=True, exist_ok=True)
    # pipeline working
    Path(cfg['working_directory']) / 'PipelineWorking' / args.telescope
    return base


def get_tokenfile_from_info(base):
    """Return the token file path described in ``info_for_eva.json`` if present."""
    info_path = Path(base) / 'info_for_eva.json'
    if info_path.exists():
        try:
            info = json.load(open(info_path))
            return info.get('tokenfile') or info.get('token_file')
        except Exception:
            pass
    return None


def prepare_local_output_dirs(files, cfg):
    """Create DAY-OBS specific directories inside the local output folder."""
    if not cfg.get('copy_to_a_local_output_folder', False):
        return

    dayobs_set = set()
    for f in files:
        try:
            hd = fits.open(f)
            hdr = hd[1].header if ('BZESK' in f) or ('.fits.fz' in f) else hd[0].header
            day = hdr.get('DAY-OBS')
            if day:
                dayobs_set.add(str(day))
            hd.close()
        except Exception:
            continue

    for d in dayobs_set:
        base = Path(cfg['local_output_folder']) / d
        for sub in [
            'fits', 'photometry', 'pngs', 'previews', 'thumbnails',
            'quickanalysis'
        ]:
            (base / sub).mkdir(parents=True, exist_ok=True)


def download_phase(cfg, args, base):
    if args.mode != 'online': return
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=2.0, allowed_methods=frozenset(['GET','POST']))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    fronts = []
    if args.telescope != 'lco':
        d = args.rundate
        fronts.append(f"https://archiveapi.photonranch.org/frames/?dayobs={d[:4]}-{d[4:6]}-{d[6:8]}&limit=10000&reduction_level=00&include_configuration_type=EXPOSE&telescope_id={args.telescope}")
    else:
        d = args.rundate.replace('-','')
        for prop in ['LCO2023A-115','LCOEPO2014B-007','LCOEPO2020A-001','LCOEPO2014B-010',
                     'LCOEPO2018A-006','LCOEPO2018A-009','LCOEPO2018B-001','LCOEPO2024B-004','LCO2024B-117']:
            fronts.append(f"https://archive-api.lco.global/frames/?DAY_OBS={d[:4]}-{d[4:6]}-{d[6:8]}&limit=10000&reduction_level=91&include_configuration_type=EXPOSE&PROPID={prop}")
    logging.info('Download frontends: %s', fronts)
    queue_dl = queue.Queue()
    for fe in fronts:
        try:
            data = session.get(fe).json()
            frames = data.get('results', [])
        except:
            continue
        off = 0
        while len(frames) % 1000 == 0 and frames:
            off += 1000
            url = f"{fe}&offset={off}"
            try:
                page = session.get(url).json()
                frames += page.get('results', [])
            except:
                break
        for fr in frames:
            # Save each downloaded file inside the working directory
            fr['filename'] = str(Path(base) / Path(fr['filename']).name)
            queue_dl.put(fr)
    wait_for_diskspace(cfg['working_directory'], 0.9)
    def dl(item): return archive_downloader(item, session)
    
    while not queue_dl.empty():
        batch = [queue_dl.get() for _ in range(min(8, queue_dl.qsize()))]
        with ThreadPool(8) as tp:
            tp.map(dl, batch)
        for _ in batch:
            queue_dl.task_done()


def collect_files(cfg, args, base, tokenfile=None):
    if args.rundate == 'localfolder':
        return [str(p) for p in Path(base).glob('*.fit*')]
    if args.mode == 'generic':
        info = json.load(open(Path(base)/'info_for_eva.json'))
        flist = []
        for f in info['files_to_process']:
            splitfile=f.split('_')
            camname=splitfile[0].split('-')[1]
            filedate=splitfile[2].split('-')[1]
            p=Path(cfg['localptrarchivefolder'] +'/'+camname+'/'+filedate+'/'+f)
            alt=Path(cfg['localptrarchivefolder'] +'/'+camname+'/'+filedate+'/'+f.replace('.fits.fz','.fits'))

            if not p.exists() and not alt.exists():
                if not wait_for_file(p, altpath=alt, timeout=60 * 60):
                    remove_token = token_is_older_than(Path(base).name)
                    if remove_token and tokenfile:
                        move_token_to_failed(tokenfile)
                    cleanup_and_exit(
                        os.path.expanduser('~'),
                        base,
                        original_token_file=tokenfile,
                        remove_token=True
                    )
                    return []

            if p.suffixes[-2:] == ['.fits','.fz']:
                # Sometimes, particularly if this is run twice or more
                # The file will already be uncompressed.
                out = Path(cfg['localptrarchivefolder'] +'/'+camname+'/'+filedate) / f.replace('.fits.fz','.fits')

                if not p.exists() and alt.exists():
                    flist.append(str(alt))
                    continue

                try:
                    hd = fits.open(p, ignore_missing_end=True)
                    arr = np.asarray(hd[1].data if len(hd)>1 else hd[0].data)
                    hdr = hd[1].header if len(hd)>1 else hd[0].header
                    hd.close()
                    fits.PrimaryHDU(arr, hdr).writeto(out, overwrite=True)
                    try:
                        p.unlink()
                    except:
                        try:
                            os.remove(cfg['localptrarchivefolder'] +'/'+camname+'/'+filedate+'/'+f)
                        except:
                            logging.info("Couldn't remove " + cfg['localptrarchivefolder'] +'/'+camname+'/'+filedate+'/'+f)

                    flist.append(str(out))

                except:
                    p=alt

                    if os.path.exists(p):
                        logging.info ("couldn't open fits.fz,already uncompressed. ")
                        flist.append(str(out))
                    else:
                        logging.info ("Couldn't find file")
            else:
                if p.exists():
                    flist.append(str(p))
                elif alt.exists():
                    flist.append(str(alt))
                else:
                    logging.info ("Couldn't find file")
        return flist

    # if neither of them then return the files in the root of the current basedirectory
    return [str(p) for p in Path(base).glob('*.fi*')]


def check_and_deflate(files, cfg, args, base):
    logging.info('Checking files')
    # Get the subprocess script into the directory
    shutil.copy(os.path.expanduser(cfg['codedir']) + '/subprocesses/filechecker.py', Path(base) / 'filechecker.py')
    script_path = str(Path(base) / 'filechecker.py')
    wait_for_resources_cfg(cfg)
    cpu = os.cpu_count() or 1
    n = max(1, min(math.floor(cpu*0.25), len(files)))
    if cfg['multiprocess']['file_checking']:
        with Pool(n) as p:
            fn = partial(check_that_file_opens, script_path=script_path, python_command=cfg['python_command'])
            res = p.map(fn, files)
    else:
        fn = partial(check_that_file_opens, script_path=script_path, python_command=cfg['python_command'])
        res = [fn(f) for f in files]
    val = [(r.split()[-1], args.mode) for r in res if r and 'failed' not in r]
    fps, mods = zip(*val) if val else ([], [])
    headers = []
    if cfg['multiprocess']['de_fz_file']:
        with Pool(n) as p:
            headers = p.starmap(de_fz_file, [(x, base) for x in val])
    else:
        headers = [de_fz_file(x, base) for x in val]
    return list(fps), [h for h in headers if h]


def target_phase(base):
    tdir = Path(base) / 'Targets'
    shutil.rmtree(tdir, ignore_errors=True)
    tdir.mkdir(parents=True, exist_ok=True)
    for f in Path(base).glob('*.np*'):
        shutil.move(str(f), tdir / f.name)
    return tdir


def pre_astrometry(tdir, headers, cfg, args):
    logging.info('Pre-Astrometry')
    wait_for_resources_cfg(cfg, wait_for_harddrive=True, workdrive=cfg['workdrive'])
    files = [str(p) for p in Path(tdir).glob('*.npy')]
    cpu = os.cpu_count() or 1
    n = max(1, min(math.floor(cpu*0.25), len(files)))
    fn = process_lco_preastrom if args.telescope=='lco' else process_preastrom
    if cfg['multiprocess']['bias_dark_flat_bpm']:
        with Pool(n) as p:
            p.map(fn, files)
    else:
        for f in files:
            fn(f)
    wait_for_resources_cfg(cfg, wait_for_harddrive=True, workdrive=cfg['workdrive'])

    # Refresh the file list to pick up the newly created FLATTED files
    files = [str(p) for p in Path(tdir).glob('FLATTED*.npy')]
    n2 = max(1, min(math.floor(cpu*0.25), len(files)))

    astrometry_timeout = cfg.get('astrometry_timeout', 900)
    run_timeout = cfg.get('astrometry_run_timeout', astrometry_timeout)
    args_list = [(f, cfg['codedir'], astrometry_timeout) for f in files]

    results = []
    executor = ProcessPoolExecutor(max_workers=n2)
    try:
        futures = [executor.submit(run_astrometry_net, *a) for a in args_list]
        for f, fut in zip(files, futures):
            try:
                results.append(fut.result(timeout=run_timeout))
            except FuturesTimeoutError:
                logging.info("Astrometry timed out for %s", f)
                fut.cancel()
                results.append((Path(f).name.split('PIXSCALE')[-1], None))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Map returned WCS headers back to their originating FITS files
    wcs_map = {name: hdr for name, hdr in results if hdr}
    for h in headers:
        if h['ORIGNAME'].replace('.fits','').replace('.fz','') in wcs_map:
            h.update(wcs_map[h['ORIGNAME'].replace('.fits','').replace('.fz','')])
        
    return headers

def header_merge(headers, base):
    """
    Merge WCS and FWHM info into headers and relocate all .npy files
    from Targets to workingdirectory, renaming based on PIXSCALE suffix.
    """
    targets_dir = Path(base) / 'Targets'
    working_dir = Path(base) / 'workingdirectory'
    working_dir.mkdir(parents=True, exist_ok=True)

    # Move and rename every .npy in Targets
    for npy_path in targets_dir.glob('*.npy'):
        parts = npy_path.name.split('PIXSCALE')
        new_name = parts[-1] if len(parts) > 1 else npy_path.name
        dest = working_dir / new_name
        shutil.move(str(npy_path), str(dest))

    out = []
    for h in headers:
        name = h['ORIGNAME']
        # Merge WCS if present
        try:
            wfile = name.replace('.fits.fz', '.wcs').replace('.fits', '.wcs')
            wfile_path = Path(base) / wfile
            if wfile_path.exists():
                wh = fits.open(wfile_path)[0].header
                h.update(wcs.WCS(wh).to_header(relax=True))
                # Calculate camera rotation from WCS
                try:
                    c11 = h.get('CD1_1', h.get('PC1_1'))
                    c12 = h.get('CD1_2', h.get('PC1_2'))
                    if c11 is not None and c12 is not None:
                        rot = np.degrees(np.arctan2(-c12, c11)) % 360
                        h['CAMROTAT'] = (rot, 'Camera rotation angle [deg]')
                except Exception:
                    pass
                wfile_path.unlink()
        except:
            print (traceback.format_exc())
        # Merge FWHM info if present
        try:
            ffile = name.replace('.fits.fz', '.fwhm').replace('.fits', '.fwhm')
            ffile_path = Path(base) / ffile
            if ffile_path.exists():
                ff = pickle.load(open(ffile_path, 'rb'))
                for k in ['SKYLEVEL', 'FWHM', 'FWHMpix', 'FWHMasec', 'FWHMstd', 'NSTARS']:
                    # h[k] = ff.get(k, 'Unknown')
                    val = ff.get(k, None)
                    # 1) missing entirely → we’ll use "Unknown"
                    # 2) present but NaN → also "Unknown"
                    if val is None or (isinstance(val, float) and math.isnan(val)) \
                                    or (isinstance(val, np.floating) and np.isnan(val)):
                        h[k] = 'nan'
                    else:
                        try:
                            h[k] = val
                        except:
                            h[k] = 'nan'
                ffile_path.unlink()
        except:
            print (traceback.format_exc())
        
        out.append(h)
    return out

def cleanup_intermediate(base):
    for pat in ['*.fi*', '*.npy', '*.wcs']:
        for f in Path(base).glob(pat):
            try:
                f.unlink()
            except Exception:
                pass

def make_dirs_output(base):
    for d in ["workingdirectory", "outputdirectory", "sstacksdirectory", "lstacksdirectory",
              "cimagesdirectory", "previews", "thumbnails", "smalljpgs", "colours",
              "photometry", "fullphotcatalogues", "quickanalysis"]:
        Path(base, d).mkdir(exist_ok=True)

def enrich_build(headers, cfg, args, base):
    logging.info('Value-add to headers and naming')
    human_names = []
    for i, h in enumerate(headers):
        headers[i] = value_add_header(h, args.telescope)
        human_names.append(human_namer(headers[i]))
    wait_for_resources_cfg(cfg)
    logging.info('Freshening headers')
    with Pool() as p:
        headers = p.map(freshen_header, headers)
    wait_for_resources_cfg(cfg)
    logging.info('Cropping for flatness')
    with Pool() as p:
        tasks = [(h, base) for h in headers]
        headers = p.starmap(multiprocess_crop_images_for_flatness, tasks)

    # Ensure SMARTSTK exists before files are organised into smartstack
    # directories. Some historical LCO headers may lack this field.
    if args.telescope.strip().lower() == 'lco':
        for h in headers:
            if 'SMARTSTK' not in h:
                reqnum = str(h.get('REQNUM', '')).strip()
                filt = str(h.get('FILTER', '')).split('_')[-1]
                h['SMARTSTK'] = 'lco' + reqnum + filt

    return headers, human_names

def construct_images(headers, human_names, cfg, args, base):
    logging.info('Constructing images')
    wait_for_resources_cfg(cfg, wait_for_harddrive=True, workdrive=cfg['workdrive'])
    cpu = os.cpu_count() or 1
    tasks = [(h['ORIGNAME'].replace('.fits.fz','.npy').replace('.fits','.npy'), h, nm, base)
             for h, nm in zip(headers, human_names)]
    n = max(1, min(math.floor(cpu*0.25), len(tasks)))
    with Pool(n) as p:
        p.starmap(multiprocess_final_image_construction_single_image, tasks)

    # Prepare smartstack files
    # Remove entries with SMARTSTK=='no' and move .npy accordingly
    # workingdirectory holds the npy outputs
    working_dir = Path(base) / 'workingdirectory'
    sstacks_dir = Path(base) / 'sstacksdirectory'
    sstacks_dir.mkdir(exist_ok=True)

    valid_headers = []
    for h in headers:
        stk = h.get('SMARTSTK')
        npy_name = h['ORIGNAME'].replace('.fits.fz','.npy').replace('.fits','.npy')
        src = working_dir / npy_name
        if stk == 'no':
            # remove unwanted
            if src.exists(): src.unlink()
            continue
        valid_headers.append(h)
        # make subdir
        subdir = sstacks_dir / str(stk)
        subdir.mkdir(parents=True, exist_ok=True)
        dest = subdir / npy_name
        shutil.move(str(src), str(dest))
        # write header placeholder without mutating WCS information
        headfile = subdir / npy_name.replace('.npy', '.head')
        with open(headfile, 'wb') as fp:
            pickle.dump(h, fp)

    # Now run smart_stack on each sub-directory
    for sub in sstacks_dir.iterdir():
        if not sub.is_dir() or sub.name == 'no':
            continue
        files = list(sub.glob('*.npy'))
        if not files:
            logging.warning('No files in smartstack dir %s', sub)
            continue
        if len(files) > 100:
            logging.info('Skipping smart stack %s: too many files (%d)', sub.name, len(files))
            continue
        # convert Paths to str
        wait_for_resources_cfg(cfg)
        smart_stack([str(f) for f in files], args.telescope, str(base), cfg['memmappath'], cfg['calibration_directory'], cfg['codedir'])

    wait_for_resources_cfg(cfg)
    # Final smart-stack image construction
    fits_ss = glob.glob(str(Path(base) / 'sstacksdirectory' / '*.fits'))
    # collect any variance frames; these will be copied out but not processed
    variance_frames = [f for f in fits_ss if os.path.basename(f).startswith('variance_')]
    fits_ss = [f for f in fits_ss if not os.path.basename(f).startswith('variance_')]
    
    n2 = max(1, min(math.floor(cpu*0.25), len(fits_ss)))
    with Pool(n2) as p:
        tasks = [(f, base) for f in fits_ss]
        p.starmap(multiprocess_final_image_construction_smartstack, tasks)

    # Copy variance frames directly to outputdirectory without processing
    for vf in variance_frames:
        dest = Path(base) / 'outputdirectory' / os.path.basename(vf)
        shutil.copy(vf, dest)

    # Previews
    wait_for_resources_cfg(cfg)
    previews = glob.glob(str(Path(base) / 'outputdirectory' / '*.fits'))
    # don't make preview jpgs of variance frames
    previews = [f for f in previews if not os.path.basename(f).startswith('variance_')]
    try:
        n3 = max(1, min(math.floor(cpu*0.25), len(previews)))
        with Pool(n3) as p:
            p.map(multiprocess_preview_images, previews)
    except:
        logging.info("Failed on making previews... usually due to blank images")

def do_photometry(cfg, base):
    logging.info('Photometry')
    
    # Can't find a way around actually copying the file into the directory
    shutil.copy(os.path.expanduser(cfg['codedir']) + '/photometryparams/default.psfex', Path(base)/'default.psfex')
    
    wait_for_resources_cfg(cfg, wait_for_harddrive=True, workdrive=cfg['workdrive'])
    files = glob.glob(str(Path(base)/'outputdirectory/*.fits'))
    # don't photometer variance frames
    files = [f for f in files if not os.path.basename(f).startswith('variance_')]
    cpu = os.cpu_count() or 1
    n = max(1, min(math.floor(cpu*0.25), len(files)))
    with Pool(n) as p:
        p.starmap(run_source_extractor, [(f, cfg['codedir']) for f in files])
    wait_for_resources_cfg(cfg)
    with Pool(n) as p:
        p.starmap(run_pre_psfex, [(f, cfg['codedir']) for f in files])
    wait_for_resources_cfg(cfg)
    with Pool(n) as p:
        p.starmap(run_actual_psfex, [(f, cfg['codedir']) for f in files])
    

def do_quickanalysis(cfg, base):
    logging.info('Quickanalysis files')
    files = glob.glob(str(Path(base)/'outputdirectory/*.fits'))
    cpu = os.cpu_count() or 1
    n = max(1, min(math.floor(cpu*0.25), len(files)))
    with Pool(n) as p:
        p.map(make_quickanalysis_file, files)

        
def do_banzai_file_type(cfg, telescope, base):
    logging.info('Banzai-esque Files')
    files = glob.glob(str(Path(base)/'outputdirectory/*.fits'))
    cpu = os.cpu_count() or 1
    n = max(1, min(math.floor(cpu*0.25), len(files)))

    smstack_ids = set()
    for f in files:
        if 'SmSTACK-' in os.path.basename(f):
            try:
                sid = fits.getheader(f, memmap=False).get('SSTKID')
                if sid:
                    smstack_ids.add(sid)
            except Exception:
                logging.info('Could not read header for %s', f)

    banzai_list = []
    for file in files:
        skip = False
        if 'SmSTACK-' not in os.path.basename(file):
            try:
                sid = fits.getheader(file, memmap=False).get('SSTKID')
                if sid and sid in smstack_ids:
                    logging.info('Skipping BZESK creation for %s as SmSTACK exists', file)
                    skip = True
            except Exception:
                logging.info('Could not read header for %s', file)
        if not skip:
            banzai_list.append([file, telescope, str(base), cfg['calibration_directory']])

    if banzai_list:
        with Pool(n) as p:
            p.starmap(make_banzai_file_out_of_EVA, banzai_list)


def do_archive(cfg, base):
    logging.info('Archive preparation')
    wait_for_diskspace(cfg['working_directory'], 0.9)
    wait_for_resources_cfg(cfg, wait_for_harddrive=True, workdrive=cfg['workdrive'])
    files = glob.glob(str(Path(base)/'outputdirectory/*.fit*'))
    tasks = [[f, cfg['largedataset_output_folder'], cfg['shortexposure_output_folder'], cfg['ingestion_output_folder'], cfg['local_output_folder'], True, True] # The two trues are local copy then ingest... will be a config soon
             for f in files]
    cpu = os.cpu_count() or 1
    n = max(1, min(math.floor(cpu*0.25), len(tasks)))
    with Pool(n) as p:
        p.starmap(archive_preparer, tasks)
    

def main():    
    
    # Sample commands
    # python3 EVApipeline.py 20250101 lco online na na mono ecopipe         << A small set of LCO files that do not smartstack but download from LCO
    # python3 EVApipeline.py localfolder aro1 local na na mono ecopipe     << Run a set of files in the local folder
    
    args = parse_args()
    setup_logging()
    cfg = init_config(args)
    base = prepare_dirs(cfg, args)
    tokenfile = get_tokenfile_from_info(base)
    download_phase(cfg, args, base)
    files = collect_files(cfg, args, base, tokenfile)
    if not files:
        logging.info('No files found to process. Exiting early.')
        if args.rundate != 'localfolder':
            cleanup_and_exit(
                os.path.expanduser('~'),
                base,
                original_token_file=tokenfile,
                remove_token=True
            )
        return
    prepare_local_output_dirs(files, cfg)
    print (files)
    files, headers = check_and_deflate(files, cfg, args, base)
    tdir = target_phase(base)
    # Pre-astrometry and WCS generation run inside Targets dir
    wait_for_resources_cfg(cfg)
    headers = pre_astrometry(tdir, headers, cfg, args)
    headers = header_merge(headers, base)
    cleanup_intermediate(base)
    make_dirs_output(base)
    headers, human_names = enrich_build(headers, cfg, args, base)
    wait_for_resources_cfg(cfg)
    construct_images(headers, human_names, cfg, args, base)
    wait_for_resources_cfg(cfg)
    do_photometry(cfg, base)
    wait_for_resources_cfg(cfg)
    do_quickanalysis(cfg, base)

    # Make BANZAI files
    wait_for_resources_cfg(cfg)
    do_banzai_file_type(cfg, args.telescope, base)
        
    do_archive(cfg, base)


    if not args.rundate == 'localfolder':
        if tokenfile:
            move_token_to_successful(tokenfile)
        cleanup_and_exit(
            os.path.expanduser('~'),
            base,
            original_token_file=tokenfile,
            remove_token=False
        )


if __name__ == '__main__':
    main()
