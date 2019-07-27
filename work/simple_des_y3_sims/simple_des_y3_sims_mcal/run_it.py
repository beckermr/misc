from __future__ import print_function
import os
import glob
import logging
import subprocess
from contextlib import contextmanager

import joblib
import yaml

import desmeds
from .megamixer import ImSimMegaMixer

logger = logging.getLogger(__name__)

# never chang this
MEDSCONF = 'y3v02'


def run_mcal(tilename, output_desdata, mcal_config, seed, meds_files=None):
    """
    """

    try:
        # temporarily set the environ
        ngmixer_output_dir = os.path.join(
            output_desdata, "ngmixer")
        meds_dir = os.path.join(output_desdata, 'meds')
        safe_mkdir(ngmixer_output_dir)
        safe_mkdir(meds_dir)
        kwargs = {
            'NGMIXER_OUTPUT_DIR': ngmixer_output_dir,
            'MEDS_DIR': meds_dir
        }
        with environ(**kwargs):
            # get MEDS files
            if meds_files is None:
                meds_files = get_meds_files(
                    MEDSCONF,
                    tilename,
                    mcal_config['jobs']['bands'])
            logger.info(' MEDS files %s', meds_files)

            # we write a tempfile with the config since we might have changed
            # it
            config_file = os.path.join(
                os.environ['TMPDIR'], 'run-%s.yaml' % mcal_config['run'])
            with open(config_file, 'w') as fp:
                yaml.dump(mcal_config, fp)

            # now setup the scripts
            mmixer = ImSimMegaMixer(
                config_file, meds_files,
                psf_map_files=None,
                clobber=True,
                seed=seed)
            mmixer.setup()

            # we remove checkpoint files since those can be hanging around
            # from failed runs
            for s in mmixer.scripts:
                d = os.path.dirname(s)
                check_files = glob.glob(
                    os.path.join(d, "*checkpoint.fits"))
                for f in check_files:
                    logger.debug("removing checkpoint file %s" % f)
                    try:
                        os.remove(f)
                    except Exception:
                        pass

            # Now its time to run the actual ngmixer scripts.
            # for this part, each task picks up a single tile and runs it via
            # joblib
            def _run_script(chunk_script, output_file):
                for _ in range(10):
                    try:
                        os.remove(output_file)
                    except Exception:
                        pass

                    returncode, _ = run_subprocess(["bash", chunk_script])

                    output_file_exists = os.path.isfile(output_file)
                    if returncode == 0 and output_file_exists:
                        break

            jobs = [
                joblib.delayed(_run_script)(chunk_script, output_file)
                for chunk_script, output_file in zip(
                    mmixer.scripts, mmixer.output_files)]

            print('# of jobs', len(jobs))

            joblib.Parallel(
                n_jobs=-1, backend='loky',
                verbose=100, pre_dispatch='2*n_jobs',
                max_nbytes=None)(jobs)

            ngmixer_file = mmixer.collate()
            logger.debug("collated file %s", ngmixer_file)
    finally:
        try:
            os.remove(config_file)
        except Exception:
            pass


def safe_mkdir(pth):
    try:
        os.makedirs(pth)
    except Exception:
        pass


@contextmanager
def environ(**env):
    """Temporarily set environment variables inside the context manager and
    fully restore previous environment afterwardsself.

    Pass the env vars as keywords.

    Code from https://gist.github.com/igniteflow/7267431#gistcomment-2553451.

    Example
    -------
    >>> with environ(USER='xyz'):
    >>>     print(os.environ['USER'])
    >>> print(os.environ['USER'])
    """
    original_env = {key: os.getenv(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


def run_subprocess(command):
    c = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = c.communicate()[0]
    return c.returncode, stdout


def get_meds_files(medsconf, tile_id, bands):
    """Return the paths to the MEDS files.

    Note these paths use the current MEDS_DIR env var.
    """
    fnames = []
    for band in bands:
        fnames.append(desmeds.files.get_meds_file(medsconf, tile_id, band))

    return fnames
