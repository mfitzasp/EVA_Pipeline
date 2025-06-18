(E)ducation (V)alue (A)dded Pipeline: EVApipeline

Previously known (<2024) as OSS Pipeline (https://rtsre.org/index.php/rtsre/article/view/12, https://www.oursolarsiblings.com/the-our-solar-siblings-pipeline-oss-pipeline/).

June 2025: photometry subprocesses now run in their temporary
directories so that the bundled `default.param` and `default.psfex`
configuration files are found correctly.

When the pipeline cannot locate the expected FITS files and the token is
older than 30 days, the token file is moved into a ``failed_tokens``
folder for later inspection.

Upon successful completion, any provided token file is instead moved
to a ``successful_token`` directory.

The script accepts an optional ``--tokenfile`` argument pointing to the
token that should be removed on completion.

