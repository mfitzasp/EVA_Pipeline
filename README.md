(E)ducation (V)alue (A)dded Pipeline: EVApipeline

Previously known (<2024) as OSS Pipeline (https://rtsre.org/index.php/rtsre/article/view/12, https://www.oursolarsiblings.com/the-our-solar-siblings-pipeline-oss-pipeline/).

June 2025: photometry subprocesses now run in their temporary
directories so that the bundled `default.param` and `default.psfex`
configuration files are found correctly.

When the pipeline cannot locate the expected FITS files and the token
in ``localptrarchive/tokens`` is older than 30 days, the pipeline moves
that token from ``cfg['localptrarchivefolder']/tokens`` into the
``EVAreducedfiles/failedtokens`` directory for later inspection.

Upon successful completion, the token file stored in
``cfg['localptrarchivefolder']/tokens`` is moved to
``EVAreducedfiles/successfultokens``.

If a token cannot be moved into these directories it is deleted instead.

The pipeline automatically detects this token file when running in
``generic`` mode.

When generating quickanalysis QAJSON files, only the brightest 500 sources
are profiled if more than 100,000 detections are present in the photometry
catalogue. This keeps the quickanalysis files manageable on extremely
dense fields.

