(E)ducation (V)alue (A)dded Pipeline: EVApipeline

Previously known (<2024) as OSS Pipeline (https://rtsre.org/index.php/rtsre/article/view/12, https://www.oursolarsiblings.com/the-our-solar-siblings-pipeline-oss-pipeline/).

June 2025: photometry subprocesses now run in their temporary
directories so that the bundled `default.param` and `default.psfex`
configuration files are found correctly.

Tokens older than 30 days are no longer deleted on failure. Instead they
are moved to a `failed_tokens` directory next to the original token
location.

