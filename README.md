# galah-li-DR4
analysing Li in GALAH DR4

# setup 
1. Change directories in config.py to the ones being used, then run config.py (to generate the data and fits folder), or you can put it in manually

2. Download the GALAH DR# file into the info directory (make sure name matches in config.py). e.g. for DR4:
`galah_dr4_allstar_240207.fits` goes into the data folder

3. Setup the `get_id.py` argparse in `setup.sh`, then run it using `bash setup.sh`

4. Download and install `https://github.com/ellawang44/astro_tools`. If you already have `astro_tools` (my own repository) installed then you can skip this step. Might move all things over to the current repo in the future. Currently a bit messy though. 

5. check `qsub_template` and `qsub_loose_template` setup, if you need PBS commands add them. run `get_ids.py` to generate a filled in version of the template. You can change the threshold in `get_ids.py` to control the splitting done. 
If you need to regen the qsub file since the keys are messed up, `run get_ids.py`

6. submit qsub jobs, either via `qsub` or `qsub_dir` depending on how `get_id.py` was run.

7. allstars.py is the script using Breidablik converting EW to abundances. Creates a separate file containing stars with abundances.

