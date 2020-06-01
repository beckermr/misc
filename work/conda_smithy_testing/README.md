# How to use this code

1. Run the script `run_test.sh` feeding it the version number of the test as `vXYZ`
   (e.g., `./run_test.sh v8`).

2. The script will make PRs. Make sure those PRs pass. Then merge them and check
   that they all pass again.

3. Run script `check_pkg.py` feeding it the build number of the test
   (14 + your version number from 2). It will check that the conda-forge channel has
   all of the builds uploaded.

4. Run the script `clean_branches.sh` to clean out the branches made in step 1.
