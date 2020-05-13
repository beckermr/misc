# How to use this code

1. Check out the repo https://github.com/beckermr/cf-autotick-bot-test-package-feedstock.git

2. Run the script `make_branches.sh` feeding it the version number of the test as `vXYZ`
   (e.g., `./make_branches.sh v8`).

3. The script will make 9 PRs. Make sure those PRs pass. Then merge them and check
   that they all pass again.

4. Finally, run script `check_pkg.py` feeding it the build number of the test
   (14 + your version number from 2). It will check that the conda-forge channel has
   all of the builds uploaded.
