# examples and instructions

Here is an example query from Robert

```bash
$ mkdir test_run
$ query_coadd_for_meds.py -A 2758897 -o test_run/junk --segmap --bkgimg --psfmodel --usepiff --imglist test_run/test.img.list --ima_list test_run/test.ima.list --head_list test_run/test.head.list --bkg_list test_run/test.bkg.list --seg_list test_run/test.seg.list --psf_list test_run/test.psf.list -s db-desoper -S prod -v 2 --pifftag Y6A1_PIFF --zeropoint NONE --pizza-cutter-yaml=test_run/pizza.yaml --me_proctag=Y6A2_COADD
```
