#define TNAME CONCAT(CONCAT(test, TEST_NAME), TEST_TYPE)

#pragma message "defining test case: " STR(TNAME)

int TNAME(void) {
    fitsfile *fptr;
    int status, anynull, test_status, bump_sptr;
    char filename[] = "mem://";
    int bitpix = IMG_CFITSIO_BITPIX;
    long naxis = 2;
    long naxes[2] = {5, 4};
    long nimg = 20;
    long i;
    char test_name[] = STR(TNAME);

    printf("running test: %s", test_name);

    IMG_C_TYPE *img, *read_img, *read_img_null;
    IMG_C_TYPE nullval = NAN;
    short *sptr;

    img = (IMG_C_TYPE*) malloc(
        naxes[0] * naxes[1] * sizeof(IMG_C_TYPE)
    );
    read_img = (IMG_C_TYPE*) malloc(
        naxes[0] * naxes[1] * sizeof(IMG_C_TYPE)
    );
    read_img_null = (IMG_C_TYPE*) malloc(
        naxes[0] * naxes[1] * sizeof(IMG_C_TYPE)
    );

    status = 0;

    if (fits_create_file(&fptr, filename, &status)) {
        printerror(status);
    }

    if (fits_create_img(fptr, bitpix, naxis, naxes, &status)) {
        printerror(status);
    }

    // ensure correct bytes for fnan
    img[0] = NAN;
    sptr = (short *) img;
    if (fnan(*sptr) != 1) {
        bump_sptr = 1;
    } else {
        bump_sptr = 0;
    }

    for (i=0; i<nimg; ++i) {
        img[i] = i;
    }
    img[1] = TEST_UNDERFLOW_VAL;
    img[2] = NAN;

    sptr = (short *) img;
    if (bump_sptr) {
        ++sptr;
    }

    if (fits_write_img(fptr, IMG_CFITSIO_TYPE, 1, nimg, img, &status)) {
        printerror(status);
    }

    if (fits_read_img(
        fptr,
        IMG_CFITSIO_TYPE,
        1,
        nimg,
        NULL,
        read_img,
        &anynull,
        &status
    )) {
        printerror(status);
    }

    if (fits_read_img(
        fptr,
        IMG_CFITSIO_TYPE,
        1,
        nimg,
        &nullval,
        read_img_null,
        &anynull,
        &status
    )) {
        printerror(status);
    }

    test_status = 0;
    for (i=0; i<nimg; ++i) {
        if (i == 2) {
            continue;
        }
        if (read_img[i] != read_img_null[i]) {
            if (test_status == 0) {
                printf("\n");
            }
            printf(
                "ERROR: images not equal at index %ld:"
                "\n    read value w/ nullcheck = 1: '%g'"
                "\n    read value w/ nullcheck = 0: '%g'"
                "\n",
                i,
                read_img_null[i],
                read_img[i]
            );
            test_status = 1;
        }
    }

    if (isnan(read_img[1]) != 0) {
        printf("ERROR: nan not read back in properly w/ nullcheck = 0\n");
        test_status = 1;
    }
    if (isnan(read_img_null[1]) != 0) {
        printf("ERROR: nan not read back in properly w/ nullcheck = 1\n");
        test_status = 1;
    }

    free(img);
    free(read_img);

    if (fits_close_file(fptr, &status)) {
        printerror(status);
    }

    if (test_status == 0) {
        printf(" - SUCCESS!\n");
    } else {
        printf("FAILED!\n");
    }

    return test_status;
}
