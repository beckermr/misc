#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fitsio.h>

#define IMG_COMPRESSION "[compress G; qz 0]"
// #define IMG_COMPRESSION ""

// #define IMG_C_TYPE unsigned short
// #define IMG_CFITSIO_BITPIX USHORT_IMG
// #define IMG_CFITSIO_TYPE TUSHORT

#define IMG_C_TYPE short
#define IMG_CFITSIO_BITPIX SHORT_IMG
#define IMG_CFITSIO_TYPE TSHORT

// #define IMG_C_TYPE int
// #define IMG_CFITSIO_BITPIX LONG_IMG
// #define IMG_CFITSIO_TYPE TINT

// #define IMG_C_TYPE unsigned int
// #define IMG_CFITSIO_BITPIX ULONG_IMG
// #define IMG_CFITSIO_TYPE TUINT


void printerror(int status) {
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
    return;
}

int main(void) {
    fitsfile *fptr;
    int status, anynull, test_status;
    char filename[] = "mem://"IMG_COMPRESSION;
    int bitpix = IMG_CFITSIO_BITPIX;
    long naxis = 2;
    long naxes[2] = {5, 2};
    long nimg = 10;
    long i;

    IMG_C_TYPE *img, *read_img;
    IMG_C_TYPE nullval = 100;

    img = (IMG_C_TYPE*) malloc(
        naxes[0] * naxes[1] * sizeof(IMG_C_TYPE)
    );
    read_img = (IMG_C_TYPE*) malloc(
        naxes[0] * naxes[1] * sizeof(IMG_C_TYPE)
    );

    status = 0;

    if (fits_create_file(&fptr, filename, &status)) {
        printerror(status);
    }

    if (fits_create_img(fptr, bitpix, naxis, naxes, &status)) {
        printerror(status);
    }

    for (i=0; i<nimg; ++i) {
        img[i] = i;
    }

    if (fits_write_img(fptr, IMG_CFITSIO_TYPE, 1, nimg, img, &status)) {
        printerror(status);
    }

    if (fits_read_img(
        fptr,
        IMG_CFITSIO_TYPE,
        1,
        nimg,
        &nullval,
        read_img,
        &anynull,
        &status
    )) {
        printerror(status);
    }

    test_status = 0;
    for (i=0; i<nimg; ++i) {
        if (img[i] != read_img[i]) {
            printf(
                "ERROR: images not equal at index %ld: read value %d != correct value %d\n",
                i,
                read_img[i],
                img[i]
            );
            test_status = 1;
        }
    }

    free(img);
    free(read_img);

    if (fits_close_file(fptr, &status)) {
        printerror(status);
    }

    if (test_status == 0) {
        printf("SUCCESS!\n");
    }

    exit(test_status);
}
