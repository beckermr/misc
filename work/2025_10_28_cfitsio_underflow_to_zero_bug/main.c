#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fitsio.h>
#include <fitsio2.h>

#define XSTR(x) #x
#define STR(x) XSTR(x)

#define CONCAT_IMPL(A, B) A##_##B
#define CONCAT(A, B) CONCAT_IMPL(A, B)

void printerror(int status) {
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
    return;
}

#define TEST_NAME NONE

#define TEST_TYPE f4
#define TEST_UNDERFLOW_VAL 10
#define IMG_C_TYPE float
#define IMG_CFITSIO_BITPIX FLOAT_IMG
#define IMG_CFITSIO_TYPE TFLOAT
#include "test_case.c"

#define TEST_TYPE f8
#define TEST_UNDERFLOW_VAL 10
#define IMG_C_TYPE double
#define IMG_CFITSIO_BITPIX DOUBLE_IMG
#define IMG_CFITSIO_TYPE TDOUBLE
#include "test_case.c"

#define TEST_NAME UNDER

#define TEST_TYPE f4
#define TEST_UNDERFLOW_VAL -9.0E-44
#define IMG_C_TYPE float
#define IMG_CFITSIO_BITPIX FLOAT_IMG
#define IMG_CFITSIO_TYPE TFLOAT
#include "test_case.c"

#define TEST_TYPE f8
#define TEST_UNDERFLOW_VAL -9.0E-309
#define IMG_C_TYPE double
#define IMG_CFITSIO_BITPIX DOUBLE_IMG
#define IMG_CFITSIO_TYPE TDOUBLE
#include "test_case.c"

int main(void) {
    int test_status = 0;

    test_status |= test_NONE_f4();
    test_status |= test_NONE_f8();

    test_status |= test_UNDER_f4();
    test_status |= test_UNDER_f8();

    exit(test_status);
}
