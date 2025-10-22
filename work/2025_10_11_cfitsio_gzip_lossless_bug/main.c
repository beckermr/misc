#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fitsio.h>

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
#define IMG_COMPRESSION ""

#define TEST_TYPE u2
#define IMG_C_TYPE unsigned short
#define IMG_CFITSIO_BITPIX USHORT_IMG
#define IMG_CFITSIO_TYPE TUSHORT
#include "test_case.c"

#define TEST_TYPE i2
#define IMG_C_TYPE short
#define IMG_CFITSIO_BITPIX SHORT_IMG
#define IMG_CFITSIO_TYPE TSHORT
#include "test_case.c"

#define TEST_TYPE u4
#define IMG_C_TYPE unsigned int
#define IMG_CFITSIO_BITPIX ULONG_IMG
#define IMG_CFITSIO_TYPE TUINT
#include "test_case.c"

#define TEST_TYPE i4
#define IMG_C_TYPE int
#define IMG_CFITSIO_BITPIX LONG_IMG
#define IMG_CFITSIO_TYPE TINT
#include "test_case.c"


#define TEST_NAME PLIO
#define IMG_COMPRESSION "[compress P; qz 0]"

#define TEST_TYPE u2
#define IMG_C_TYPE unsigned short
#define IMG_CFITSIO_BITPIX USHORT_IMG
#define IMG_CFITSIO_TYPE TUSHORT
#include "test_case.c"

#define TEST_TYPE i2
#define IMG_C_TYPE short
#define IMG_CFITSIO_BITPIX SHORT_IMG
#define IMG_CFITSIO_TYPE TSHORT
#include "test_case.c"


#define TEST_NAME RICE
#define IMG_COMPRESSION "[compress R; qz 0]"

#define TEST_TYPE u2
#define IMG_C_TYPE unsigned short
#define IMG_CFITSIO_BITPIX USHORT_IMG
#define IMG_CFITSIO_TYPE TUSHORT
#include "test_case.c"

#define TEST_TYPE i2
#define IMG_C_TYPE short
#define IMG_CFITSIO_BITPIX SHORT_IMG
#define IMG_CFITSIO_TYPE TSHORT
#include "test_case.c"

#define TEST_TYPE u4
#define IMG_C_TYPE unsigned int
#define IMG_CFITSIO_BITPIX ULONG_IMG
#define IMG_CFITSIO_TYPE TUINT
#include "test_case.c"

#define TEST_TYPE i4
#define IMG_C_TYPE int
#define IMG_CFITSIO_BITPIX LONG_IMG
#define IMG_CFITSIO_TYPE TINT
#include "test_case.c"


#define TEST_NAME HCOMP
#define IMG_COMPRESSION "[compress H; qz 0]"

#define TEST_TYPE u2
#define IMG_C_TYPE unsigned short
#define IMG_CFITSIO_BITPIX USHORT_IMG
#define IMG_CFITSIO_TYPE TUSHORT
#include "test_case.c"

#define TEST_TYPE i2
#define IMG_C_TYPE short
#define IMG_CFITSIO_BITPIX SHORT_IMG
#define IMG_CFITSIO_TYPE TSHORT
#include "test_case.c"

#define TEST_TYPE u4
#define IMG_C_TYPE unsigned int
#define IMG_CFITSIO_BITPIX ULONG_IMG
#define IMG_CFITSIO_TYPE TUINT
#include "test_case.c"

#define TEST_TYPE i4
#define IMG_C_TYPE int
#define IMG_CFITSIO_BITPIX LONG_IMG
#define IMG_CFITSIO_TYPE TINT
#include "test_case.c"


#define TEST_NAME HSCOMP
#define IMG_COMPRESSION "[compress HS; qz 0]"

#define TEST_TYPE u2
#define IMG_C_TYPE unsigned short
#define IMG_CFITSIO_BITPIX USHORT_IMG
#define IMG_CFITSIO_TYPE TUSHORT
#include "test_case.c"

#define TEST_TYPE i2
#define IMG_C_TYPE short
#define IMG_CFITSIO_BITPIX SHORT_IMG
#define IMG_CFITSIO_TYPE TSHORT
#include "test_case.c"

#define TEST_TYPE u4
#define IMG_C_TYPE unsigned int
#define IMG_CFITSIO_BITPIX ULONG_IMG
#define IMG_CFITSIO_TYPE TUINT
#include "test_case.c"

#define TEST_TYPE i4
#define IMG_C_TYPE int
#define IMG_CFITSIO_BITPIX LONG_IMG
#define IMG_CFITSIO_TYPE TINT
#include "test_case.c"


#define TEST_NAME GZIP
#define IMG_COMPRESSION "[compress G; qz 0]"

#define TEST_TYPE u2
#define IMG_C_TYPE unsigned short
#define IMG_CFITSIO_BITPIX USHORT_IMG
#define IMG_CFITSIO_TYPE TUSHORT
#include "test_case.c"

#define TEST_TYPE i2
#define IMG_C_TYPE short
#define IMG_CFITSIO_BITPIX SHORT_IMG
#define IMG_CFITSIO_TYPE TSHORT
#include "test_case.c"

#define TEST_TYPE u4
#define IMG_C_TYPE unsigned int
#define IMG_CFITSIO_BITPIX ULONG_IMG
#define IMG_CFITSIO_TYPE TUINT
#include "test_case.c"

#define TEST_TYPE i4
#define IMG_C_TYPE int
#define IMG_CFITSIO_BITPIX LONG_IMG
#define IMG_CFITSIO_TYPE TINT
#include "test_case.c"


int main(void) {
    int test_status = 0;

    test_status |= test_NONE_u2();
    test_status |= test_NONE_i2();
    test_status |= test_NONE_u4();
    test_status |= test_NONE_i4();

    test_status |= test_PLIO_u2();
    test_status |= test_PLIO_i2();

    test_status |= test_RICE_u2();
    test_status |= test_RICE_i2();
    test_status |= test_RICE_u4();
    test_status |= test_RICE_i4();

    test_status |= test_HCOMP_u2();
    test_status |= test_HCOMP_i2();
    test_status |= test_HCOMP_u4();
    test_status |= test_HCOMP_i4();

    test_status |= test_HSCOMP_u2();
    test_status |= test_HSCOMP_i2();
    test_status |= test_HSCOMP_u4();
    test_status |= test_HSCOMP_i4();

    test_status |= test_GZIP_u2();
    test_status |= test_GZIP_i2();
    test_status |= test_GZIP_u4();
    test_status |= test_GZIP_i4();

    exit(test_status);
}
