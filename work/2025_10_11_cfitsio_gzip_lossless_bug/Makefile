test:
	${CC} ${CFLAGS} ${LDFLAGS} -I./cfitsio/include -L./cfitsio/lib -lm -lcfitsio main.c -o test_cfitsio_gzip_lossless_bug
	./test_cfitsio_gzip_lossless_bug

cfitsio: FORCE
	cd cfitsio && ./configure
	cd cfitsio && make clean
	cd cfitsio && make
	cd cfitsio && make install

FORCE: ;
