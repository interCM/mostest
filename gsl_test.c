// Install GSL with:
// ./configure --prefix=/home/shadrin/github/gsl/install/
// make
// make install
//
// Update .basrc with:
// export LD_LIBRARY_PATH="/home/shadrin/github/gsl/install/lib"
//
// Then compile, link and run:
// gcc -Wall -I/home/shadrin/github/gsl/install/include -c gsl_test.c
// gcc -L/home/shadrin/github/gsl/install/lib gsl_test.o -lgsl -lgslcblas -lm -o gsl_test
// ./gsl_test
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>

int
main (void)
{
    double x = 15.0;
    double y = gsl_sf_bessel_J0 (x);
    printf ("J0(%g) = %.18e\n", x, y);
    return 0;
}
