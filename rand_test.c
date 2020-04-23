// gcc -Wall -I/home/shadrin/github/gsl/install/include -c rand_test.c
// gcc -L/home/shadrin/github/gsl/install/lib rand_test.o -lgsl -lgslcblas -lm -o rand_test
// ./rand_test
#include <stdio.h>
#include <gsl/gsl_rng.h>

int main ()
{
    int upper = 101;
    int N = 10;
    int seed = 2;
    int res;
    gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
    // 0 seed sets to default seed, which is the same as 1 for gsl_rng_taus, so should start seeding with 1 with multithreading
    gsl_rng_set(r, seed);

    for ( int i=0; i<N; i++ )
    {
        res = gsl_rng_uniform_int(r, upper);
        printf ("%i\n", res);
    }

    free(r);
    return 0;
}
