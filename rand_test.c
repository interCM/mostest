// gcc -Wall -O2 -fopenmp -I/home/shadrin/github/gsl/install/include -c rand_test.c
// gcc -L/home/shadrin/github/gsl/install/lib rand_test.o -lgsl -lgslcblas -lm -o rand_test
// ./rand_test
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <omp.h>

int main ()
{
    int max_threads = 4;
    // https://stackoverflow.com/questions/11095309/openmp-set-num-threads-is-not-working
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(max_threads); // Use 4 threads for all consecutive parallel regions

    int nthreads = (int)omp_get_max_threads();
    printf("Max number of threads = %i\n", nthreads);

    int upper = 11;
    int N = 100000000;
    int res, i;
    size_t tid;
    gsl_rng *rngs[nthreads];
    for( i=0; i<nthreads; i++ )
    {
        rngs[i] = gsl_rng_alloc(gsl_rng_taus);
        // 0 seed sets to default seed, which is the same as 1 for gsl_rng_taus, so should start seeding with 1 with multithreading
        gsl_rng_set(rngs[i], i+1);
    }
    
    long long int sum = 0;

    #pragma omp parallel for default(shared) private(i, res, tid) schedule(dynamic) reduction(+:sum)
    for ( i=0; i<N; i++ )
    {
        tid = omp_get_thread_num();
        res = gsl_rng_uniform_int(rngs[tid], upper);
        sum += res;
    }

    printf ("%f\n", (float)sum/(float)N);
    for( i=0; i<nthreads; i++ )
            free(rngs[i]);
    return 0;
}
