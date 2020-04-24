// gcc -c mostlib.c -I/home/shadrin/github/gsl/install/include -O2 -fopenmp -pedantic-errors -Wall -Wextra -Wsign-conversion -Wconversion -Werror
// gcc mostlib.o -o mostlib -L/home/shadrin/github/gsl/install/lib -O2 -pedantic-errors -Wall -Wextra -Wsign-conversion -Wconversion -Werror -fopenmp -lgsl -lgslcblas -lm
// ./mostlib

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <omp.h>


void getByteMap(signed char *byteMap)
{
    const signed char genotypeCodes[4] = {2, -1, 1, 0};
    for( int b=0; b<256; b++ )
    {
        for( int i=0; i<4; i++ )
            byteMap[4*b + i] = genotypeCodes[(b >> 2*i) & 3];
    }
}


void getHetHomMissInd(unsigned char *bedGeno, int nSamples, int *iiHeterozygous,
    int *iiHomozygous, int *iiMiss, int *nHeterozygous, int *nHomozygous,
    int *nMiss, signed char *byteMap)
{
    // bedGeno = single genotype from bed file
    // iiHeterozygous, iiHomozygous, iiMiss = arrays to fill with indices of 2, 1 and missing values in bedGeno
    // len(iiHeterozygous) == len(iiHomozygous) = nSamples
    // nHeterozygous, nHomozygous, nMiss = pointers to int, number of 2, 1 and missing values in bedGeno
    int iSample = 0; // current sample index
    int nM = 0; // number of missing genotypes
    int nHet = 0;
    int nHom = 0;
    unsigned char bedByte;
    int iBedByte = 0;
    signed char byteGeno;
    int iByteGeno;

    while( iSample < nSamples )
    {
        bedByte = bedGeno[iBedByte++];
        for( iByteGeno=0; iByteGeno<4; iByteGeno++ )
        {
            byteGeno = byteMap[4*bedByte + iByteGeno];
            if( byteGeno != -1 )
            {
                if( byteGeno == 2 )
                    iiHeterozygous[nHet++] = iSample;
                else if( byteGeno == 1 )
                    iiHomozygous[nHom++] = iSample;
            }
            else
            {
                iiMiss[nM++] = iSample;
            }
            if( ++iSample == nSamples )
                break;
        }
    }
    *nHeterozygous = nHet;
    *nHomozygous = nHom;
    *nMiss = nM;
}


float quadraticNorm(float *vector, float **matrix, int size)
{
    // quadNorm = vector*(matrix*vector'), [size] vector, [size x size] matrix
    float quadNorm = 0.0;
    float tmp_f;
    float *tmp_fp;
    for( int i=0; i<size; i++ )
    {
        tmp_f = 0.0;
        tmp_fp = matrix[i];
        for( int j=0; j<size; j++ )
        {
             tmp_f += tmp_fp[j]*vector[j];
        }
        quadNorm += vector[i]*tmp_f;
    }
    return quadNorm;
}


float getMinNegAbs(float *vector, int size)
{
    // minNegAbs = -max(abs(vector))
    float minNegAbs = -fabsf(vector[0]);
    float tmp_f;
    for( int i=1; i<size; i++ )
    {
        tmp_f = -fabsf(vector[i]);
        if( minNegAbs > tmp_f)
            minNegAbs = tmp_f;
    }
    return minNegAbs;
}


void getTStat(int *iiHeterozygous, int *iiHomozygous, int *iiMiss, int nHeterozygous,
    int nHomozygous, int nMiss, int nNonmiss, float **phenoMat, int nPheno, float *sumPheno,
    float *sumPheno2, float genoMean, float genoStd, float *tStat)
{
    // Fill t statistics array, tStat[i] = ri*sqrt((n - 2)/(1 - ri*ri)),
    // where ri is correlation between genotype vector and i-th phenotype vector.
    int i;
    float *phenoVec;
    float sumHetPheno, sumHomPheno, sumMissPheno, sumMissPheno2, phenoMean, phenoStd, corr;
    float tmp_f;
    float nNonmiss_f = (float)nNonmiss;

    for( int iPheno=0; iPheno<nPheno; iPheno++ )
    {
        phenoVec = phenoMat[iPheno];
        sumMissPheno = 0.0;
        sumMissPheno2 = 0.0;
        sumHetPheno = 0.0;
        sumHomPheno = 0.0;
        for( i=0; i<nMiss; i++ )
        {
            tmp_f = phenoVec[iiMiss[i]];
            sumMissPheno += tmp_f;
            sumMissPheno2 += tmp_f*tmp_f;
        }
        for( i=0; i<nHeterozygous; i++ )
            sumHetPheno += phenoVec[iiHeterozygous[i]];
        sumHetPheno *= 2.0f;
        for( i=0; i<nHomozygous; i++ )
            sumHomPheno += phenoVec[iiHomozygous[i]];
        phenoMean = (sumPheno[iPheno] - sumMissPheno)/nNonmiss_f;
        phenoStd = sqrtf((sumPheno2[iPheno] - sumMissPheno2)/nNonmiss_f - phenoMean*phenoMean);
        corr = ((sumHetPheno + sumHomPheno)/nNonmiss_f - phenoMean*genoMean)/(phenoStd*genoStd);
        tStat[iPheno] = corr*sqrtf((nNonmiss_f - 2.0f)/(1.0f - corr*corr));
    }
}


void partialPermutation(int *vector, int size, int n2perm, gsl_rng *r)
{
    // Permute n2perm first elements of the vector inplace
    long unsigned int upper_rand = (long unsigned int)size;
    int ind2swap, tmp_i;
    for( int i=0; i<n2perm; i++ )
    {
        ind2swap = (int)gsl_rng_uniform_int(r, upper_rand--) + i;
        tmp_i = vector[ind2swap];
        vector[ind2swap] = vector[i];
        vector[i] = tmp_i;
    }
}


void corrPhenoGeno(int nSnps, int nSamples, int nPheno, float **phenoMat,
    float *sumPheno, float *sumPheno2, float **invCovMat, unsigned char **bed, int nThreads,
    float *mostestStat, float *mostestStatPerm, float *minpStat, float *minpStatPerm)
{
    // phenoMat = [nPheno x nSamples] matrix.
    // invCovMat = [nPheno x nPheno] matrix.
    // bed = [nSnps x N] matrix, chunk of plink bed file, N = nSamples/4 (rounded up)
    // mostestStat, mostestStatPerm, minpStat, minpStatPerm = [nSnps] arrays to fill

    // Configure OMP
    // https://stackoverflow.com/questions/11095309/openmp-set-num-threads-is-not-working
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(nThreads); 
    printf("Max number of threads = %i\n", nThreads);

    int i, tid;
    const float SQRT2 = 1.4142135623730951f; // sqrt(2)
    signed char *byteMap = (signed char *)malloc(256*4*sizeof(signed char));
    getByteMap(byteMap);

    float *tStat[nThreads];
    int *iiHeterozygous[nThreads];
    int *iiHomozygous[nThreads];
    int *iiMiss[nThreads];
    int *sampleIndices[nThreads];
    int nHeterozygous, nHomozygous, nMiss, nNonmiss;
    float genoMean, genoStd;
    gsl_rng *rng[nThreads];

    for( i=0; i<nThreads; i++ )
    {
        tStat[i] = (float *)malloc((size_t)nPheno*sizeof(float));
        iiHeterozygous[i] = (int *)malloc((size_t)nSamples*sizeof(int)); // indices of 2
        iiHomozygous[i] = (int *)malloc((size_t)nSamples*sizeof(int));   // indices of 1
        iiMiss[i] = (int *)malloc((size_t)nSamples*sizeof(int));   // indices of missing values
        sampleIndices[i] = (int *)malloc((size_t)nSamples*sizeof(int));
        for( int j=0; j<nSamples; j++ )
                sampleIndices[i][j] = j;
        rng[i] = gsl_rng_alloc(gsl_rng_taus);
        // 0 seed sets to default seed, which is the same as 1 for gsl_rng_taus, so should start seeding with 1 with multithreading
        gsl_rng_set(rng[i], (long unsigned int)i+1);
    }

    // parallelize the following loop with OMP
    #pragma omp parallel for default(shared) private(nHeterozygous, nHomozygous, nMiss, nNonmiss, genoMean, genoStd, tid) schedule(dynamic)
    for( int iSnp=0; iSnp<nSnps; iSnp++ )
    {
        tid = omp_get_thread_num();
        getHetHomMissInd(bed[iSnp], nSamples, iiHeterozygous[tid], iiHomozygous[tid], iiMiss[tid],
            &nHeterozygous, &nHomozygous, &nMiss, byteMap);
        nNonmiss = nSamples - nMiss;
        genoMean = (float)(2*nHeterozygous + nHomozygous)/(float)nNonmiss;
        genoStd = sqrtf((float)(4*nHeterozygous + nHomozygous)/(float)nNonmiss - genoMean*genoMean);

        // for original genotypes
        getTStat(iiHeterozygous[tid], iiHomozygous[tid], iiMiss[tid], nHeterozygous, nHomozygous, nMiss,
            nNonmiss, phenoMat, nPheno, sumPheno, sumPheno2, genoMean, genoStd, tStat[tid]);
        mostestStat[iSnp] = quadraticNorm(tStat[tid], invCovMat, nPheno);
        minpStat[iSnp] = 1.0f + erff(getMinNegAbs(tStat[tid], nPheno)/SQRT2); // 2*norm.cdf(x)

        // for shuffled genotypes
        // we need to permute (select randomly) only positions of 2, 1 and missing genotypes
        partialPermutation(sampleIndices[tid], nSamples, nHeterozygous+nHomozygous+nMiss, rng[tid]);
        getTStat(sampleIndices[tid], &sampleIndices[tid][nHeterozygous], &sampleIndices[tid][nHeterozygous+nHomozygous],
            nHeterozygous, nHomozygous, nMiss, nNonmiss, phenoMat, nPheno, sumPheno, sumPheno2,
            genoMean, genoStd, tStat[tid]);
        mostestStatPerm[iSnp] = quadraticNorm(tStat[tid], invCovMat, nPheno);
        minpStatPerm[iSnp] = 1.0f + erff(getMinNegAbs(tStat[tid], nPheno)/SQRT2);
    }

    for( i=0; i<nThreads; i++ )
    {
        free(rng[i]);
        free(sampleIndices[i]);
        free(iiMiss[i]);
        free(iiHomozygous[i]);
        free(iiHeterozygous[i]);
        free(tStat[i]);
    }
    free(byteMap);
}



void test()
{
    #define N_SAMPLES 7
    #define N_BYTE 2 // == nSamples/4 rounded up
    #define N_SNPS 3
    #define N_PHENO 2
    #define I_GENO 0 // index of genotype to use in function, where only a single genotype is required

    unsigned char bedArr[N_SNPS][N_BYTE] = { {3, 137},
                                             {198, 42},
                                             {237, 9} };

    float phenoMatArr[N_PHENO][N_SAMPLES] = { {1.21f, 0.41f, 0.87f, 1.02f, 0.74f, 1.11f, 0.65f},
                                              {1.14f, 0.62f, 0.91f, 1.00f, 0.68f, 1.07f, 0.87f} };

    float invCovMatArr[N_PHENO][N_PHENO] = { {7.04376822f, -6.52463811f},
                                             {-6.52463811f,  7.04376822f} };


    int i, j;
    int nSamples = N_SAMPLES;
    int nByte = N_BYTE;
    int nSnps = N_SNPS;
    int nPheno = N_PHENO;

    signed char *byteMap = (signed char *)malloc(256*4*sizeof(signed char));
    getByteMap(byteMap);

    unsigned char *bedGeno = (unsigned char *)malloc((size_t)nByte*sizeof(unsigned char));
    printf("byteMap:\n");
    for( i=0; i<nByte; i++ )
    {
        bedGeno[i] = bedArr[I_GENO][i];
        printf("Byte %i:  ", bedArr[I_GENO][i]);
        for ( j=0; j<4; j++)
            printf("%i  ", byteMap[4*bedArr[I_GENO][i]+j]);
        printf("\n");
    }
    int *iiHeterozygous = (int *)malloc((size_t)nSamples*sizeof(int));
    int *iiHomozygous = (int *)malloc((size_t)nSamples*sizeof(int));
    int *iiMiss = (int *)malloc((size_t)nSamples*sizeof(int));
    int nHeterozygous, nHomozygous, nMiss, nNonmiss;

    getHetHomMissInd(bedGeno, nSamples, iiHeterozygous, iiHomozygous, iiMiss, &nHeterozygous, &nHomozygous, &nMiss, byteMap);
    nNonmiss = nSamples - nMiss;

    printf("nHet = %i\n", nHeterozygous);
    printf("nHom = %i\n", nHomozygous);
    printf("nNonmiss = %i\n", nNonmiss);

    printf("iiHeterozygous:  ");
    for( i=0; i<nHeterozygous; i++)
    {
        printf("%i   ", iiHeterozygous[i]);
    }
    printf("\n");
    printf("iiHomozygous:  ");
    for( i=0; i<nHomozygous; i++)
    {
        printf("%i   ", iiHomozygous[i]);
    }
    printf("\n");
    printf("iiMiss:  ");
    for( i=0; i<nMiss; i++)
    {
        printf("%i   ", iiMiss[i]);
    }
    printf("\n");

    float **phenoMat = (float **)malloc((size_t)nPheno*sizeof(float *));
    float *sumPheno = (float *)malloc((size_t)nPheno*sizeof(float));
    float *sumPheno2 = (float *)malloc((size_t)nPheno*sizeof(float));
    float **invCovMat = (float **)malloc((size_t)nPheno*sizeof(float *));
    for( i=0; i<nPheno; i++ )
    {
        phenoMat[i] = (float *)malloc((size_t)nSamples*sizeof(float));
        sumPheno[i] = 0.;
        sumPheno2[i] = 0.;
        for( j=0; j<nSamples; j++ )
        {
            phenoMat[i][j] = phenoMatArr[i][j];
            sumPheno[i] += phenoMat[i][j];
            sumPheno2[i] += phenoMat[i][j]*phenoMat[i][j];
        }
        invCovMat[i] = (float *)malloc((size_t)nPheno*sizeof(float));
        for( j=0; j<nPheno; j++ )
        {
            invCovMat[i][j] = invCovMatArr[i][j];
        }
    }
    float genoMean = (float)(2*nHeterozygous + nHomozygous)/(float)nNonmiss;
    float genoStd = sqrtf((float)(4*nHeterozygous + nHomozygous)/(float)nNonmiss - genoMean*genoMean);
    float *tStat = (float *)malloc((size_t)nPheno*sizeof(float));

    getTStat(iiHeterozygous, iiHomozygous, iiMiss, nHeterozygous, nHomozygous, nMiss,
        nNonmiss, phenoMat, nPheno, sumPheno, sumPheno2, genoMean, genoStd, tStat);

    unsigned char **bed = (unsigned char **)malloc((size_t)nSnps*sizeof(unsigned char *));
    for( i=0; i<nSnps; i++ )
    {
        bed[i] = (unsigned char *)malloc((size_t)nByte*sizeof(unsigned char));
        for( j=0; j<nByte; j++ )
            bed[i][j] = bedArr[i][j];
    }

    int nThreads = (int)omp_get_max_threads();
    
    float *mostestStat = (float *)malloc((size_t)nSnps*sizeof(float));
    float *mostestStatPerm = (float *)malloc((size_t)nSnps*sizeof(float));
    float *minpStat = (float *)malloc((size_t)nSnps*sizeof(float));
    float *minpStatPerm = (float *)malloc((size_t)nSnps*sizeof(float));
    corrPhenoGeno(nSnps, nSamples, nPheno, phenoMat, sumPheno, sumPheno2, invCovMat, bed, nThreads,
        mostestStat, mostestStatPerm, minpStat, minpStatPerm);

    for( i=0; i<nSnps; i++ )
    {
        printf("%f  ", mostestStat[i]);
    }
    printf("\n");
    for( i=0; i<nSnps; i++ )
    {
        printf("%f  ", minpStat[i]);
    }
    printf("\n");
    for( i=0; i<nSnps; i++ )
    {
        printf("%f  ", mostestStatPerm[i]);
    }
    printf("\n");
    for( i=0; i<nSnps; i++ )
    {
        printf("%f  ", minpStatPerm[i]);
    }
    printf("\n");

    // free memory
    free(minpStatPerm);
    free(minpStat);
    free(mostestStatPerm);
    free(mostestStat);
    for( i=0; i<nSnps; i++)
    {
        free(bed[i]);
    }
    free(bed);
    free(tStat);
    free(sumPheno);
    free(sumPheno2);
    for( i=0; i<nPheno; i++)
    {
        free(phenoMat[i]);
        free(invCovMat[i]);
    }
    free(phenoMat);
    free(invCovMat);
    free(iiMiss);
    free(iiHomozygous);
    free(iiHeterozygous);
    free(bedGeno);
    free(byteMap);
}


int main() {

    test();

    printf("Done.\n");
    return 0;
}
