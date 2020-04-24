import ctypes
import numpy as np
import multiprocessing
import pandas as pd
from scipy.stats import pearsonr

_mostlib = np.ctypeslib.load_library("mostlib.so", "./")

_floatP = ctypes.POINTER(ctypes.c_float)
_genericPP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C') # https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
_corrPhenoGeno = _mostlib.corrPhenoGeno
_corrPhenoGeno.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _genericPP,
    _floatP, _floatP, _genericPP, _genericPP, ctypes.c_int, _floatP, _floatP, _floatP, _floatP] 
_corrPhenoGeno.restype = None 

def makeContiguous(arr):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr, dtype= arr.dtype)
    return arr

def corrPhenoGeno(phenoMat, invCovMat, bed, nThreads=None):
    if nThreads is None:
        nThreads = multiprocessing.cpu_count()
    
    nSnps = bed.shape[0]
    sumPheno = phenoMat.sum(axis=1)
    sumPheno2 = np.sum(phenoMat**2, axis=1)
    mostestStat = np.empty(nSnps, dtype=np.float32)
    mostestStatPerm = np.empty(nSnps, dtype=np.float32)
    minpStat = np.empty(nSnps, dtype=np.float32)
    minpStatPerm = np.empty(nSnps, dtype=np.float32)

    phenoMat = makeContiguous(phenoMat)
    invCovMatPP = makeContiguous(invCovMat)
    bed = makeContiguous(bed)

    nSnps = ctypes.c_int(nSnps)
    nSamples = ctypes.c_int(phenoMat.shape[1])
    nPheno = ctypes.c_int(phenoMat.shape[0])
    phenoMat_pp = (phenoMat.ctypes.data + np.arange(phenoMat.shape[0])*phenoMat.strides[0]).astype(np.uintp)
    invCovMat_pp = (invCovMat.ctypes.data + np.arange(invCovMat.shape[0])*invCovMat.strides[0]).astype(np.uintp)
    bed_pp = (bed.ctypes.data + np.arange(bed.shape[0])*bed.strides[0]).astype(np.uintp) 
    
    sumPheno_p = sumPheno.ctypes.data_as(_floatP)
    sumPheno2_p = sumPheno2.ctypes.data_as(_floatP)
    mostestStat_p = mostestStat.ctypes.data_as(_floatP)
    mostestStatPerm_p = mostestStatPerm.ctypes.data_as(_floatP)
    minpStat_p = minpStat.ctypes.data_as(_floatP)
    minpStatPerm_p = minpStatPerm.ctypes.data_as(_floatP)

    nThreads = ctypes.c_int(nThreads)

    _corrPhenoGeno(nSnps, nSamples, nPheno, phenoMat_pp, sumPheno_p, sumPheno2_p,
        invCovMat_pp, bed_pp, nThreads, mostestStat_p, mostestStatPerm_p, minpStat_p, minpStatPerm_p)

    return mostestStat, mostestStatPerm, minpStat, minpStatPerm

if __name__ == "__main__":

    if False:
        bed = np.array([[3, 137],[198, 42],[237, 9]], dtype=np.uint8)
        phenoMat = np.array([[1.21, 0.41, 0.87, 1.02, 0.74, 1.11, 0.65], [1.14, 0.62, 0.91, 1.00, 0.68, 1.07, 0.87]], dtype=np.float32)
        invCovMat = np.array([[7.04376822, -6.52463811], [-6.52463811,  7.04376822]], dtype=np.float32)

        mostestStat, mostestStatPerm, minpStat, minpStatPerm = corrPhenoGeno(phenoMat, invCovMat, bed)
        print(mostestStat)
        print(minpStat)
        print(mostestStatPerm)
        print(minpStatPerm)

    if True:
        N2run = 200000
        bed_file = "chr21.bed"
        pheno_file = "pheno.txt"
        n_snps = 149454
        n_samples = 10000
        n_cols = n_samples//4
        if 4*n_cols != n_samples:
            n_cols += 1

        bed = np.memmap(bed_file, dtype=np.uint8, offset=3, mode='r', shape=(n_snps,n_cols))
        pheno_df = pd.read_csv(pheno_file, sep='\t')
        phenoMat = pheno_df.values.T # the code currently is designed to have phenotypes in rows and samples in columns

        phenoCorrMat = np.corrcoef(phenoMat, rowvar=True)
        invCovMat = np.linalg.inv(phenoCorrMat)

        phenoMat = phenoMat.astype(np.float32)
        invCovMat = invCovMat.astype(np.float32)

        mostestStat, mostestStatPerm, minpStat, minpStatPerm = corrPhenoGeno(phenoMat, invCovMat, bed[:N2run])

        # compare with test
        conpare_df = pd.read_csv("test.csv", sep='\t', nrows=N2run)
        mostest_r, mostest_p = pearsonr(conpare_df.most_orig, mostestStat)
        mostest_max_dist = np.max(np.abs(conpare_df.most_orig - mostestStat))
        minp_r, minp_p = pearsonr(conpare_df.minp_orig, minpStat)
        minp_max_dist = np.max(np.abs(conpare_df.minp_orig - minpStat))

        most_perm_mean = mostestStatPerm.mean()
        most_perm_mean_ref = conpare_df.most_perm.mean()
        minp_perm_mean = minpStatPerm.mean()
        minp_perm_mean_ref = conpare_df.minp_perm.mean()
        most_perm_std = mostestStatPerm.std()
        most_perm_std_ref = conpare_df.most_perm.std()
        minp_perm_std = minpStatPerm.std()
        minp_perm_std_ref = conpare_df.minp_perm.std()

        print(f"N = {N2run}")
        print("Orig:")
        print(f"r(most) = {mostest_r:.6f}, p = {mostest_p}")
        print(f"maxDist(most) = {mostest_max_dist:.6f}")
        print(f"r(minp) = {minp_r:.6f}, p = {minp_p}")
        print(f"maxDist(minp) = {minp_max_dist:.6f}")
        print("Perm:")
        print(f"most cur: mean = {most_perm_mean:.3f}, std = {most_perm_std:.3f}")
        print(f"most ref: mean = {most_perm_mean_ref:.3f}, std = {most_perm_std_ref:.3f}")
        print(f"minp cur: mean = {minp_perm_mean:.3f}, std = {minp_perm_std:.3f}")
        print(f"minp ref: mean = {minp_perm_mean_ref:.3f}, std = {minp_perm_std_ref:.3f}")

