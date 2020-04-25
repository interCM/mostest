import sys
import numpy as np
import pandas as pd
from collections import namedtuple
import argparse
import ctypes
import multiprocessing
from scipy.io import savemat

# Arg parsing
example_text =  """Example:
python mostest.py --bfile chr21 --pheno pheno.txt --num-eigval-to-regularize 1 --out test2"""

def parse_args(args):
    parser = argparse.ArgumentParser(description="Produce mostest and minp statistics.", epilog=example_text)
    
    parser.add_argument("--bfile", required=True, help="Prefix of Plink bim/bed/fam file.")
    parser.add_argument("--pheno", required=True, 
        help="Phenotypes file. Must be tab-separated and have a header line. Row order must correspond to bfile.")
    parser.add_argument("--out", required=True, help="Output file prefix.")
    parser.add_argument("--num-eigval-to-regularize", type=int, default=0,
        help="Number of smallest eigen values to regularize.")
    parser.add_argument("--no-csv", action="store_true", help="Do not save csv.")
    parser.add_argument("--save-npz", action="store_true", help="Save as npz.")
    parser.add_argument("--save-mat", action="store_true", help="Save as mat.")

    return parser.parse_args(args)
# End Arg parsing


# Load C lib
_mostlib = np.ctypeslib.load_library("mostlib.so", "./")

_floatP = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C')
_genericPP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C') # https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
_corrPhenoGeno = _mostlib.corrPhenoGeno
_corrPhenoGeno.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _genericPP,
    _floatP, _floatP, _genericPP, _genericPP, ctypes.c_int, _floatP, _floatP, _floatP, _floatP] 
_corrPhenoGeno.restype = None 


def makeContiguous(arr):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr, dtype=arr.dtype)
    return arr

def corrPhenoGeno(phenoMat, invCovMat, bed, nThreads=None):
    if nThreads is None:
        nThreads = multiprocessing.cpu_count()
    
    phenoMat = makeContiguous(phenoMat).astype(np.float32)
    invCovMat = makeContiguous(invCovMat).astype(np.float32)
    bed = makeContiguous(bed)

    nSnps = bed.shape[0]
    nSamples = phenoMat.shape[1]
    nPheno = phenoMat.shape[0]
    sumPheno = phenoMat.sum(axis=1)
    sumPheno2 = np.sum(phenoMat**2, axis=1)
    mostestStat = np.empty(nSnps, dtype=np.float32)
    mostestStatPerm = np.empty(nSnps, dtype=np.float32)
    minpStat = np.empty(nSnps, dtype=np.float32)
    minpStatPerm = np.empty(nSnps, dtype=np.float32)

    #TODO: replace 2D **arrays in mostest.c with 1D *arrays and updated the following declarations
    phenoMat_pp = (phenoMat.ctypes.data + np.arange(phenoMat.shape[0])*phenoMat.strides[0]).astype(np.uintp)
    invCovMat_pp = (invCovMat.ctypes.data + np.arange(invCovMat.shape[0])*invCovMat.strides[0]).astype(np.uintp)
    bed_pp = (bed.ctypes.data + np.arange(bed.shape[0])*bed.strides[0]).astype(np.uintp) 

    _corrPhenoGeno(nSnps, nSamples, nPheno, phenoMat_pp, sumPheno, sumPheno2,
         invCovMat_pp, bed_pp, nThreads, mostestStat, mostestStatPerm, minpStat, minpStatPerm)

    return mostestStat, mostestStatPerm, minpStat, minpStatPerm
# End Load C lib


# Plink utils
Plink = namedtuple("Plink", ["iid", "snp", "chr", "bp", "a1", "a2", "n_samples", "n_snps", "bed"])

def read_plink(bfile):
    bim_file = f"{bfile}.bim"
    bim = pd.read_csv(bim_file, sep='\t', header=None, usecols=[0,1,3,4,5], names=["CHR","SNP","BP","A1","A2"])
    fam_file = f"{bfile}.fam"
    fam = pd.read_csv(fam_file, delim_whitespace=True, header=None, usecols=[1], names=["IID"])
    
    bed_file = f"{bfile}.bed"
    magic_bits = np.fromfile(bed_file,count=3,dtype=np.uint8) # read whole bed file at once
    if (magic_bits != [108,27,1]).any():
        # check magic bits
        # [108,27,1] are integers corresponding to bytes([0x6c, 0x1b, 0x01])
        raise ValueError(f"{bed_file} file is not a valid bed file!")
    n_snps = len(bim)
    n_samples = len(fam)
    n_cols = n_samples//4
    if 4*n_cols != n_samples:
        n_cols += 1
    bed = np.memmap(bed_file, dtype=np.uint8, offset=3, mode='r', shape=(n_snps,n_cols))
    return Plink(iid=fam.IID.values, snp=bim.SNP.values, chr=bim.CHR.values, bp=bim.BP.values,
                 a1=bim.A1.values, a2=bim.A2.values, n_samples=n_samples, n_snps=n_snps, bed=bed)
# End Plink utils


def run_gwas(pheno_mat, plink, inv_C0reg, snp_chunk_size=10000):
    n_snps = plink.n_snps
    mosttest_stat = np.empty(plink.n_snps, dtype=np.float32)
    mosttest_stat_shuf = np.empty(plink.n_snps, dtype=np.float32)
    minp_stat = np.empty(plink.n_snps, dtype=np.float32)
    minp_stat_shuf = np.empty(plink.n_snps, dtype=np.float32)

    for snp_chunk_start in range(0, n_snps, snp_chunk_size):
        snp_chunk_end = min(n_snps, snp_chunk_start + snp_chunk_size)
        
        bed_chunk = plink.bed[snp_chunk_start:snp_chunk_end]
        most, most_perm, minp, minp_perm = corrPhenoGeno(pheno_mat, inv_C0reg, plink.bed[snp_chunk_start:snp_chunk_end])

        mosttest_stat[snp_chunk_start:snp_chunk_end] = most
        mosttest_stat_shuf[snp_chunk_start:snp_chunk_end] = most_perm
        minp_stat[snp_chunk_start:snp_chunk_end] = minp
        minp_stat_shuf[snp_chunk_start:snp_chunk_end] = minp_perm

        print(f"{snp_chunk_end} variants processed ({100*snp_chunk_end/n_snps:.2f}%)")
    return mosttest_stat, mosttest_stat_shuf, minp_stat, minp_stat_shuf


def main_most(args):
    print(f"Reading {args.bfile}")
    plink = read_plink(args.bfile)
    print(f"    {plink.n_samples} samples")
    print(f"    {plink.n_snps} variants")

    print(f"Reading {args.pheno}")
    pheno_df = pd.read_csv(args.pheno, sep='\t')
    print(f"    {pheno_df.shape[0]} samples")
    print(f"    {pheno_df.shape[1]} phenotypes")

    # remove phenotypes with all constant values
    tmp_pheno_mat = pheno_df.to_numpy()
    i_col_not_const = (tmp_pheno_mat[0] != tmp_pheno_mat[1:]).any(axis=0)
    pheno_df = pheno_df.loc[:,i_col_not_const]
    print(f"    {pheno_df.shape[1]} non-constant phenotypes")

    pheno_mat = pheno_df.values.T # the code currently is designed to have phenotypes in rows and samples in columns

    pheno_corr_mat = np.corrcoef(pheno_mat, rowvar=True)
    #TODO: check whether args.num_eigval_to_regularize is less than the number of Phenotypes
    print(f"Regularizing {args.num_eigval_to_regularize} eigen values.")
    if args.num_eigval_to_regularize == 0:
        pheno_corr_mat_reg = pheno_corr_mat
    else:
        U, S, _ = np.linalg.svd(pheno_corr_mat, hermitian=True)
        max_lambda = S[-args.num_eigval_to_regularize - 1]
        S[-args.num_eigval_to_regularize:] = max_lambda
        pheno_corr_mat_reg = (U * S[..., None, :]) @ U.T
    inv_C0reg = np.linalg.inv(pheno_corr_mat_reg)

    print("Running correlation analysis.")
    mosttest_stat, mosttest_stat_shuf, minp_stat, minp_stat_shuf = run_gwas(pheno_mat, plink, inv_C0reg, snp_chunk_size=50000)

    if not args.no_csv:
        out_df = pd.DataFrame({"most_orig":mosttest_stat, "minp_orig":minp_stat,
            "most_perm":mosttest_stat_shuf, "minp_perm":minp_stat_shuf})
        if not args.out.endswith(".csv"):
            fname = f"{args.out}.csv"
        out_df.to_csv(fname, index=False, sep='\t')
        print(f"Results saved to {fname}")
    if args.save_npz:
        if not args.out.endswith(".npz"):
            fname = f"{args.out}.npz"
        np.savez_compressed(fname, most_orig=mosttest_stat, minp_orig=minp_stat,
            most_perm=mosttest_stat_shuf, minp_perm=minp_stat_shuf)
        print(f"Results saved to {fname}")
    if args.save_mat:
        if not args.out.endswith(".mat"):
            fname = f"{args.out}.mat"
        mdict = {"most_orig":mosttest_stat, "minp_orig":minp_stat,
            "most_perm":mosttest_stat_shuf, "minp_perm":minp_stat_shuf}
        savemat(fname, mdict)
        print(f"Results saved to {fname}")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main_most(args)
