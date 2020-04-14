import sys
import math
import numpy as np
import pandas as pd
from collections import namedtuple
import numba
import argparse


example_text =  """Example:
python mostest.py --bfile chr21 --pheno pheno.txt --out test"""

def parse_args(args):
    parser = argparse.ArgumentParser(description="Produce mostest and minp statistics.", epilog=example_text)
    
    parser.add_argument("--bfile", required=True, help="Prefix of Plink bim/bed/fam file.")
    parser.add_argument("--pheno", required=True, 
        help="Phenotypes file. Must be tab-separated and have a header line. Row order must correspond to bfile.")
    parser.add_argument("--out", required=True, help="Output file prefix.")
    parser.add_argument("--no-npz", action="store_true", help="Do not save npz.")
    parser.add_argument("--no-csv", action="store_true", help="Do not save csv.")

    return parser.parse_args(args)


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


@numba.jit(nopython=True, nogil=True, cache=True)
def get_byte_map():
    """
    Construct mapping between bytes 0..255 and 4-element arrays of a1 genotypes
    from plink bed file.
    Return 256 x 4 array A, where A[i] = [a1, a2, a3, a4], each ai from [2, -1, 1, 0].
    """
    genotype_codes = np.array([2, -1, 1, 0],dtype=np.int8)
    byte_map = np.empty((256,4), dtype=np.int8)
    for b in range(256):
        for i in range(4):
            byte_map[b,i] = genotype_codes[(b >> 2*i) & 3]
    return byte_map



@numba.jit(nopython=True, nogil=True, cache=True)
def get_geno_idx(i_geno, bed, n_samples, geno_idx, ii1_tmp, ii2_tmp, byte_map):
    # Fill geno_idx array.
    # geno_idx = [n_nonmiss, n2, n1, {ii2}, {ii1}, {empty}]
    # len(geno_idx) = 3 + n_samples
    # len(ii2) = n2, {ii2} = [i20, i21, ...], i20 = index of the first occurance of 2 genotype
    # {empty} = empty array to fill remaining space
    # geno_idx = np.empty(3+n_sampels, dtype=np.int)
    # ii1_tmp, ii2_tmp = np.empty(n_samples, dtype=np.int), to fill indices of 2 and 1 genotypes correspondingly
    i = 0 # current sample index
    n2 = 0
    n1 = 0
    n_nonmiss = 0
    for b in bed[i_geno]:
        for g in byte_map[b]:
            if g != -1:
                n_nonmiss += 1
                if g == 2:
                    ii2_tmp[n2] = i
                    n2 += 1
                elif g == 1:
                    ii1_tmp[n1] = i
                    n1 += 1
            i += 1
            if i == n_samples:
                break
    geno_idx[0] = n_nonmiss
    geno_idx[1] = n2
    geno_idx[2] = n1
    geno_idx[3:3+n2] = ii2_tmp[:n2]
    geno_idx[3+n2:3+n2+n1] = ii1_tmp[:n1]


def run_gwas(pheno_mat, plink, inv_C0reg, snp_chunk_size=10000):
    pheno_mean = np.mean(pheno_mat, axis=1, dtype=np.float32)
    pheno_std = np.std(pheno_mat, axis=1, dtype=np.float32, ddof=0)
    n_snps = plink.n_snps

    inv_C0reg = inv_C0reg.astype(np.float32)
    pheno_mat = pheno_mat.astype(np.float32)

    mosttest_stat = np.empty(plink.n_snps, dtype=np.float32)
    mosttest_stat_shuf = np.empty(plink.n_snps, dtype=np.float32)
    minp_stat = np.empty(plink.n_snps, dtype=np.float32)
    minp_stat_shuf = np.empty(plink.n_snps, dtype=np.float32)

    for snp_chunk_start in range(0, n_snps, snp_chunk_size):
        snp_chunk_end = min(n_snps, snp_chunk_start + snp_chunk_size)
        actual_chunk_size = snp_chunk_end - snp_chunk_start

        mosttest_stat_chunk = np.empty(actual_chunk_size, dtype=np.float32)
        mosttest_stat_shuf_chunk = np.empty(actual_chunk_size, dtype=np.float32)
        minp_stat_chunk = np.empty(actual_chunk_size, dtype=np.float32)
        minp_stat_shuf_chunk = np.empty(actual_chunk_size, dtype=np.float32)

        bed_chunk = plink.bed[snp_chunk_start:snp_chunk_end]
        chunk_gwas(pheno_mat, pheno_mean, pheno_std, inv_C0reg, bed_chunk,
               mosttest_stat_chunk, mosttest_stat_shuf_chunk, minp_stat_chunk, minp_stat_shuf_chunk)

        mosttest_stat[snp_chunk_start:snp_chunk_end] = mosttest_stat_chunk
        mosttest_stat_shuf[snp_chunk_start:snp_chunk_end] = mosttest_stat_shuf_chunk
        minp_stat[snp_chunk_start:snp_chunk_end] = minp_stat_chunk
        minp_stat_shuf[snp_chunk_start:snp_chunk_end] = minp_stat_shuf_chunk

        print(f"{snp_chunk_end} variants processed ({100*snp_chunk_end/n_snps:.2f}%)")
    return mosttest_stat, mosttest_stat_shuf, minp_stat, minp_stat_shuf


@numba.jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def chunk_gwas(pheno_mat, pheno_mean, pheno_std, inv_C0reg, bed_chunk,
               mostest_stat_chunk, mostest_stat_shuf_chunk, minp_stat_chunk, minp_stat_shuf_chunk):
    byte_map = get_byte_map()
    n_pheno, n_samples = pheno_mat.shape
    n_snps_chunk = mostest_stat_chunk.shape[0]
    for geno_i in numba.prange(n_snps_chunk):
        #TODO: this can be improved when numba has a better control over threads (can get thread id)
        t_stat = np.empty(n_pheno, dtype=np.float32) # this array can be preallocated by thread
        geno_idx = np.empty(3+n_samples, dtype=np.int32) # this array can be preallocated by thread
        ii1_tmp = np.empty(n_samples, dtype=np.int32) # this array can be preallocated by thread
        ii2_tmp = np.empty(n_samples, dtype=np.int32) # this array can be preallocated by thread
        get_geno_idx(geno_i, bed_chunk, n_samples, geno_idx, ii1_tmp, ii2_tmp, byte_map)
        geno_mean = (geno_idx[1]*2 + geno_idx[2])/geno_idx[0]
        geno_std = math.sqrt((geno_idx[1]*4 + geno_idx[2])/geno_idx[0] - geno_mean*geno_mean)
        n_nonmiss = geno_idx[0]
        n2 = geno_idx[1]
        n1 = geno_idx[2]
        # for original genotypes
        get_t_stat(geno_idx[3:3+n2], geno_idx[3+n2:3+n2+n1], pheno_mat, n_pheno, pheno_mean, pheno_std,
                   geno_mean, geno_std, n_nonmiss, t_stat)
        mostest_stat_chunk[geno_i] = t_stat @ inv_C0reg @ t_stat 
        x = -np.max(np.abs(t_stat))
        minp_stat_chunk[geno_i] = 1.0 + math.erf(x/math.sqrt(2.0)) # 2*norm.cdf(x)

        # for shuffled genotypes
        # we need to shuffle (select randomly) only positions of 2 and 1 genotypes
        geno_idx_shuf = np.random.choice(n_samples,n1+n2,replace=False)
        get_t_stat(geno_idx_shuf[:n2], geno_idx_shuf[n2:], pheno_mat, n_pheno, pheno_mean, pheno_std,
                   geno_mean, geno_std, n_nonmiss, t_stat)
        mostest_stat_shuf_chunk[geno_i] = t_stat @ inv_C0reg @ t_stat
        x = -np.max(np.abs(t_stat))
        minp_stat_shuf_chunk[geno_i] = 1.0 + math.erf(x/math.sqrt(2.0)) # 2*norm.cdf(x)
    

@numba.jit(nopython=True, nogil=True, fastmath=True, cache=True)
def get_t_stat(idx2, idx1, pheno_mat, n_pheno, pheno_mean_arr, pheno_std_arr,
               geno_mean, geno_std, n_nonmiss, t_stat):
    # Fill t statistics array, t_stat[i] = ri*sqrt((n - 2)/(1 - ri*ri)),
    # where ri is correlation between genotype vector and i-th phenotype vector.
    for pheno_i in range(n_pheno):
        p2 = 2.0*pheno_mat[pheno_i][idx2].sum()
        p1 = pheno_mat[pheno_i][idx1].sum()
        pg_mean = (p2 + p1)/n_nonmiss
        pg_r = (pg_mean - pheno_mean_arr[pheno_i]*geno_mean)/(pheno_std_arr[pheno_i]*geno_std)
        pg_t = pg_r*math.sqrt((n_nonmiss - 2.0)/(1.0 - pg_r*pg_r))
        t_stat[pheno_i] = pg_t


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
    #TODO: add regularization here before inversion
    inv_C0reg = np.linalg.inv(pheno_corr_mat)

    print("Running correlation analysis.")
    mosttest_stat, mosttest_stat_shuf, minp_stat, minp_stat_shuf = run_gwas(pheno_mat, plink, inv_C0reg, snp_chunk_size=10000)

    if args.no_npz and args.no_csv:
        print("No results will be saved.")
    else:
        print("Saving results.")

    if not args.no_npz:
        if not args.out.endswith(".npz"):
            fname = f"{args.out}.npz"
        np.savez_compressed(fname, most_orig=mosttest_stat, minp_orig=minp_stat,
            most_perm=mosttest_stat_shuf, minp_perm=minp_stat_shuf)
        print(f"Results saved to {fname}")
    if not args.no_csv:
        out_df = pd.DataFrame({"most_orig":mosttest_stat, "minp_orig":minp_stat,
            "most_perm":mosttest_stat_shuf, "minp_perm":minp_stat_shuf})
        if not args.out.endswith(".csv"):
            fname = f"{args.out}.csv"
        out_df.to_csv(fname, index=False, sep='\t')
        print(f"Results saved to {fname}")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    main_most(args)
