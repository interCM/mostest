import ctypes
import numpy as np

_mostlib = np.ctypeslib.load_library("mostlib.so", "./")

_floatP = ctypes.POINTER(ctypes.c_float)
_genericPP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C') # https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
_corrPhenoGeno = _mostlib.corrPhenoGeno
_corrPhenoGeno.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _genericPP,
    _floatP, _floatP, _genericPP, _genericPP, _floatP, _floatP, _floatP, _floatP] 
_corrPhenoGeno.restype = None 

def makeContiguous(arr):
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr, dtype= arr.dtype)
    return arr

def corrPhenoGeno(phenoMat, invCovMat, bed):
    
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

    _corrPhenoGeno(nSnps, nSamples, nPheno, phenoMat_pp, sumPheno_p, sumPheno2_p,
        invCovMat_pp, bed_pp, mostestStat_p, mostestStatPerm_p, minpStat_p, minpStatPerm_p)

    return mostestStat, mostestStatPerm, minpStat, minpStatPerm

if __name__ == "__main__":
    bed = np.array([[3, 137],[198, 42],[237, 9]], dtype=np.uint8)
    phenoMat = np.array([[1.21, 0.41, 0.87, 1.02, 0.74, 1.11, 0.65], [1.14, 0.62, 0.91, 1.00, 0.68, 1.07, 0.87]], dtype=np.float32)
    invCovMat = np.array([[7.04376822, -6.52463811], [-6.52463811,  7.04376822]], dtype=np.float32)

    mostestStat, mostestStatPerm, minpStat, minpStatPerm = corrPhenoGeno(phenoMat, invCovMat, bed)
    print(mostestStat)
    print(minpStat)
    print(mostestStatPerm)
    print(minpStatPerm)
