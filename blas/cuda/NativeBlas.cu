#include "../NativeBlas.h"
#include <cublas_v2.h>


cublasStatus_t convertStatus(int status) {
    return reinterpret_cast<cublasStatus_t>(status);
}

cublasFillMode_t convertFillMode(int fillMode) {
    return reinterpret_cast<cublasStatus_t>(fillMode);
}

cublasDiagType_t convertDiag(int diag) {
    return reinterpret_cast<cublasDiagType_t>(diag);
}

cublasOperation_t convertTranspose(int op) {
    return reinterpret_cast<cublasOperation_t>(op);
}
cublasPointerMode_t convertPointerMode(int pointerMode) {
    return reinterpret_cast<cublasPointerMode_t>(pointerMode);
}

cublasSideMode_t convertSideMode(int sideMode) {
    return reinterpret_cast<cublasSideMode_t>(sideMode);
}


/*
 *
 * /* CUBLAS status type returns
typedef enum{
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
} cublasStatus_t;


typedef enum {
    CUBLAS_FILL_MODE_LOWER=0,
    CUBLAS_FILL_MODE_UPPER=1
} cublasFillMode_t;

typedef enum {
    CUBLAS_DIAG_NON_UNIT=0,
    CUBLAS_DIAG_UNIT=1
} cublasDiagType_t;

typedef enum {
    CUBLAS_SIDE_LEFT =0,
    CUBLAS_SIDE_RIGHT=1
} cublasSideMode_t;


typedef enum {
    CUBLAS_OP_N=0,
    CUBLAS_OP_T=1,
    CUBLAS_OP_C=2
} cublasOperation_t;


typedef enum {
    CUBLAS_POINTER_MODE_HOST   = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
    CUBLAS_ATOMICS_NOT_ALLOWED   = 0,
    CUBLAS_ATOMICS_ALLOWED       = 1
} cublasAtomicsMode_t;
 //Used by cublasSgemmEx
typedef enum
{
    CUBLAS_DATA_FLOAT    = 0,
    CUBLAS_DATA_DOUBLE   = 1,
    CUBLAS_DATA_HALF     = 2,
    CUBLAS_DATA_INT8     = 3
} cublasDataType_t;

 * */
/*
 * ======================================================
 * Level 1 BLAS functions
 * ======================================================
 */


/*
 * ------------------------------------------------------
 * DOT
 * ------------------------------------------------------
 */

float Nd4jBlas::sdsdot(long *extraParams,int N, float alpha,
                       long X, int incX,
                       long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    //nothing found?

}

double Nd4jBlas::dsdot(long *extraParams,int N,
                       long X, int incX,
                       long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    //nothing found?
}

double Nd4jBlas::ddot(long *extraParams,int N,
                      long X, int incX,
                      long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    double *resultPointer = reinterpret_cast<double *>(extraParams[1]);
    cublasDdot_v2(*handle,n,xPointer,incX,yPointer,incY,resultPointer);
    return 0.0;
}

float Nd4jBlas::sdot(long *extraParams,int N,
                     long X, int incX,
                     long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    float *resultPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasSdot_v2(*handle,n,xPointer,incX,yPointer,incY,resultPointer);
    return 0.0f;
}

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

float Nd4jBlas::snrm2(long *extraParams,int N, long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    float *resultPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasSnrm2_v2(*handle,n,xPointer,incX,resultPointer);
    return 0.0f;


}

double Nd4jBlas::dnrm2(long *extraParams,int N, long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    double *resultPointer = reinterpret_cast<double *>(extraParams[1]);
    cublasDnrm2_v2(*handle,n,xPointer,incX,resultPointer);
    return 0.0;
}

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

float Nd4jBlas::sasum(long *extraParams,int N, long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    float *resultPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasSasum_v2(*handle,n,xPointer,incX,resultPointer);
    return 0.0f;


}

double Nd4jBlas::dasum(long *extraParams,int N, long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    double *resultPointer = reinterpret_cast<double *>(extraParams[1]);
    cublasDasum_v2(*handle,n,xPointer,incX,resultPointer);
    return 0.0;

}

/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

int Nd4jBlas::isamax(long *extraParams,int N, long X, int incX){
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    float *resultPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasIsamax_v2(*handle,n,xPointer,incX,resultPointer);
    return 0;

}

int Nd4jBlas::idamax(long *extraParams,int N, long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    double *resultPointer = reinterpret_cast<double *>(extraParams[1]);
    cublasIdamax_v2(*handle,n,xPointer,incX,resultPointer);
    return 0;


}

/*
 * ======================================================
 * Level 1 BLAS procedures
 * ======================================================
 */

/*
 * ------------------------------------------------------
 * ROT
 * ------------------------------------------------------
 */

void Nd4jBlas::srot(long *extraParams,int N,
                    long X, int incX,
                    long Y, int incY,
                    float c, float s) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSrot_v2(*handle,n,xPointer,incX,yPointer,incY,&c,&s);
}

void Nd4jBlas::drot(long *extraParams,int N,
                    long X, int incX,
                    long Y, int incY,
                    double c, double s) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDrot_v2(*handle,n,xPointer,incX,yPointer,incY,&c,&s);
}

/*
 * ------------------------------------------------------
 * ROTG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotg(long *extraParams,long args) {
    float *argsPointers = reinterpret_cast<float *>(args);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSrotg_v2(*handle,&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3]);
}

void Nd4jBlas::drotg(long *extraParams,long args) {
    double *argsPointers = reinterpret_cast<double *>(args);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDrotg_v2(*handle,&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3]);

}

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotmg(long *extraParams,long args,
                      long P) {
    float *argsPointers = reinterpret_cast<float *>(args);
    float *pPointers = reinterpret_cast<float *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSrotmg_v2(*handle,&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3],pPointers);

}

void Nd4jBlas::drotmg(long *extraParams,long args,
                      long P) {
    double *argsPointers = reinterpret_cast<double *>(args);
    double *pPointers = reinterpret_cast<double *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDrotmg_v2(*handle,&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3],pPointers);

}

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

void Nd4jBlas::srotm(long *extraParams,int N,
                     long X, int incX,
                     long Y, int incY,
                     long P) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *pPointer = reinterpret_cast<float *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSrotm_v2(*handle,n,xPointer,incX,yPointer,incY,pPointer);

}

void Nd4jBlas::drotm(long *extraParams,int N,
                     long X, int incX,
                     long Y, int incY,
                     long P) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *pPointer = reinterpret_cast<double *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDrotm_v2(*handle,n,xPointer,incX,yPointer,incY,pPointer);

}

/*
 * ------------------------------------------------------
 * SWAP
 * ------------------------------------------------------
 */

void Nd4jBlas::sswap(long *extraParams,int N,
                     long X, int incX,
                     long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSswap_v2(*handle,n,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::dswap(long *extraParams,int N,
                     long X, int incX,
                     long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDswap_v2(*handle,n,xPointer,incX,yPointer,incY);

}

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

void Nd4jBlas::sscal(long *extraParams,int N, float alpha,
                     long X, int incX){
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSscal_v2(*handle,n,&alpha,xPointer,incX);

}

void Nd4jBlas::dscal(long *extraParams,int N, double alpha,
                     long X, int incX){
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDscal_v2(*handle,n,&alpha,xPointer,incX);
}

/*
 * ------------------------------------------------------
 * SCOPY
 * ------------------------------------------------------
 */

void Nd4jBlas::scopy(long *extraParams,int N,
                     long X, int incX,
                     long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasScopy_v2(*handle,n,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::dcopy(long *extraParams,int N,
                     long X, int incX,
                     long Y, int incY){
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDcopy_v2(*handle,n,xPointer,incX,yPointer,incY);

}

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

void Nd4jBlas::saxpy(long *extraParams,int N, float alpha,
                     long X, int incX,
                     long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSaxpy_v2(*handle,n,&alpha,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::daxpy(long *extraParams,int N, double alpha,
                     long X, int incX,
                     long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDaxpy_v2(*handle,n,&alpha,xPointer,incX,yPointer,incY);

}

/*
 * ======================================================
 * Level 2 BLAS procedures
 * ======================================================
 */


/*
 * ------------------------------------------------------
 * GEMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sgemv(long *extraParams,int Order, int TransA,
                     int M, int N,
                     float alpha,
                     long A, int lda,
                     long X, int incX,
                     float beta,
                     long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *aPointer = reinterpret_cast<float *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSgemv_v2(*handle,convertTranspose(TransA),m,n,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);
}

void Nd4jBlas::dgemv(long *extraParams,int Order, int TransA,
                     int M, int N,
                     double alpha,
                     long A, int lda,
                     long X, int incX,
                     double beta,
                     long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *aPointer = reinterpret_cast<double *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDgbmv_v2(*handle,convertTranspose(TransA),m,n,kl,ku,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * GBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sgbmv(long *extraParams,int Order, int TransA,
                     int M, int N,
                     int KL, int KU,
                     float alpha,
                     long A, int lda,
                     long X, int incX,
                     float beta,
                     long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSgbmv_v2(*handle,convertTranspose(TransA),m,n,kl,ku,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);
}

void Nd4jBlas::dgbmv(long *extraParams,int Order, int TransA,
                     int M, int N,
                     int KL, int KU,
                     double alpha,
                     long A, int lda,
                     long X, int incX,
                     double beta,
                     long Y, int incY) {

    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDgbmv_v2(*handle,convertTranspose(TransA),m,n,kl,ku,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymv(long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long A, int lda,
                     long X, int incX,
                     float beta,
                     long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSsymv_v2(*handle,convertUplo(Uplo),n,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);
}

void Nd4jBlas::dsymv(long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long A, int lda,
                     long X, int incX,
                     double beta,
                     long Y, int incY) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDsymv_v2(*handle,convertUplo(Uplo),n,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);

}

/*
 * ------------------------------------------------------
 * SBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssbmv(long *extraParams,int Order, int Uplo,
                     int N, int K,
                     float alpha,
                     long A, int lda,
                     long X, int incX,
                     float beta,
                     long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSsbmv_v2(*handle,convertUplo(Uplo),n,k,&alpha,aPointer,lda,xPointer,incx,&beta,yPointer,incY);
}

void Nd4jBlas::dsbmv(long *extraParams,int Order, int Uplo,
                     int N, int K,
                     double alpha,
                     long A, int lda,
                     long X, int incX,
                     double beta,
                     long Y, int incY){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDsbmv_v2(*handle,convertUplo(Uplo),n,k,&alpha,aPointer,lda,xPointer,incx,&beta,yPointer,incY);

}

/*
 * ------------------------------------------------------
 * SPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sspmv(long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long Ap,
                     long X, int incX,
                     float beta,
                     long Y, int incY){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSspmv_v2(*handle,convertUplo(Uplo),n,&alpha,apPointer,xPointer,incX,&beta,yPointer,incY);

}

void Nd4jBlas::dspmv(long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long Ap,
                     long X, int incX,
                     double beta,
                     long Y, int incY){
    double *apPointer = reinterpret_cast<double *>(p);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDspmv_v2(*handle,convertUplo(Uplo),n,&alpha,apPointer,xPointer,incX,&beta,yPointer,incY);


}

/*
 * ------------------------------------------------------
 * TRMV
 * ------------------------------------------------------
 */

void Nd4jBlas::strmv(long *extraParams,int Order, int Uplo, int TransA,
                     int Diag,
                     int N, float alpha,
                     long A, int lda,
                     long X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasStrmv_v2(*handle,convertFillMode(Uplo),convertTranspose(TransA),convertDiag(Diag),n,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtrmv(long *extraParams,int Order, int Uplo, int TransA,
                     int Diag,
                     int N, double alpha,
                     long A, int lda,
                     long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtrmv_v2(*handle,convertFillMode(Uplo),convertTranspose(TransA),convertDiag(Diag),n,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbmv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long A, int lda,
                     long X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasStbmv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,k,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtbmv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long A, int lda,
                     long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtbmv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,k,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpmv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long Ap,
                     long X, int incX){
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasStpmv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,apPointer,xPointer,incx);

}

void Nd4jBlas::dtpmv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long Ap,
                     long X, int incX) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtpmv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,apPointer,xPointer,incx);

}

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

void Nd4jBlas::strsv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long A, int lda,
                     long X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasStrsv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtrsv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long A, int lda,
                     long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtrsv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbsv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long A, int lda,
                     long X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasStbsv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,k,aPointer,lda,xPointer,incX);

}

void Nd4jBlas::dtbsv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long A, int lda,
                     long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtbsv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,k,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpsv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long Ap,
                     long X, int incX){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasStpsv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,apPointer,xPointer,incX);
}

void Nd4jBlas::dtpsv(long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long Ap,
                     long X, int incX){
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtpsv_v2(*handle,convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),n,apPointer,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

void Nd4jBlas::sger(long *extraParams,int Order,
                    int M, int N,
                    float alpha,
                    long X, int incX,
                    long Y, int incY,
                    long A, int lda){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSger_v2(*handle,m,n,&alpha,xPointer,incx,yPointer,incY,aPointer,lda);

}

void Nd4jBlas::dger(long *extraParams,int Order,
                    int M, int N,
                    double alpha,
                    long X, int incX,
                    long Y, int incY,
                    long A, int lda){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDger_v2(*handle,m,n,&alpha,xPointer,incx,yPointer,incY,aPointer,lda);

}

/*
 * ------------------------------------------------------
 * SYR
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr(long *extraParams,int Order, int Uplo,
                    int N,
                    float alpha,
                    long X, int incX,
                    long A, int lda){
    float *xPointer = reinterpret_cast<float *>(X);
    float *aPointer = reinterpret_cast<float *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSsyr_v2(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,aPointer,lda);
}

void Nd4jBlas::dsyr(long *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    long X, int incX,
                    long A, int lda){
    double *xPointer = reinterpret_cast<double *>(X);
    double *aPointer = reinterpret_cast<double *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDsyr_v2(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,aPointer,lda);

}

/*
 * ------------------------------------------------------
 * SPR
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr(long *extraParams,int Order, int Uplo,
                    int N,
                    float alpha,
                    long X, int incX,
                    long Ap){
    float *xPointer = reinterpret_cast<float *>(X);
    float *apPointer = reinterpret_cast<float *>(Ap);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSspr(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,apPointer);
}

void Nd4jBlas::dspr(long *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    long X, int incX,
                    long Ap){
    double *xPointer = reinterpret_cast<double *>(X);
    double *apPointer = reinterpret_cast<double *>(Ap);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDspr(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,apPointer);

}

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2(long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long X, int incX,
                     long Y, int incY,
                     long A, int lda){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSsyr2_v2(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,yPointer,incY,aPointer,lda);

}

void Nd4jBlas::dsyr2(long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long X, int incX,
                     long Y, int incY,
                     long A, int lda){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDsyr2_v2(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,yPointer,incY,aPointer,lda);

}

/*
 * ------------------------------------------------------
 * SPR2
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr2(long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long X, int incX,
                     long Y, int incY,
                     long Ap){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSspr2_v2(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,yPointer,incY,apPointer);
}

void Nd4jBlas::dspr2(long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long X, int incX,
                     long Y, int incY,
                     long Ap){
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDspr2_v2(*handle,convertUplo(Uplo),n,&alpha,xPointer,incX,yPointer,incY,apPointer);

}

/*
 * ======================================================
 * Level 3 BLAS procedures
 * ======================================================
 */


/*
 * ------------------------------------------------------
 * GEMM
 * ------------------------------------------------------
 */

void Nd4jBlas::sgemm(long *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha,
                     long A, int lda,
                     long B, int ldb,
                     float beta,
                     long C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSgemm_v2(*handle,convertTranspose(TransA),convertTranspose(TransB),m,n,k,&alpha,aPointer,lda,bPointer,ldb,&beta,cPointer,ldc);

}

void Nd4jBlas::dgemm(long *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha,
                     long A, int lda,
                     long B, int ldb,
                     double beta,
                     long C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDgemm_v2(*handle,convertTranspose(TransA),convertTranspose(TransB),m,n,k,&alpha,aPointer,lda,bPointer,ldb,&beta,cPointer,ldc);

}

/*
 * ------------------------------------------------------
 * SYMM
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymm(long *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     float alpha,
                     long A, int lda,
                     long B, int ldb,
                     float beta,
                     long C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSsymm_v2(*handle,convertSideMode(Side),convertUplo(Uplo),m,n,&alpha,aPointer,lda,bPointer,ldb,&beta,cPointer,ldc);

}

void Nd4jBlas::dsymm(long *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     double alpha,
                     long A, int lda,
                     long B, int ldb,
                     double beta,
                     long C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDsymm_v2(*handle,convertSideMode(Side),convertUplo(Uplo),m,n,&alpha,aPointer,lda,bPointer,ldb,&beta,cPointer,ldc);

}

/*
 * ------------------------------------------------------
 * SYRK
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyrk(long *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     float alpha,
                     long A, int lda,
                     float beta,
                     long C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSsyrk_v2(*handle,convertUplo(Uplo),convertTranspose(Trans),n,k,&alpha,aPointer,lda,&beta,cPointer,ldc);
}

void Nd4jBlas::dsyrk(long *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     double alpha,
                     long A, int lda,
                     double beta,
                     long C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDsyrk_v2(*handle,convertUplo(Uplo),convertTranspose(Trans),n,k,&alpha,aPointer,lda,&beta,cPointer,ldc);

}

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2k(long *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      float alpha,
                      long A, int lda,
                      long B, int ldb,
                      float beta,
                      long C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasSsyr2k_v2(*handle,convertUplo(Uplo),convertTranspose(Trans),n,k,&alpha,aPointer,lda,bPointer,ldb,&beta,cPointer,ldc);

}

void Nd4jBlas::dsyr2k(long *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      double alpha,
                      long A, int lda,
                      long B, int ldb,
                      double beta,
                      long C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDsyr2k_v2(*handle,convertUplo(Uplo),convertTranspose(Trans),n,k,&alpha,aPointer,lda,bPointer,ldb,&beta,cPointer,ldc);

}

/*
 * ------------------------------------------------------
 * TRMM
 * ------------------------------------------------------
 */

void Nd4jBlas::strmm(long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     long A, int lda,
                     long B, int ldb){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    float *cPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasStrmm_v2(*handle,convertSideMode(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),m,n,&alpha,aPointer,lda,bPointer,ldb,bPointer,ldb);

}

void Nd4jBlas::dtrmm(long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     long A, int lda,
                     long B, int ldb){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtrmm_v2(*handle,convertSideMode(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),m,n,&alpha,aPointer,lda,bPointer,ldb,bPointer,ldb);


}

/*
 * ------------------------------------------------------
 * TRSM
 * ------------------------------------------------------
 */

void Nd4jBlas::strsm(long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     long A, int lda,
                     long B, int ldb){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasStrsm_v2(*handle,convertSideMode(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),m,n,&alpha,aPointer,lda,bPointer,ldb);

}

void Nd4jBlas::dtrsm(long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     long A, int lda,
                     long B, int ldb){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t  *>(extraParams[0]);
    cublasDtrsm_v2(*handle,convertSideMode(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),m,n,&alpha,aPointer,lda,bPointer,ldb);


}

