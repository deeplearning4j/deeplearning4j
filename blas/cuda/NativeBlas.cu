#include "../NativeBlas.h"
#include <cublas_v2.h>



cublasStatus_t convertStatus(int status) {
    switch (status) {
        case 0:
            return CUBLAS_STATUS_SUCCESS;
        case 1:
            return CUBLAS_STATUS_NOT_INITIALIZED;
        case 3:
            return CUBLAS_STATUS_ALLOC_FAILED;
        case 7:
            return CUBLAS_STATUS_INVALID_VALUE;
        case 8:
            return CUBLAS_STATUS_ARCH_MISMATCH;
        case 11:
            return CUBLAS_STATUS_MAPPING_ERROR;
        case 13:
            return CUBLAS_STATUS_EXECUTION_FAILED;
        case 14:
            return CUBLAS_STATUS_INTERNAL_ERROR;
        case 15:
            return CUBLAS_STATUS_NOT_SUPPORTED;
        case 16:
            return CUBLAS_STATUS_LICENSE_ERROR;
        default:
            return CUBLAS_STATUS_SUCCESS;
    }
}

cublasFillMode_t convertUplo(int fillMode) {
    switch (fillMode) {
        case 0:
            return CUBLAS_FILL_MODE_LOWER;
        case 1:
            return CUBLAS_FILL_MODE_UPPER;
        default:
            return CUBLAS_FILL_MODE_LOWER;
    }
}

cublasDiagType_t convertDiag(int diag) {
    switch (diag) {
        case 0:
            return CUBLAS_DIAG_NON_UNIT;
        case 1:
            return CUBLAS_DIAG_UNIT;
        default:
            return CUBLAS_DIAG_NON_UNIT;
    }
}

cublasOperation_t convertTranspose(int op) {
    switch(op) {
        case 0: return CUBLAS_OP_N;
        case 1: return CUBLAS_OP_T;
        case 2: return CUBLAS_OP_C;
        default: return CUBLAS_OP_N;
    }
}

cublasPointerMode_t convertPointerMode(int pointerMode) {
    switch(pointerMode) {
        case 0: return CUBLAS_POINTER_MODE_HOST;
        case 1: return CUBLAS_POINTER_MODE_DEVICE;
        default: return CUBLAS_POINTER_MODE_HOST;
    }}

cublasSideMode_t convertSideMode(int sideMode) {
    switch(sideMode) {
        case 0: return CUBLAS_SIDE_LEFT;
        case 1: return CUBLAS_SIDE_RIGHT;
        default: return CUBLAS_SIDE_LEFT;
    }
}


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

float Nd4jBlas::sdsdot(long long *extraParams, int N, float alpha,
                       long long X, int incX,
                       long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    //nothing found?
    return 0.0f;
}

double Nd4jBlas::dsdot(long long *extraParams, int N,
                       long long X, int incX,
                       long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    //nothing found?
    return 0.0f;
}

double Nd4jBlas::ddot(long long *extraParams, int N,
                      long long X, int incX,
                      long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    double *resultPointer = reinterpret_cast<double *>(extraParams[1]);
    cublasDdot_v2(*handle, N, xPointer, incX, yPointer, incY, resultPointer);
    return 0.0;
}

float Nd4jBlas::sdot(long long *extraParams, int N,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    float *resultPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasSdot_v2(*handle, N, xPointer, incX, yPointer, incY, resultPointer);
    return 0.0f;
}

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

float Nd4jBlas::snrm2(long long *extraParams, int N, long long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    float *resultPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasSnrm2_v2(*handle, N, xPointer, incX, resultPointer);
    return 0.0f;


}

double Nd4jBlas::dnrm2(long long *extraParams, int N, long long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    double *resultPointer = reinterpret_cast<double *>(extraParams[1]);
    cublasDnrm2_v2(*handle, N, xPointer, incX, resultPointer);
    return 0.0;
}

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

float Nd4jBlas::sasum(long long *extraParams, int N, long long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    float *resultPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasSasum_v2(*handle, N, xPointer, incX, resultPointer);
    return 0.0f;


}

double Nd4jBlas::dasum(long long *extraParams, int N, long long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    double *resultPointer = reinterpret_cast<double *>(extraParams[1]);
    cublasDasum_v2(*handle, N, xPointer, incX, resultPointer);
    return 0.0;

}

/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

int Nd4jBlas::isamax(long long *extraParams, int N, long long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    int *resultPointer = reinterpret_cast<int *>(extraParams[1]);
    cublasIsamax_v2(*handle,N,xPointer,incX,resultPointer);
    return 0;

}

int Nd4jBlas::idamax(long long *extraParams, int N, long long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    int *resultPointer = reinterpret_cast<int *>(extraParams[1]);
    cublasIdamax_v2(*handle, N, xPointer, incX, resultPointer);
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

void Nd4jBlas::srot(long long *extraParams, int N,
                    long long X, int incX,
                    long long Y, int incY,
                    float c, float s) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSrot_v2(*handle, N, xPointer, incX, yPointer, incY, &c, &s);
}

void Nd4jBlas::drot(long long *extraParams, int N,
                    long long X, int incX,
                    long long Y, int incY,
                    double c, double s) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDrot_v2(*handle, N, xPointer, incX, yPointer, incY, &c, &s);
}

/*
 * ------------------------------------------------------
 * ROTG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotg(long long *extraParams, long long args) {
    float *argsPointers = reinterpret_cast<float *>(args);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSrotg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3]);
}

void Nd4jBlas::drotg(long long *extraParams, long long args) {
    double *argsPointers = reinterpret_cast<double *>(args);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDrotg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3]);

}

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotmg(long long *extraParams, long long args,
                      long long P) {
    float *argsPointers = reinterpret_cast<float *>(args);
    float *pPointers = reinterpret_cast<float *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSrotmg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3], pPointers);

}

void Nd4jBlas::drotmg(long long *extraParams, long long args,
                      long long P) {
    double *argsPointers = reinterpret_cast<double *>(args);
    double *pPointers = reinterpret_cast<double *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDrotmg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3], pPointers);

}

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

void Nd4jBlas::srotm(long long *extraParams, int N,
                     long long X, int incX,
                     long long Y, int incY,
                     long long P) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *pPointer = reinterpret_cast<float *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSrotm_v2(*handle, N, xPointer, incX, yPointer, incY, pPointer);

}

void Nd4jBlas::drotm(long long *extraParams, int N,
                     long long X, int incX,
                     long long Y, int incY,
                     long long P) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *pPointer = reinterpret_cast<double *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDrotm_v2(*handle, N, xPointer, incX, yPointer, incY, pPointer);

}

/*
 * ------------------------------------------------------
 * SWAP
 * ------------------------------------------------------
 */

void Nd4jBlas::sswap(long long *extraParams, int N,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSswap_v2(*handle, N, xPointer, incX, yPointer, incY);
}

void Nd4jBlas::dswap(long long *extraParams, int N,
                     long long X, int incX,
                     long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDswap_v2(*handle, N, xPointer, incX, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

void Nd4jBlas::sscal(long long *extraParams, int N, float alpha,
                     long long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSscal_v2(*handle, N, &alpha, xPointer, incX);

}

void Nd4jBlas::dscal(long long *extraParams, int N, double alpha,
                     long long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDscal_v2(*handle, N, &alpha, xPointer, incX);
}

/*
 * ------------------------------------------------------
 * SCOPY
 * ------------------------------------------------------
 */

void Nd4jBlas::scopy(long long *extraParams, int N,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasScopy_v2(*handle, N, xPointer, incX, yPointer, incY);
}

void Nd4jBlas::dcopy(long long *extraParams, int N,
                     long long X, int incX,
                     long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDcopy_v2(*handle, N, xPointer, incX, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

void Nd4jBlas::saxpy(long long *extraParams, int N, float alpha,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSaxpy_v2(*handle, N, &alpha, xPointer, incX, yPointer, incY);
}

void Nd4jBlas::daxpy(long long *extraParams, int N, double alpha,
                     long long X, int incX,
                     long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDaxpy_v2(*handle, N, &alpha, xPointer, incX, yPointer, incY);

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

void Nd4jBlas::sgemv(long long *extraParams, int Order, int TransA,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *aPointer = reinterpret_cast<float *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSgemv_v2(*handle, convertTranspose(TransA), M, N, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer,
                   incY);
}

void Nd4jBlas::dgemv(long long *extraParams, int Order, int TransA,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *aPointer = reinterpret_cast<double *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDgemv_v2(*handle,convertTranspose(TransA),M,N,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * GBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sgbmv(long long *extraParams, int Order, int TransA,
                     int M, int N,
                     int KL, int KU,
                     float alpha,
                     long long A, int lda,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSgbmv_v2(*handle, convertTranspose(TransA), M, N, KL, KU, &alpha, aPointer, lda, xPointer, incX, &beta,
                   yPointer, incY);
}

void Nd4jBlas::dgbmv(long long *extraParams, int Order, int TransA,
                     int M, int N,
                     int KL, int KU,
                     double alpha,
                     long long A, int lda,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY) {

    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDgbmv_v2(*handle, convertTranspose(TransA), M, N, KL, KU, &alpha, aPointer, lda, xPointer, incX, &beta,
                   yPointer, incY);
}

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymv(long long *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     long long A, int lda,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSsymv_v2(*handle, convertUplo(Uplo), N, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);
}

void Nd4jBlas::dsymv(long long *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     long long A, int lda,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDsymv_v2(*handle, convertUplo(Uplo), N, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * SBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssbmv(long long *extraParams, int Order, int Uplo,
                     int N, int K,
                     float alpha,
                     long long A, int lda,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSsbmv_v2(*handle, convertUplo(Uplo), N, K, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);
}

void Nd4jBlas::dsbmv(long long *extraParams, int Order, int Uplo,
                     int N, int K,
                     double alpha,
                     long long A, int lda,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDsbmv_v2(*handle, convertUplo(Uplo), N, K, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * SPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sspmv(long long *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     long long Ap,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSspmv_v2(*handle, convertUplo(Uplo), N, &alpha, apPointer, xPointer, incX, &beta, yPointer, incY);

}

void Nd4jBlas::dspmv(long long *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     long long Ap,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDspmv_v2(*handle, convertUplo(Uplo), N, &alpha, apPointer, xPointer, incX, &beta, yPointer, incY);


}

/*
 * ------------------------------------------------------
 * TRMV
 * ------------------------------------------------------
 */

void Nd4jBlas::strmv(long long *extraParams, int Order, int Uplo, int TransA,
                     int Diag,
                     int N, float alpha,
                     long long A, int lda,
                     long long X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasStrmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda,
                   xPointer, incX);
}

void Nd4jBlas::dtrmv(long long *extraParams, int Order, int Uplo, int TransA,
                     int Diag,
                     int N, double alpha,
                     long long A, int lda,
                     long long X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtrmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda,
                   xPointer, incX);

}

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbmv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasStbmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);
}

void Nd4jBlas::dtbmv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtbmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);

}

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpmv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasStpmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);

}

void Nd4jBlas::dtpmv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtpmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);

}

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

void Nd4jBlas::strsv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long A, int lda,
                     long long X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasStrsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda, xPointer,
                   incX);
}

void Nd4jBlas::dtrsv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long A, int lda,
                     long long X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtrsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda, xPointer,
                   incX);

}

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbsv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasStbsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);

}

void Nd4jBlas::dtbsv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtbsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);

}

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpsv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasStpsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);
}

void Nd4jBlas::dtpsv(long long *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtpsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);

}

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

void Nd4jBlas::sger(long long *extraParams, int Order,
                    int M, int N,
                    float alpha,
                    long long X, int incX,
                    long long Y, int incY,
                    long long A, int lda) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSger_v2(*handle, M, N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

void Nd4jBlas::dger(long long *extraParams, int Order,
                    int M, int N,
                    double alpha,
                    long long X, int incX,
                    long long Y, int incY,
                    long long A, int lda) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDger_v2(*handle, M, N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

/*
 * ------------------------------------------------------
 * SYR
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr(long long *extraParams, int Order, int Uplo,
                    int N,
                    float alpha,
                    long long X, int incX,
                    long long A, int lda) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *aPointer = reinterpret_cast<float *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSsyr_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, aPointer, lda);
}

void Nd4jBlas::dsyr(long long *extraParams, int Order, int Uplo,
                    int N,
                    double alpha,
                    long long X, int incX,
                    long long A, int lda) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *aPointer = reinterpret_cast<double *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDsyr_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, aPointer, lda);

}

/*
 * ------------------------------------------------------
 * SPR
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr(long long *extraParams, int Order, int Uplo,
                    int N,
                    float alpha,
                    long long X, int incX,
                    long long Ap) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *apPointer = reinterpret_cast<float *>(Ap);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSspr(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, apPointer);
}

void Nd4jBlas::dspr(long long *extraParams, int Order, int Uplo,
                    int N,
                    double alpha,
                    long long X, int incX,
                    long long Ap) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *apPointer = reinterpret_cast<double *>(Ap);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDspr(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, apPointer);

}

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2(long long *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long A, int lda) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSsyr2_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

void Nd4jBlas::dsyr2(long long *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long A, int lda) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDsyr2_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

/*
 * ------------------------------------------------------
 * SPR2
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr2(long long *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long Ap) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSspr2_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, yPointer, incY, apPointer);
}

void Nd4jBlas::dspr2(long long *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long Ap) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDspr2_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, yPointer, incY, apPointer);

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

void Nd4jBlas::sgemm(long long *extraParams, int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     float beta,
                     long long C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSgemm_v2(*handle, convertTranspose(TransA), convertTranspose(TransB), M, N, K, &alpha, aPointer, lda,
                   bPointer, ldb, &beta, cPointer, ldc);

}

void Nd4jBlas::dgemm(long long *extraParams, int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     double beta,
                     long long C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDgemm_v2(*handle, convertTranspose(TransA), convertTranspose(TransB), M, N, K, &alpha, aPointer, lda,
                   bPointer, ldb, &beta, cPointer, ldc);

}

/*
 * ------------------------------------------------------
 * SYMM
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymm(long long *extraParams, int Order, int Side, int Uplo,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     float beta,
                     long long C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSsymm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), M, N, &alpha, aPointer, lda, bPointer, ldb, &beta,
                   cPointer, ldc);

}

void Nd4jBlas::dsymm(long long *extraParams, int Order, int Side, int Uplo,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     double beta,
                     long long C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDsymm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), M, N, &alpha, aPointer, lda, bPointer, ldb, &beta,
                   cPointer, ldc);

}

/*
 * ------------------------------------------------------
 * SYRK
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyrk(long long *extraParams, int Order, int Uplo, int Trans,
                     int N, int K,
                     float alpha,
                     long long A, int lda,
                     float beta,
                     long long C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSsyrk_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, &beta, cPointer,
                   ldc);
}

void Nd4jBlas::dsyrk(long long *extraParams, int Order, int Uplo, int Trans,
                     int N, int K,
                     double alpha,
                     long long A, int lda,
                     double beta,
                     long long C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDsyrk_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, &beta, cPointer,
                   ldc);

}

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2k(long long *extraParams, int Order, int Uplo, int Trans,
                      int N, int K,
                      float alpha,
                      long long A, int lda,
                      long long B, int ldb,
                      float beta,
                      long long C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasSsyr2k_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, bPointer, ldb,
                    &beta, cPointer, ldc);

}

void Nd4jBlas::dsyr2k(long long *extraParams, int Order, int Uplo, int Trans,
                      int N, int K,
                      double alpha,
                      long long A, int lda,
                      long long B, int ldb,
                      double beta,
                      long long C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDsyr2k_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, bPointer, ldb,
                    &beta, cPointer, ldc);

}

/*
 * ------------------------------------------------------
 * TRMM
 * ------------------------------------------------------
 */

void Nd4jBlas::strmm(long long *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    float *cPointer = reinterpret_cast<float *>(extraParams[1]);
    cublasStrmm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb, bPointer, ldb);

}

void Nd4jBlas::dtrmm(long long *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtrmm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb, bPointer, ldb);


}

/*
 * ------------------------------------------------------
 * TRSM
 * ------------------------------------------------------
 */

void Nd4jBlas::strsm(long long *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasStrsm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb);

}

void Nd4jBlas::dtrsm(long long *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(extraParams[0]);
    cublasDtrsm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb);


}

