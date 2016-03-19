#include "../NativeBlas.h"
#include <cublas_v2.h>
#include <pointercast.h>
#include <stdio.h>



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
        case 78: return CUBLAS_OP_N;
        case 84: return CUBLAS_OP_T;
        case 67: return CUBLAS_OP_C;
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

float Nd4jBlas::sdsdot(Nd4jPointer *extraParams, int N, float alpha,
                       Nd4jPointer X, int incX,
                       Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    //nothing found?
    return 0.0f;
}

double Nd4jBlas::dsdot(Nd4jPointer *extraParams, int N,
                       Nd4jPointer X, int incX,
                       Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    //nothing found?
    return 0.0f;
}

double Nd4jBlas::ddot(Nd4jPointer *extraParams, int N,
                      Nd4jPointer X, int incX,
                      Nd4jPointer Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    double resultPointer = 0.0f;
    cublasDdot_v2(*handle, N, xPointer, incX, yPointer, incY, &resultPointer);
    return resultPointer;
}

float Nd4jBlas::sdot(Nd4jPointer *extraParams, int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    float resultPointer = 0.0f;
    cublasSdot_v2(*handle, N, xPointer, incX, yPointer, incY, &resultPointer);
    return resultPointer;
}

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

float Nd4jBlas::snrm2(Nd4jPointer *extraParams, int N, Nd4jPointer X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    float resultPointer = 0.0f;
    cublasSnrm2_v2(*handle, N, xPointer, incX, &resultPointer);
    return resultPointer;


}

double Nd4jBlas::dnrm2(Nd4jPointer *extraParams, int N, Nd4jPointer X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    double resultPointer = 0.0;
    cublasDnrm2_v2(*handle, N, xPointer, incX, &resultPointer);
    return resultPointer;
}

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

float Nd4jBlas::sasum(Nd4jPointer *extraParams, int N, Nd4jPointer X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    float resultPointer = 0.0f;
    cublasSasum_v2(*handle, N, xPointer, incX, &resultPointer);
    return resultPointer;


}

double Nd4jBlas::dasum(Nd4jPointer *extraParams, int N, Nd4jPointer X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    double resultPointer = 0.0f;
    cublasDasum_v2(*handle, N, xPointer, incX, &resultPointer);
    return resultPointer;

}

/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

int Nd4jBlas::isamax(Nd4jPointer *extraParams, int N, Nd4jPointer X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    int resultPointer = 0;
    cublasIsamax_v2(*handle,N,xPointer,incX,&resultPointer);
    return resultPointer;

}

int Nd4jBlas::idamax(Nd4jPointer *extraParams, int N, Nd4jPointer X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    int resultPointer = 0;
    cublasIdamax_v2(*handle, N, xPointer, incX, &resultPointer);
    return resultPointer;


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

void Nd4jBlas::srot(Nd4jPointer *extraParams, int N,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
                    float c, float s) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSrot_v2(*handle, N, xPointer, incX, yPointer, incY, &c, &s);
}

void Nd4jBlas::drot(Nd4jPointer *extraParams, int N,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
                    double c, double s) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDrot_v2(*handle, N, xPointer, incX, yPointer, incY, &c, &s);
}

/*
 * ------------------------------------------------------
 * ROTG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotg(Nd4jPointer *extraParams, Nd4jPointer args) {
    float *argsPointers = reinterpret_cast<float *>(args);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSrotg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3]);
}

void Nd4jBlas::drotg(Nd4jPointer *extraParams, Nd4jPointer args) {
    double *argsPointers = reinterpret_cast<double *>(args);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDrotg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3]);

}

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotmg(Nd4jPointer *extraParams, Nd4jPointer args,
                      Nd4jPointer P) {
    float *argsPointers = reinterpret_cast<float *>(args);
    float *pPointers = reinterpret_cast<float *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSrotmg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3], pPointers);

}

void Nd4jBlas::drotmg(Nd4jPointer *extraParams, Nd4jPointer args,
                      Nd4jPointer P) {
    double *argsPointers = reinterpret_cast<double *>(args);
    double *pPointers = reinterpret_cast<double *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDrotmg_v2(*handle, &argsPointers[0], &argsPointers[1], &argsPointers[2], &argsPointers[3], pPointers);

}

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

void Nd4jBlas::srotm(Nd4jPointer *extraParams, int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer P) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *pPointer = reinterpret_cast<float *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSrotm_v2(*handle, N, xPointer, incX, yPointer, incY, pPointer);

}

void Nd4jBlas::drotm(Nd4jPointer *extraParams, int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer P) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *pPointer = reinterpret_cast<double *>(P);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDrotm_v2(*handle, N, xPointer, incX, yPointer, incY, pPointer);

}

/*
 * ------------------------------------------------------
 * SWAP
 * ------------------------------------------------------
 */

void Nd4jBlas::sswap(Nd4jPointer *extraParams, int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSswap_v2(*handle, N, xPointer, incX, yPointer, incY);
}

void Nd4jBlas::dswap(Nd4jPointer *extraParams, int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDswap_v2(*handle, N, xPointer, incX, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

void Nd4jBlas::sscal(Nd4jPointer *extraParams, int N, float alpha,
                     Nd4jPointer X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSscal_v2(*handle, N, &alpha, xPointer, incX);

}

void Nd4jBlas::dscal(Nd4jPointer *extraParams, int N, double alpha,
                     Nd4jPointer X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDscal_v2(*handle, N, &alpha, xPointer, incX);
}

/*
 * ------------------------------------------------------
 * SCOPY
 * ------------------------------------------------------
 */

void Nd4jBlas::scopy(Nd4jPointer *extraParams, int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasScopy_v2(*handle, N, xPointer, incX, yPointer, incY);
}

void Nd4jBlas::dcopy(Nd4jPointer *extraParams, int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDcopy_v2(*handle, N, xPointer, incX, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

void Nd4jBlas::saxpy(Nd4jPointer *extraParams, int N, float alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSaxpy_v2(*handle, N, &alpha, xPointer, incX, yPointer, incY);
}

void Nd4jBlas::daxpy(Nd4jPointer *extraParams, int N, double alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
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

void Nd4jBlas::sgemv(Nd4jPointer *extraParams, int Order, int TransA,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *aPointer = reinterpret_cast<float *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSgemv_v2(*handle, convertTranspose(TransA), M, N, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer,
                   incY);
}

void Nd4jBlas::dgemv(Nd4jPointer *extraParams, int Order, int TransA,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *aPointer = reinterpret_cast<double *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDgemv_v2(*handle,convertTranspose(TransA),M,N,&alpha,aPointer,lda,xPointer,incX,&beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * GBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sgbmv(Nd4jPointer *extraParams, int Order, int TransA,
                     int M, int N,
                     int KL, int KU,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSgbmv_v2(*handle, convertTranspose(TransA), M, N, KL, KU, &alpha, aPointer, lda, xPointer, incX, &beta,
                   yPointer, incY);
}

void Nd4jBlas::dgbmv(Nd4jPointer *extraParams, int Order, int TransA,
                     int M, int N,
                     int KL, int KU,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY) {

    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDgbmv_v2(*handle, convertTranspose(TransA), M, N, KL, KU, &alpha, aPointer, lda, xPointer, incX, &beta,
                   yPointer, incY);
}

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSsymv_v2(*handle, convertUplo(Uplo), N, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);
}

void Nd4jBlas::dsymv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDsymv_v2(*handle, convertUplo(Uplo), N, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * SBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssbmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N, int K,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSsbmv_v2(*handle, convertUplo(Uplo), N, K, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);
}

void Nd4jBlas::dsbmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N, int K,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDsbmv_v2(*handle, convertUplo(Uplo), N, K, &alpha, aPointer, lda, xPointer, incX, &beta, yPointer, incY);

}

/*
 * ------------------------------------------------------
 * SPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sspmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSspmv_v2(*handle, convertUplo(Uplo), N, &alpha, apPointer, xPointer, incX, &beta, yPointer, incY);

}

void Nd4jBlas::dspmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDspmv_v2(*handle, convertUplo(Uplo), N, &alpha, apPointer, xPointer, incX, &beta, yPointer, incY);


}

/*
 * ------------------------------------------------------
 * TRMV
 * ------------------------------------------------------
 */

void Nd4jBlas::strmv(Nd4jPointer *extraParams, int Order, int Uplo, int TransA,
                     int Diag,
                     int N, float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasStrmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda,
                   xPointer, incX);
}

void Nd4jBlas::dtrmv(Nd4jPointer *extraParams, int Order, int Uplo, int TransA,
                     int Diag,
                     int N, double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtrmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda,
                   xPointer, incX);

}

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasStbmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);
}

void Nd4jBlas::dtbmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtbmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);

}

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasStpmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);

}

void Nd4jBlas::dtpmv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtpmv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);

}

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

void Nd4jBlas::strsv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasStrsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda, xPointer,
                   incX);
}

void Nd4jBlas::dtrsv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtrsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, aPointer, lda, xPointer,
                   incX);

}

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbsv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasStbsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);

}

void Nd4jBlas::dtbsv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtbsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, K, aPointer, lda,
                   xPointer, incX);

}

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpsv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasStpsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);
}

void Nd4jBlas::dtpsv(Nd4jPointer *extraParams, int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtpsv_v2(*handle, convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), N, apPointer, xPointer,
                   incX);

}

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

void Nd4jBlas::sger(Nd4jPointer *extraParams, int Order,
                    int M, int N,
                    float alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
                    Nd4jPointer A, int lda) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSger_v2(*handle, M, N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

void Nd4jBlas::dger(Nd4jPointer *extraParams, int Order,
                    int M, int N,
                    double alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
                    Nd4jPointer A, int lda) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDger_v2(*handle, M, N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

/*
 * ------------------------------------------------------
 * SYR
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr(Nd4jPointer *extraParams, int Order, int Uplo,
                    int N,
                    float alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer A, int lda) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *aPointer = reinterpret_cast<float *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSsyr_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, aPointer, lda);
}

void Nd4jBlas::dsyr(Nd4jPointer *extraParams, int Order, int Uplo,
                    int N,
                    double alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer A, int lda) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *aPointer = reinterpret_cast<double *>(A);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDsyr_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, aPointer, lda);

}

/*
 * ------------------------------------------------------
 * SPR
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr(Nd4jPointer *extraParams, int Order, int Uplo,
                    int N,
                    float alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Ap) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *apPointer = reinterpret_cast<float *>(Ap);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSspr(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, apPointer);
}

void Nd4jBlas::dspr(Nd4jPointer *extraParams, int Order, int Uplo,
                    int N,
                    double alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Ap) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *apPointer = reinterpret_cast<double *>(Ap);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDspr(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, apPointer);

}

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer A, int lda) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSsyr2_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

void Nd4jBlas::dsyr2(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer A, int lda) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDsyr2_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, yPointer, incY, aPointer, lda);

}

/*
 * ------------------------------------------------------
 * SPR2
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr2(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer Ap) {
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSspr2_v2(*handle, convertUplo(Uplo), N, &alpha, xPointer, incX, yPointer, incY, apPointer);
}

void Nd4jBlas::dspr2(Nd4jPointer *extraParams, int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer Ap) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
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

void Nd4jBlas::sgemm(Nd4jPointer *extraParams, int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     float beta,
                     Nd4jPointer C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSgemm_v2(*handle, convertTranspose(TransA), convertTranspose(TransB), M, N, K, &alpha, aPointer, lda,
                   bPointer, ldb, &beta, cPointer, ldc);

}

void Nd4jBlas::dgemm(Nd4jPointer *extraParams, int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     double beta,
                     Nd4jPointer C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDgemm_v2(*handle, convertTranspose(TransA), convertTranspose(TransB), M, N, K, &alpha, aPointer, lda,
                   bPointer, ldb, &beta, cPointer, ldc);

}

/*
 * ------------------------------------------------------
 * SYMM
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymm(Nd4jPointer *extraParams, int Order, int Side, int Uplo,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     float beta,
                     Nd4jPointer C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSsymm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), M, N, &alpha, aPointer, lda, bPointer, ldb, &beta,
                   cPointer, ldc);

}

void Nd4jBlas::dsymm(Nd4jPointer *extraParams, int Order, int Side, int Uplo,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     double beta,
                     Nd4jPointer C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDsymm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), M, N, &alpha, aPointer, lda, bPointer, ldb, &beta,
                   cPointer, ldc);

}

/*
 * ------------------------------------------------------
 * SYRK
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyrk(Nd4jPointer *extraParams, int Order, int Uplo, int Trans,
                     int N, int K,
                     float alpha,
                     Nd4jPointer A, int lda,
                     float beta,
                     Nd4jPointer C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSsyrk_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, &beta, cPointer,
                   ldc);
}

void Nd4jBlas::dsyrk(Nd4jPointer *extraParams, int Order, int Uplo, int Trans,
                     int N, int K,
                     double alpha,
                     Nd4jPointer A, int lda,
                     double beta,
                     Nd4jPointer C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDsyrk_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, &beta, cPointer,
                   ldc);

}

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2k(Nd4jPointer *extraParams, int Order, int Uplo, int Trans,
                      int N, int K,
                      float alpha,
                      Nd4jPointer A, int lda,
                      Nd4jPointer B, int ldb,
                      float beta,
                      Nd4jPointer C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasSsyr2k_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, bPointer, ldb,
                    &beta, cPointer, ldc);

}

void Nd4jBlas::dsyr2k(Nd4jPointer *extraParams, int Order, int Uplo, int Trans,
                      int N, int K,
                      double alpha,
                      Nd4jPointer A, int lda,
                      Nd4jPointer B, int ldb,
                      double beta,
                      Nd4jPointer C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDsyr2k_v2(*handle, convertUplo(Uplo), convertTranspose(Trans), N, K, &alpha, aPointer, lda, bPointer, ldb,
                    &beta, cPointer, ldc);

}

/*
 * ------------------------------------------------------
 * TRMM
 * ------------------------------------------------------
 */

void Nd4jBlas::strmm(Nd4jPointer *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    float *cPointer = reinterpret_cast<float *>(&extraParams[1]);
    cublasStrmm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb, bPointer, ldb);

}

void Nd4jBlas::dtrmm(Nd4jPointer *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtrmm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb, bPointer, ldb);


}

/*
 * ------------------------------------------------------
 * TRSM
 * ------------------------------------------------------
 */

void Nd4jBlas::strsm(Nd4jPointer *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasStrsm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb);

}

void Nd4jBlas::dtrsm(Nd4jPointer *extraParams, int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cublasHandle_t *handle = reinterpret_cast<cublasHandle_t *>(&extraParams[0]);
    cublasDtrsm_v2(*handle, convertSideMode(Side), convertUplo(Uplo), convertTranspose(TransA), convertDiag(Diag), M, N,
                   &alpha, aPointer, lda, bPointer, ldb);


}

