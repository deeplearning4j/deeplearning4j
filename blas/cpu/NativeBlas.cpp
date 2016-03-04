//
// Created by agibsonccc on 2/21/16.
//

#include "../NativeBlas.h"
#include <cblas.h>


/**
 * Converts a character
 * to its proper enum
 * for row (c) or column (f) ordering
 * default is row major
 */
CBLAS_ORDER  convertOrder(int from) {
    switch(from) {
        //'c'
        case 99:
            return CblasRowMajor;
            //'C'
        case 67: return CblasRowMajor;
            //'f'
        case 102: return CblasColMajor;
            //'F'
        case 70: return CblasColMajor;
        default: return CblasColMajor;

    }
}

/**
 * Converts a character to its proper enum
 * t -> transpose
 * n -> no transpose
 * c -> conj
 */
CBLAS_TRANSPOSE  convertTranspose(int from) {
    switch(from) {
        //'t'
        case 116: return CblasTrans;
            //'T'
        case 84: return CblasTrans;
            //'n'
        case 110: return CblasNoTrans;
            //'N'
        case 78: return CblasNoTrans;
            //'c'
        case 99: return CblasConjTrans;
            //'C'
        case 67: return CblasConjTrans;
        default: return CblasNoTrans;
    }
}
/**
 * Upper or lower
 * U/u -> upper
 * L/l -> lower
 *
 * Default is upper
 */
CBLAS_UPLO  convertUplo(int from) {
    switch(from) {
        //'u'
        case 117: return CblasUpper;
            //'U'
        case 85: return CblasUpper;
            //'l'
        case 108: return CblasLower;
            //'L'
        case 76: return CblasLower;
        default: return CblasUpper;

    }
}


/**
 * For diagonals:
 * u/U -> unit
 * n/N -> non unit
 *
 * Default: unit
 */
CBLAS_DIAG convertDiag(int from) {
    switch(from) {
        //'u'
        case 117: return CblasUnit;
            //'U'
        case 85: return CblasUnit;
            //'n'
        case 110: return CblasNonUnit;
            //'N'
        case 78: return CblasNonUnit;
        default: return CblasUnit;
    }
}


/**
 * Side of a matrix, left or right
 * l /L -> left
 * r/R -> right
 * default: left
 */
CBLAS_SIDE convertSide(int from) {
    switch(from) {
//'l'
        case 108: return CblasLeft;
            //'L'
        case 76: return CblasLeft;
        case 82: return CblasRight;
        case 114: return CblasRight;
        default: return CblasLeft;
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

float Nd4jBlas::sdsdot(long long *extraParams,int N, float alpha,
                       long long X, int incX,
                       long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    return cblas_sdsdot(N,alpha,xPointer,incX,yPointer,incY);

}

double Nd4jBlas::dsdot(long long *extraParams,int N,
                       long long X, int incX,
                       long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    return cblas_dsdot(N,xPointer,incX,yPointer,incY);
}

double Nd4jBlas::ddot(long long *extraParams,int N,
                      long long X, int incX,
                      long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    return cblas_ddot(N,xPointer,incX,yPointer,incY);
}

float Nd4jBlas::sdot(long long *extraParams,int N,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    return cblas_sdot(N,xPointer,incX,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

float Nd4jBlas::snrm2(long long *extraParams,int N, long long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    return cblas_snrm2(N,xPointer,incX);

}

double Nd4jBlas::dnrm2(long long *extraParams,int N, long long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    return cblas_dnrm2(N,xPointer,incX);
}

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

float Nd4jBlas::sasum(long long *extraParams,int N, long long X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    return cblas_sasum(N,xPointer,incX);

}

double Nd4jBlas::dasum(long long *extraParams,int N, long long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    return cblas_dasum(N,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

int Nd4jBlas::isamax(long long *extraParams,int N, long long X, int incX){
    float *xPointer = reinterpret_cast<float *>(X);
    return cblas_isamax(N,xPointer,incX);

}

int Nd4jBlas::idamax(long long *extraParams,int N, long long X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    return cblas_idamax(N,xPointer,incX);

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

void Nd4jBlas::srot(long long *extraParams,int N,
                    long long X, int incX,
                    long long Y, int incY,
                    float c, float s) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_srot(N,xPointer,incX,yPointer,incY,c,s);

}

void Nd4jBlas::drot(long long *extraParams,int N,
                    long long X, int incX,
                    long long Y, int incY,
                    double c, double s) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_drot(N,xPointer,incX,yPointer,incY,c,s);
}

/*
 * ------------------------------------------------------
 * ROTG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotg(long long *extraParams,long long args) {
    float *argsPointers = reinterpret_cast<float *>(args);
    return cblas_srotg(&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3]);
}

void Nd4jBlas::drotg(long long *extraParams,long long args) {
    double *argsPointers = reinterpret_cast<double *>(args);
    cblas_drotg(&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3]);

}

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotmg(long long *extraParams,long long args,
                      long long P) {
    float *argsPointers = reinterpret_cast<float *>(args);
    float *pPointers = reinterpret_cast<float *>(P);
    return cblas_srotmg(&argsPointers[0],&argsPointers[1],&argsPointers[2],argsPointers[3],pPointers);

}

void Nd4jBlas::drotmg(long long *extraParams,long long args,
                      long long P) {
    double *argsPointers = reinterpret_cast<double *>(args);
    double *pPointers = reinterpret_cast<double *>(P);
    cblas_drotmg(&argsPointers[0],&argsPointers[1],&argsPointers[2],argsPointers[3],pPointers);

}

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

void Nd4jBlas::srotm(long long *extraParams,int N,
                     long long X, int incX,
                     long long Y, int incY,
                     long long P) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *pPointer = reinterpret_cast<float *>(P);
    cblas_srotm(N,xPointer,incX,yPointer,incY,pPointer);

}

void Nd4jBlas::drotm(long long *extraParams,int N,
                     long long X, int incX,
                     long long Y, int incY,
                     long long P) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *pPointer = reinterpret_cast<double *>(P);
    cblas_drotm(N,xPointer,incX,yPointer,incY,pPointer);

}

/*
 * ------------------------------------------------------
 * SWAP
 * ------------------------------------------------------
 */

void Nd4jBlas::sswap(long long *extraParams,int N,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sswap(N,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::dswap(long long *extraParams,int N,
                     long long X, int incX,
                     long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dswap(N,xPointer,incX,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

void Nd4jBlas::sscal(long long *extraParams,int N, float alpha,
                     long long X, int incX){
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_sscal(N,alpha,xPointer,incX);

}

void Nd4jBlas::dscal(long long *extraParams,int N, double alpha,
                     long long X, int incX){
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dscal(N,alpha,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * SCOPY
 * ------------------------------------------------------
 */

void Nd4jBlas::scopy(long long *extraParams,int N,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_scopy(N,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::dcopy(long long *extraParams,int N,
                     long long X, int incX,
                     long long Y, int incY){
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dcopy(N,xPointer,incX,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

void Nd4jBlas::saxpy(long long *extraParams,int N, float alpha,
                     long long X, int incX,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_saxpy(N,alpha,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::daxpy(long long *extraParams,int N, double alpha,
                     long long X, int incX,
                     long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_daxpy(N,alpha,xPointer,incX,yPointer,incY);
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

void Nd4jBlas::sgemv(long long *extraParams,int Order, int TransA,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *aPointer = reinterpret_cast<float *>(A);
    cblas_sgemv(convertOrder(Order),convertTranspose(TransA),M,N,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);

}

void Nd4jBlas::dgemv(long long *extraParams,int Order, int TransA,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    double *aPointer = reinterpret_cast<double *>(A);
    cblas_dgemv(convertOrder(Order),convertTranspose(TransA),M,N,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * GBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sgbmv(long long *extraParams,int Order, int TransA,
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
    cblas_sgbmv(convertOrder(Order),convertTranspose(TransA),N,N,KL,KU,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dgbmv(long long *extraParams,int Order, int TransA,
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
    cblas_dgbmv(convertOrder(Order),convertTranspose(TransA),M,N,KL,KU,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymv(long long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long long A, int lda,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_ssymv(convertOrder(Order),convertUplo(Uplo),N,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dsymv(long long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long long A, int lda,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dsymv(convertOrder(Order),convertUplo(Uplo),N,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * SBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssbmv(long long *extraParams,int Order, int Uplo,
                     int N, int K,
                     float alpha,
                     long long A, int lda,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_ssbmv(convertOrder(Order),convertUplo(Uplo),N,K,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dsbmv(long long *extraParams,int Order, int Uplo,
                     int N, int K,
                     double alpha,
                     long long A, int lda,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dsbmv(convertOrder(Order),convertUplo(Uplo),N,K,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * SPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sspmv(long long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long long Ap,
                     long long X, int incX,
                     float beta,
                     long long Y, int incY){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sspmv(convertOrder(Order),convertUplo(Uplo),N,alpha,apPointer,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dspmv(long long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long long Ap,
                     long long X, int incX,
                     double beta,
                     long long Y, int incY){
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dspmv(convertOrder(Order),convertUplo(Uplo),N,alpha,apPointer,xPointer,incX,beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * TRMV
 * ------------------------------------------------------
 */

void Nd4jBlas::strmv(long long *extraParams,int Order, int Uplo, int TransA,
                     int Diag,
                     int N, float alpha,
                     long long A, int lda,
                     long long X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_strmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtrmv(long long *extraParams,int Order, int Uplo, int TransA,int Diag,
                     int N, double alpha,
                     long long A, int lda,
                     long long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtrmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);
}

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbmv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stbmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtbmv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtbmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);
}

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpmv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stpmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);
}

void Nd4jBlas::dtpmv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtpmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

void Nd4jBlas::strsv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long A, int lda,
                     long long X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_strsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtrsv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long A, int lda,
                     long long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtrsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbsv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stbsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);

}

void Nd4jBlas::dtbsv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     long long A, int lda,
                     long long X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtbsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpsv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stpsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);
}

void Nd4jBlas::dtpsv(long long *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     long long Ap,
                     long long X, int incX){
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtpsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

void Nd4jBlas::sger(long long *extraParams,int Order,
                    int M, int N,
                    float alpha,
                    long long X, int incX,
                    long long Y, int incY,
                    long long A, int lda){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sger(convertOrder(Order),M,N,alpha,xPointer,incX,yPointer,incY,aPointer,lda);
}

void Nd4jBlas::dger(long long *extraParams,int Order,
                    int M, int N,
                    double alpha,
                    long long X, int incX,
                    long long Y, int incY,
                    long long A, int lda){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dger(convertOrder(Order),M,N,alpha,xPointer,incX,yPointer,incY,aPointer,lda);
}

/*
 * ------------------------------------------------------
 * SYR
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr(long long *extraParams,int Order, int Uplo,
                    int N,
                    float alpha,
                    long long X, int incX,
                    long long A, int lda){
    float *xPointer = reinterpret_cast<float *>(X);
    float *aPointer = reinterpret_cast<float *>(A);
    cblas_ssyr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,aPointer,lda);
}

void Nd4jBlas::dsyr(long long *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    long long X, int incX,
                    long long A, int lda){
    double *xPointer = reinterpret_cast<double *>(X);
    double *aPointer = reinterpret_cast<double *>(A);
    cblas_dsyr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,aPointer,lda);

}

/*
 * ------------------------------------------------------
 * SPR
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr(long long *extraParams,
                    int Order,
                    int Uplo,
                    int N,
                    float alpha,
                    long long X,
                    int incX,
                    long long Ap){
    float *xPointer = reinterpret_cast<float *>(X);
    float *apPointer = reinterpret_cast<float *>(Ap);
    cblas_sspr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,apPointer);
}

void Nd4jBlas::dspr(long long *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    long long X, int incX,
                    long long Ap){
    double *xPointer = reinterpret_cast<double *>(X);
    double *apPointer = reinterpret_cast<double *>(Ap);
    cblas_dspr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,apPointer);

}

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2(long long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long A, int lda) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_ssyr2(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,yPointer,incY,aPointer,lda);
}

void Nd4jBlas::dsyr2(long long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long A, int lda){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dsyr2(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,yPointer,incY,aPointer,lda);
}

/*
 * ------------------------------------------------------
 * SPR2
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr2(long long *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long Ap){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sspr2(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,yPointer,incY,apPointer);
}

void Nd4jBlas::dspr2(long long *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     long long X, int incX,
                     long long Y, int incY,
                     long long Ap){
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dspr2(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,yPointer,incY,apPointer);
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

void Nd4jBlas::sgemm(long long *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     float beta,
                     long long C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_sgemm(convertOrder(Order),convertTranspose(TransA),convertTranspose(TransB),M,N,K,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

void Nd4jBlas::dgemm(long long *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     double beta,
                     long long C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cblas_dgemm(convertOrder(Order),convertTranspose(TransA),convertTranspose(TransB),M,N,K,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

/*
 * ------------------------------------------------------
 * SYMM
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymm(long long *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     float beta,
                     long long C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_ssymm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),M,N,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

void Nd4jBlas::dsymm(long long *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb,
                     double beta,
                     long long C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cblas_dsymm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),M,N,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

/*
 * ------------------------------------------------------
 * SYRK
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyrk(long long *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     float alpha,
                     long long A, int lda,
                     float beta,
                     long long C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_ssyrk(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,aPointer,lda,beta,cPointer,ldc);
}

void Nd4jBlas::dsyrk(long long *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     double alpha,
                     long long A, int lda,
                     double beta,
                     long long C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *cPointer = reinterpret_cast<double *>(C);
    cblas_dsyrk(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,aPointer,lda,beta,cPointer,ldc);
}

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2k(long long *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      float alpha,
                      long long A, int lda,
                      long long B, int ldb,
                      float beta,
                      long long C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_ssyr2k(convertOrder(Trans),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

void Nd4jBlas::dsyr2k(long long *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      double alpha,
                      long long A, int lda,
                      long long B, int ldb,
                      double beta,
                      long long C, int ldc) {
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    double *cPointer = reinterpret_cast<double *>(C);
    cblas_dsyr2k(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

/*
 * ------------------------------------------------------
 * TRMM
 * ------------------------------------------------------
 */

void Nd4jBlas::strmm(long long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cblas_strmm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

void Nd4jBlas::dtrmm(long long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cblas_dtrmm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

/*
 * ------------------------------------------------------
 * TRSM
 * ------------------------------------------------------
 */

void Nd4jBlas::strsm(long long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     long long A, int lda,
                     long long B, int ldb){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cblas_strsm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

void Nd4jBlas::dtrsm(long long *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     long long A, int lda,
                     long long B, int ldb){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cblas_dtrsm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

