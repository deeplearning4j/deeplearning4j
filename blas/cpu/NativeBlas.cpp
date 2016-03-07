//
// Created by agibsonccc on 2/21/16.
//

#include "../NativeBlas.h"
#include <dll.h>
#include <cblas.h>
#include <pointercast.h>



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

float Nd4jBlas::sdsdot(Nd4jPointer *extraParams,int N, float alpha,
                       Nd4jPointer X, int incX,
                       Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    return cblas_sdsdot(N,alpha,xPointer,incX,yPointer,incY);

}

double Nd4jBlas::dsdot(Nd4jPointer *extraParams,int N,
                       Nd4jPointer X, int incX,
                       Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    return cblas_dsdot(N,xPointer,incX,yPointer,incY);
}

double Nd4jBlas::ddot(Nd4jPointer *extraParams,int N,
                      Nd4jPointer X, int incX,
                      Nd4jPointer Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    return cblas_ddot(N,xPointer,incX,yPointer,incY);
}

float Nd4jBlas::sdot(Nd4jPointer *extraParams,int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    return cblas_sdot(N,xPointer,incX,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

float Nd4jBlas::snrm2(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    return cblas_snrm2(N,xPointer,incX);

}

double Nd4jBlas::dnrm2(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    return cblas_dnrm2(N,xPointer,incX);
}

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

float Nd4jBlas::sasum(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX) {
    float *xPointer = reinterpret_cast<float *>(X);
    return cblas_sasum(N,xPointer,incX);

}

double Nd4jBlas::dasum(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX) {
    double *xPointer = reinterpret_cast<double *>(X);
    return cblas_dasum(N,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

int Nd4jBlas::isamax(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX){
    float *xPointer = reinterpret_cast<float *>(X);
    return cblas_isamax(N,xPointer,incX);

}

int Nd4jBlas::idamax(Nd4jPointer *extraParams,int N, Nd4jPointer X, int incX) {
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

void Nd4jBlas::srot(Nd4jPointer *extraParams,int N,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
                    float c, float s) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_srot(N,xPointer,incX,yPointer,incY,c,s);

}

void Nd4jBlas::drot(Nd4jPointer *extraParams,int N,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
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

void Nd4jBlas::srotg(Nd4jPointer *extraParams,Nd4jPointer args) {
    float *argsPointers = reinterpret_cast<float *>(args);
    return cblas_srotg(&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3]);
}

void Nd4jBlas::drotg(Nd4jPointer *extraParams,Nd4jPointer args) {
    double *argsPointers = reinterpret_cast<double *>(args);
    cblas_drotg(&argsPointers[0],&argsPointers[1],&argsPointers[2],&argsPointers[3]);

}

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotmg(Nd4jPointer *extraParams,Nd4jPointer args,
                      Nd4jPointer P) {
    float *argsPointers = reinterpret_cast<float *>(args);
    float *pPointers = reinterpret_cast<float *>(P);
    return cblas_srotmg(&argsPointers[0],&argsPointers[1],&argsPointers[2],argsPointers[3],pPointers);

}

void Nd4jBlas::drotmg(Nd4jPointer *extraParams,Nd4jPointer args,
                      Nd4jPointer P) {
    double *argsPointers = reinterpret_cast<double *>(args);
    double *pPointers = reinterpret_cast<double *>(P);
    cblas_drotmg(&argsPointers[0],&argsPointers[1],&argsPointers[2],argsPointers[3],pPointers);

}

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

void Nd4jBlas::srotm(Nd4jPointer *extraParams,int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer P) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *pPointer = reinterpret_cast<float *>(P);
    cblas_srotm(N,xPointer,incX,yPointer,incY,pPointer);

}

void Nd4jBlas::drotm(Nd4jPointer *extraParams,int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer P) {
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

void Nd4jBlas::sswap(Nd4jPointer *extraParams,int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sswap(N,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::dswap(Nd4jPointer *extraParams,int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dswap(N,xPointer,incX,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

void Nd4jBlas::sscal(Nd4jPointer *extraParams,int N, float alpha,
                     Nd4jPointer X, int incX){
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_sscal(N,alpha,xPointer,incX);

}

void Nd4jBlas::dscal(Nd4jPointer *extraParams,int N, double alpha,
                     Nd4jPointer X, int incX){
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dscal(N,alpha,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * SCOPY
 * ------------------------------------------------------
 */

void Nd4jBlas::scopy(Nd4jPointer *extraParams,int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_scopy(N,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::dcopy(Nd4jPointer *extraParams,int N,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY){
    double *xPointer = reinterpret_cast<double *>(X);
    double *yPointer = reinterpret_cast<double *>(Y);
    cblas_dcopy(N,xPointer,incX,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

void Nd4jBlas::saxpy(Nd4jPointer *extraParams,int N, float alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_saxpy(N,alpha,xPointer,incX,yPointer,incY);
}

void Nd4jBlas::daxpy(Nd4jPointer *extraParams,int N, double alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY) {
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

void Nd4jBlas::sgemv(Nd4jPointer *extraParams,int Order, int TransA,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    float *aPointer = reinterpret_cast<float *>(A);
    cblas_sgemv(convertOrder(Order),convertTranspose(TransA),M,N,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);

}

void Nd4jBlas::dgemv(Nd4jPointer *extraParams,int Order, int TransA,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY) {
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

void Nd4jBlas::sgbmv(Nd4jPointer *extraParams,int Order, int TransA,
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
    cblas_sgbmv(convertOrder(Order),convertTranspose(TransA),N,N,KL,KU,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dgbmv(Nd4jPointer *extraParams,int Order, int TransA,
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
    cblas_dgbmv(convertOrder(Order),convertTranspose(TransA),M,N,KL,KU,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_ssymv(convertOrder(Order),convertUplo(Uplo),N,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dsymv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY) {
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

void Nd4jBlas::ssbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N, int K,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_ssbmv(convertOrder(Order),convertUplo(Uplo),N,K,alpha,aPointer,lda,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dsbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N, int K,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY){
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

void Nd4jBlas::sspmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX,
                     float beta,
                     Nd4jPointer Y, int incY){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sspmv(convertOrder(Order),convertUplo(Uplo),N,alpha,apPointer,xPointer,incX,beta,yPointer,incY);
}

void Nd4jBlas::dspmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX,
                     double beta,
                     Nd4jPointer Y, int incY){
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

void Nd4jBlas::strmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,
                     int Diag,
                     int N, float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_strmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtrmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,int Diag,
                     int N, double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtrmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);
}

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stbmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtbmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);
}

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stpmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);
}

void Nd4jBlas::dtpmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX) {
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtpmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

void Nd4jBlas::strsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_strsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);
}

void Nd4jBlas::dtrsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtrsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stbsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);

}

void Nd4jBlas::dtbsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     Nd4jPointer A, int lda,
                     Nd4jPointer X, int incX){
    double *aPointer = reinterpret_cast<double *>(A);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtbsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,aPointer,lda,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    cblas_stpsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);
}

void Nd4jBlas::dtpsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     Nd4jPointer Ap,
                     Nd4jPointer X, int incX){
    double *apPointer = reinterpret_cast<double *>(Ap);
    double *xPointer = reinterpret_cast<double *>(X);
    cblas_dtpsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,apPointer,xPointer,incX);

}

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

void Nd4jBlas::sger(Nd4jPointer *extraParams,int Order,
                    int M, int N,
                    float alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
                    Nd4jPointer A, int lda){
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sger(convertOrder(Order),M,N,alpha,xPointer,incX,yPointer,incY,aPointer,lda);
}

void Nd4jBlas::dger(Nd4jPointer *extraParams,int Order,
                    int M, int N,
                    double alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Y, int incY,
                    Nd4jPointer A, int lda){
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

void Nd4jBlas::ssyr(Nd4jPointer *extraParams,int Order, int Uplo,
                    int N,
                    float alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer A, int lda){
    float *xPointer = reinterpret_cast<float *>(X);
    float *aPointer = reinterpret_cast<float *>(A);
    cblas_ssyr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,aPointer,lda);
}

void Nd4jBlas::dsyr(Nd4jPointer *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer A, int lda){
    double *xPointer = reinterpret_cast<double *>(X);
    double *aPointer = reinterpret_cast<double *>(A);
    cblas_dsyr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,aPointer,lda);

}

/*
 * ------------------------------------------------------
 * SPR
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr(Nd4jPointer *extraParams,
                    int Order,
                    int Uplo,
                    int N,
                    float alpha,
                    Nd4jPointer X,
                    int incX,
                    Nd4jPointer Ap){
    float *xPointer = reinterpret_cast<float *>(X);
    float *apPointer = reinterpret_cast<float *>(Ap);
    cblas_sspr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,apPointer);
}

void Nd4jBlas::dspr(Nd4jPointer *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    Nd4jPointer X, int incX,
                    Nd4jPointer Ap){
    double *xPointer = reinterpret_cast<double *>(X);
    double *apPointer = reinterpret_cast<double *>(Ap);
    cblas_dspr(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,apPointer);

}

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer A, int lda) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_ssyr2(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,yPointer,incY,aPointer,lda);
}

void Nd4jBlas::dsyr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer A, int lda){
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

void Nd4jBlas::sspr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer Ap){
    float *apPointer = reinterpret_cast<float *>(Ap);
    float *xPointer = reinterpret_cast<float *>(X);
    float *yPointer = reinterpret_cast<float *>(Y);
    cblas_sspr2(convertOrder(Order),convertUplo(Uplo),N,alpha,xPointer,incX,yPointer,incY,apPointer);
}

void Nd4jBlas::dspr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     Nd4jPointer X, int incX,
                     Nd4jPointer Y, int incY,
                     Nd4jPointer Ap){
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

void Nd4jBlas::sgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     float beta,
                     Nd4jPointer C, int ldc) {
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_sgemm(convertOrder(Order),convertTranspose(TransA),convertTranspose(TransB),M,N,K,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

void Nd4jBlas::dgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     double beta,
                     Nd4jPointer C, int ldc){
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

void Nd4jBlas::ssymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     float beta,
                     Nd4jPointer C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_ssymm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),M,N,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

void Nd4jBlas::dsymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb,
                     double beta,
                     Nd4jPointer C, int ldc){
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

void Nd4jBlas::ssyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     float alpha,
                     Nd4jPointer A, int lda,
                     float beta,
                     Nd4jPointer C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_ssyrk(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,aPointer,lda,beta,cPointer,ldc);
}

void Nd4jBlas::dsyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     double alpha,
                     Nd4jPointer A, int lda,
                     double beta,
                     Nd4jPointer C, int ldc){
    double *aPointer = reinterpret_cast<double *>(A);
    double *cPointer = reinterpret_cast<double *>(C);
    cblas_dsyrk(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,aPointer,lda,beta,cPointer,ldc);
}

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      float alpha,
                      Nd4jPointer A, int lda,
                      Nd4jPointer B, int ldb,
                      float beta,
                      Nd4jPointer C, int ldc){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    float *cPointer = reinterpret_cast<float *>(C);
    cblas_ssyr2k(convertOrder(Trans),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,aPointer,lda,bPointer,ldb,beta,cPointer,ldc);
}

void Nd4jBlas::dsyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      double alpha,
                      Nd4jPointer A, int lda,
                      Nd4jPointer B, int ldb,
                      double beta,
                      Nd4jPointer C, int ldc) {
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

void Nd4jBlas::strmm(Nd4jPointer *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cblas_strmm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

void Nd4jBlas::dtrmm(Nd4jPointer *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cblas_dtrmm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

/*
 * ------------------------------------------------------
 * TRSM
 * ------------------------------------------------------
 */

void Nd4jBlas::strsm(Nd4jPointer *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     float alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb){
    float *aPointer = reinterpret_cast<float *>(A);
    float *bPointer = reinterpret_cast<float *>(B);
    cblas_strsm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

void Nd4jBlas::dtrsm(Nd4jPointer *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     Nd4jPointer A, int lda,
                     Nd4jPointer B, int ldb){
    double *aPointer = reinterpret_cast<double *>(A);
    double *bPointer = reinterpret_cast<double *>(B);
    cblas_dtrsm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,aPointer,lda,bPointer,ldb);
}

