//
// Created by agibsonccc on 1/26/16.
//

#ifndef NATIVEOPERATIONS_BLASIMPL_H
#define NATIVEOPERATIONS_BLASIMPL_H
#include <stdlib.h>
#include "java_cblas.h"
#include <cblas.h>
#include "cblas_enum_conversion.h"

#ifdef __cplusplus
extern "C" {
#endif

JavaVM *javavm;

JNIEXPORT jint JNICALL JNI_OnLoad (JavaVM *jvm, void *reserved) {
    javavm = jvm;
    return JNI_VERSION_1_2;
}

int cblas_errprn(int ierr, int info, char *form, ...) {
    JNIEnv *env;
    javavm->AttachCurrentThread((void **) &env, NULL);
    jclass iaexception = env->FindClass("java/lang/IllegalArgumentException");

    va_list argptr;
    va_start(argptr, form);

    char *message = (char *) malloc(vsnprintf(NULL, 0, form, argptr) + 1);
    vsprintf(message, form, argptr);

    env->ThrowNew(iaexception, message);
    va_end(argptr);
    if (ierr < info)
        return(ierr);
    else return(info);
};

void cblas_xerbla(int p, const char *rout, const char *form, ...) {
    // Override exit(-1) of the original cblas_xerbla.
    // In ATLAS, form is empty, so we have to use cblass_errprn
    // to get the error information.
    return;
};

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

JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sdsdot
        (JNIEnv *env, jclass clazz, jint N, jfloat alpha,
         jobject X, jint incX,
         jobject Y, jint incY) {

    float *cX = (float *)env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    return cblas_sdsdot(N, alpha, cX, incX, cY, incY);
};


JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsdot
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    return cblas_dsdot(N, cX, incX, cY, incY);
};

JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ddot
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    return cblas_ddot(N, cX, incX, cY, incY);
};

JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sdot
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    return cblas_sdot(N, cX, incX, cY, incY);
};

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_snrm2
        (JNIEnv *env, jclass clazz, jint N, jobject X, jint incX) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    return cblas_snrm2(N, cX, incX);
};

JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dnrm2
        (JNIEnv *env, jclass clazz, jint N, jobject X, jint incX) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    return cblas_dnrm2(N, cX, incX);
};

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

JNIEXPORT jfloat JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sasum
        (JNIEnv *env, jclass clazz, jint N, jobject X, jint incX) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    return cblas_sasum(N, cX, incX);

};

JNIEXPORT jdouble JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dasum
        (JNIEnv *env, jclass clazz, jint N, jobject X, jint incX) {
    double *cX = (double *) env->GetDirectBufferAddress(X);
    return cblas_dasum(N, cX, incX);
};

/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

JNIEXPORT jint JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_isamax
        (JNIEnv *env, jclass clazz, jint N, jobject X, jint incX) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    return cblas_isamax(N, cX, incX);
};

JNIEXPORT jint JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_idamax
        (JNIEnv *env, jclass clazz, jint N, jobject X, jint incX) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    return cblas_idamax(N, cX, incX);
};

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

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_srot
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY,
         jfloat c, jfloat s) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_srot(N, cX, incX, cY, incY, c, s);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_drot
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY,
         jdouble c, jdouble s) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_drot(N, cX, incX, cY, incY, c, s);
};

/*
 * ------------------------------------------------------
 * ROTG
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_srotg
        (JNIEnv *env, jclass clazz, jobject params) {

    float *ca = (float *) env->GetDirectBufferAddress(params);
    float *cb = ca + 1;
    float *cc = ca + 2;
    float *cs = ca + 3;
    cblas_srotg(ca, cb, cc, cs);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_drotg
        (JNIEnv *env, jclass clazz, jobject params) {

    double *ca = (double *) env->GetDirectBufferAddress(params);
    double *cb = ca + 1;
    double *cc = ca + 2;
    double *cs = ca + 3;
    cblas_drotg(ca, cb, cc, cs);
};

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_srotmg
        (JNIEnv *env, jclass clazz, jobject args, jobject P) {

    float *cargs = (float *) env->GetDirectBufferAddress(args);
    float *cP = (float *) env->GetDirectBufferAddress(P);
    cblas_srotmg(cargs, cargs + 1, cargs + 2, cargs[3], cP);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_drotmg
        (JNIEnv *env, jclass clazz, jobject args, jobject P) {

    double *cargs = (double *) env->GetDirectBufferAddress(args);
    double *cP = (double *) env->GetDirectBufferAddress(P);
    cblas_drotmg(cargs, cargs + 1, cargs + 2, cargs[3], cP);
};

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_srotm
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject P) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    float *cP = (float *) env->GetDirectBufferAddress(P);
    cblas_srotm(N, cX, incX, cY, incY, cP);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_drotm
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject P) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    double *cP = (double *) env->GetDirectBufferAddress(P);
    cblas_drotm(N, cX, incX, cY, incY, cP);
};

/*
 * ------------------------------------------------------
 * SWAP
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sswap
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_sswap(N, cX, incX, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dswap
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_dswap(N, cX, incX, cY, incY);

};

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sscal
        (JNIEnv *env, jclass clazz,
         jint N, jfloat alpha,
         jobject X, jint incX) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    cblas_sscal(N, alpha, cX, incX);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dscal
        (JNIEnv *env, jclass clazz,
         jint N, jdouble alpha,
         jobject X, jint incX) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    cblas_dscal(N, alpha, cX, incX);
};

/*
 * ------------------------------------------------------
 * SCOPY
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_scopy
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_scopy(N, cX, incX, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dcopy
        (JNIEnv *env, jclass clazz, jint N,
         jobject X, jint incX,
         jobject Y, jint incY) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_dcopy(N, cX, incX, cY, incY);
};

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_saxpy
        (JNIEnv *env, jclass clazz,
         jint N, jfloat alpha,
         jobject X, jint incX,
         jobject Y, jint incY) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_saxpy(N, alpha, cX, incX, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_daxpy
        (JNIEnv *env, jclass clazz,
         jint N, jdouble alpha,
         jobject X, jint incX,
         jobject Y, jint incY) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_daxpy(N, alpha, cX, incX, cY, incY);
};

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

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sgemv
        (JNIEnv * env, jclass clazz,
         jint Order, jint TransA,
         jint M, jint N,
         jfloat alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jfloat beta,
         jobject Y, jint incY) {


    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);

    cblas_sgemv(convertOrder(Order) , convertTranspose(TransA) , M, N, alpha, cA, lda, cX, incX, beta, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dgemv
        (JNIEnv * env, jclass clazz,
         jint Order, jint TransA,
         jint M, jint N,
         jdouble alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jdouble beta,
         jobject Y, jint incY) {


    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_dgemv(convertOrder(Order) , convertTranspose(TransA) , M, N, alpha, cA, lda, cX, incX, beta, cY, incY);
};

/*
 * ------------------------------------------------------
 * GBMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sgbmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint TransA,
         jint M, jint N,
         jint KL, jint KU,
         jfloat alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jfloat beta,
         jobject Y, jint incY) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_sgbmv(convertOrder(Order) , convertTranspose(TransA) , M, N, KL, KU,
                alpha, cA, lda, cX, incX, beta, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dgbmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint TransA,
         jint M, jint N,
         jint KL, jint KU,
         jdouble alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jdouble beta,
         jobject Y, jint incY) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_dgbmv(convertOrder(Order) , convertTranspose(TransA) , M, N, KL, KU,
                alpha, cA, lda, cX, incX, beta, cY, incY);
};

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ssymv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jfloat alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jfloat beta,
         jobject Y, jint incY) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_ssymv(convertOrder(Order) , convertUplo(Uplo), N, alpha, cA, lda, cX, incX, beta, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsymv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jdouble alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jdouble beta,
         jobject Y, jint incY) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_dsymv(convertOrder(Order) , convertUplo(Uplo), N, alpha, cA, lda, cX, incX, beta, cY, incY);
};

/*
 * ------------------------------------------------------
 * SBMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ssbmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N, int K,
         jfloat alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jfloat beta,
         jobject Y, jint incY) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_ssbmv(convertOrder(Order) , convertUplo(Uplo), N, K, alpha, cA, lda, cX, incX, beta, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsbmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N, int K,
         jdouble alpha,
         jobject A, jint lda,
         jobject X, jint incX,
         jdouble beta,
         jobject Y, jint incY) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_dsbmv(convertOrder(Order) , convertUplo(Uplo), N, K, alpha, cA, lda, cX, incX, beta, cY, incY);
};

/*
 * ------------------------------------------------------
 * SPMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sspmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jfloat alpha,
         jobject Ap,
         jobject X, jint incX,
         jfloat beta,
         jobject Y, jint incY) {

    float *cAp = (float *) env->GetDirectBufferAddress(Ap);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    cblas_sspmv(convertOrder(Order) , convertUplo(Uplo), N, alpha, cAp, cX, incX, beta, cY, incY);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dspmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jdouble alpha,
         jobject Ap,
         jobject X, jint incX,
         jdouble beta,
         jobject Y, jint incY) {

    double *cAp = (double *) env->GetDirectBufferAddress(Ap);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    cblas_dspmv(convertOrder(Order) , convertUplo(Uplo), N, alpha, cAp, cX, incX, beta, cY, incY);
};

/*
 * ------------------------------------------------------
 * TRMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_strmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA,
         jint N, jfloat alpha,
         jobject A, jint lda,
         jobject X, jint incX) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    cblas_strmv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(N), alpha, cA, lda, cX, incX);
};


JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtrmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA,
         jint N, jdouble alpha,
         jobject A, jint lda,
         jobject X, jint incX) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    cblas_dtrmv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(N), alpha, cA, lda, cX, incX);
};

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_stbmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N, jint K,
         jobject A, jint lda,
         jobject X, jint incX) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    cblas_strmv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), convertDiag(N), cA, lda, cX, incX);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtbmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N, jint K,
         jobject A, jint lda,
         jobject X, jint incX) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    cblas_dtrmv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, cA, lda, cX, incX);
};

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_stpmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N,
         jobject Ap,
         jobject X, jint incX) {

    float *cAp = (float *) env->GetDirectBufferAddress(Ap);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    cblas_stpmv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, cAp, cX, incX);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtpmv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N,
         jobject Ap,
         jobject X, jint incX) {

    double *cAp = (double *) env->GetDirectBufferAddress(Ap);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    cblas_dtpmv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, cAp, cX, incX);
};

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_strsv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N,
         jobject A, jint lda,
         jobject X, jint incX) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    cblas_strsv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, cA, lda, cX, incX);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtrsv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N,
         jobject A, jint lda,
         jobject X, jint incX) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    cblas_dtrsv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, cA, lda, cX, incX);
};

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_stbsv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N, jint K,
         jobject A, jint lda,
         jobject X, jint incX) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    cblas_stbsv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, K, cA, lda, cX, incX);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtbsv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N, jint K,
         jobject A, jint lda,
         jobject X, jint incX) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    cblas_dtbsv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, K, cA, lda, cX, incX);
};

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_stpsv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N,
         jobject Ap,
         jobject X, jint incX) {

    float *cAp = (float *) env->GetDirectBufferAddress(Ap);
    float *cX = (float *) env->GetDirectBufferAddress(X);
    cblas_stpsv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, cAp, cX, incX);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtpsv
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint TransA, jint Diag,
         jint N,
         jobject Ap,
         jobject X, jint incX) {

    double *cAp = (double *) env->GetDirectBufferAddress(Ap);
    double *cX = (double *) env->GetDirectBufferAddress(X);
    cblas_dtpsv(convertOrder(Order) , convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), N, cAp, cX, incX);
};

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sger
        (JNIEnv *env, jclass clazz,
         jint Order,
         jint M, jint N,
         jfloat alpha,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject A, jint lda) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    float *cA = (float *) env->GetDirectBufferAddress(A);
    cblas_sger(convertOrder(Order) , M, N, alpha, cX, incX, cY, incY, cA, lda);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dger
        (JNIEnv *env, jclass clazz,
         jint Order,
         jint M, jint N,
         jdouble alpha,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject A, jint lda) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    double *cA = (double *) env->GetDirectBufferAddress(A);
    cblas_dger(convertOrder(Order) , M, N, alpha, cX, incX, cY, incY, cA, lda);
};

/*
 * ------------------------------------------------------
 * SYR
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ssyr
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jfloat alpha,
         jobject X, jint incX,
         jobject A, jint lda) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cA = (float *) env->GetDirectBufferAddress(A);
    cblas_ssyr(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cA, lda);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsyr
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jdouble alpha,
         jobject X, jint incX,
         jobject A, jint lda) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cA = (double *) env->GetDirectBufferAddress(A);
    cblas_dsyr(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cA, lda);
}

/*
 * ------------------------------------------------------
 * SPR
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sspr
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jfloat alpha,
         jobject X, jint incX,
         jobject Ap) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cAp = (float *) env->GetDirectBufferAddress(Ap);
    cblas_sspr(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cAp);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dspr
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jdouble alpha,
         jobject X, jint incX,
         jobject Ap) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cAp = (double *) env->GetDirectBufferAddress(Ap);
    cblas_dspr(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cAp);
};

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ssyr2
        (JNIEnv *env, jclass clazz,
         jint Order, int Uplo,
         jint N,
         jfloat alpha,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject A, jint lda) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    float *cA = (float *) env->GetDirectBufferAddress(A);
    cblas_ssyr2(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cY, incY, cA, lda);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsyr2
        (JNIEnv *env, jclass clazz,
         jint Order, int Uplo,
         jint N,
         jdouble alpha,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject A, jint lda) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    double *cA = (double *) env->GetDirectBufferAddress(A);
    cblas_dsyr2(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cY, incY, cA, lda);
};

/*
 * ------------------------------------------------------
 * SPR2
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sspr2
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jfloat alpha,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject Ap) {

    float *cX = (float *) env->GetDirectBufferAddress(X);
    float *cY = (float *) env->GetDirectBufferAddress(Y);
    float *cAp = (float *) env->GetDirectBufferAddress(Ap);
    cblas_sspr2(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cY, incY, cAp);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dspr2
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo,
         jint N,
         jdouble alpha,
         jobject X, jint incX,
         jobject Y, jint incY,
         jobject Ap) {

    double *cX = (double *) env->GetDirectBufferAddress(X);
    double *cY = (double *) env->GetDirectBufferAddress(Y);
    double *cAp = (double *) env->GetDirectBufferAddress(Ap);
    cblas_dspr2(convertOrder(Order) , convertUplo(Uplo), N, alpha, cX, incX, cY, incY, cAp);
};

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

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_sgemm
        (JNIEnv *env, jclass clazz,
         jint Order, jint TransA, jint TransB,
         jint M, jint N, jint K,
         jfloat alpha,
         jobject A, jint lda,
         jobject B, jint ldb,
         jfloat beta,
         jobject C, jint ldc) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cB = (float *) env->GetDirectBufferAddress(B);
    float *cC = (float *) env->GetDirectBufferAddress(C);
    cblas_sgemm(convertOrder(Order) , convertTranspose(TransA) , convertTranspose(TransB) , M, N, K, alpha, cA, lda, cB, ldb, beta, cC, ldc);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dgemm
        (JNIEnv *env, jclass clazz,
         jint Order, jint TransA, jint TransB,
         jint M, jint N, jint K,
         jdouble alpha,
         jobject A, jint lda,
         jobject B, jint ldb,
         jdouble beta,
         jobject C, jint ldc) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cB = (double *) env->GetDirectBufferAddress(B);
    double *cC = (double *) env->GetDirectBufferAddress(C);
    cblas_dgemm(convertOrder(Order) , convertTranspose(TransA) , convertTranspose(TransB) , M, N, K, alpha, cA, lda, cB, ldb, beta, cC, ldc);
};

/*
 * ------------------------------------------------------
 * SYMM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ssymm
        (JNIEnv *env, jclass clazz,
         jint Order, jint Side, jint Uplo,
         jint M, jint N,
         jfloat alpha,
         jobject A, jint lda,
         jobject B, jint ldb,
         jfloat beta,
         jobject C, jint ldc) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cB = (float *) env->GetDirectBufferAddress(B);
    float *cC = (float *) env->GetDirectBufferAddress(C);

    cblas_ssymm(convertOrder(Order) , convertSide(Side), convertUplo(Uplo), M, N, alpha, cA, lda, cB, ldb, beta, cC, ldc);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsymm
        (JNIEnv *env, jclass clazz,
         jint Order, jint Side, jint Uplo,
         jint M, jint N,
         jdouble alpha,
         jobject A, jint lda,
         jobject B, jint ldb,
         jdouble beta,
         jobject C, jint ldc) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cB = (double *) env->GetDirectBufferAddress(B);
    double *cC = (double *) env->GetDirectBufferAddress(C);
    cblas_dsymm(convertOrder(Order) , convertSide(Side), convertUplo(Uplo), M, N, alpha, cA, lda, cB, ldb, beta, cC, ldc);
};

/*
 * ------------------------------------------------------
 * SYRK
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ssyrk
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint Trans,
         jint N, jint K,
         jfloat alpha,
         jobject A, jint lda,
         jfloat beta,
         jobject C, jint ldc) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cC = (float *) env->GetDirectBufferAddress(C);

    cblas_ssyrk(convertOrder(Order) , convertUplo(Uplo), convertTranspose(Trans), N, K, alpha, cA, lda, beta, cC, ldc);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsyrk
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint Trans,
         jint N, jint K,
         jdouble alpha,
         jobject A, jint lda,
         jdouble beta,
         jobject C, jint ldc) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cC = (double *) env->GetDirectBufferAddress(C);
    cblas_dsyrk(convertOrder(Order) , convertUplo(Uplo), convertTranspose(Trans), N, K, alpha, cA, lda, beta, cC, ldc);
};

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_ssyr2k
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint Trans,
         jint N, jint K,
         jfloat alpha,
         jobject A, jint lda,
         jobject B, jint ldb,
         jfloat beta,
         jobject C, jint ldc) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cB = (float *) env->GetDirectBufferAddress(B);
    float *cC = (float *) env->GetDirectBufferAddress(C);
    cblas_ssyr2k(convertOrder(Order) , convertUplo(Uplo), convertTranspose(Trans), N, K, alpha, cA, lda, cB, ldb, beta, cC, ldc);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dsyr2k
        (JNIEnv *env, jclass clazz,
         jint Order, jint Uplo, jint Trans,
         jint N, jint K,
         jdouble alpha,
         jobject A, jint lda,
         jobject B, jint ldb,
         jdouble beta,
         jobject C, jint ldc) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cB = (double *) env->GetDirectBufferAddress(B);
    double *cC = (double *) env->GetDirectBufferAddress(C);
    cblas_dsyr2k(convertOrder(Order) , convertUplo(Uplo), convertTranspose(Trans), N, K, alpha, cA, lda, cB, ldb, beta, cC, ldc);
};

/*
 * ------------------------------------------------------
 * TRMM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_strmm
        (JNIEnv *env, jclass clazz,
         jint Order, jint Side,
         jint Uplo, jint TransA, jint Diag,
         jint M, jint N,
         jfloat alpha,
         jobject A, jint lda,
         jobject B, jint ldb) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cB = (float *) env->GetDirectBufferAddress(B);
    cblas_strmm(convertOrder(Order) , convertSide(Side), convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), M, N, alpha, cA, lda, cB, ldb);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtrmm
        (JNIEnv *env, jclass clazz,
         jint Order, jint Side,
         jint Uplo, jint TransA, jint Diag,
         jint M, jint N,
         jdouble alpha,
         jobject A, jint lda,
         jobject B, jint ldb) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cB = (double *) env->GetDirectBufferAddress(B);
    cblas_dtrmm(convertOrder(Order) , convertSide(Side), convertUplo(Uplo), convertTranspose(TransA) ,
                convertDiag(Diag), M, N, alpha, cA, lda, cB, ldb);
};

/*
 * ------------------------------------------------------
 * TRSM
 * ------------------------------------------------------
 */

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_strsm
        (JNIEnv *env, jclass clazz,
         jint Order, jint Side,
         jint Uplo, jint TransA, jint Diag,
         jint M, jint N,
         jfloat alpha,
         jobject A, jint lda,
         jobject B, jint ldb) {

    float *cA = (float *) env->GetDirectBufferAddress(A);
    float *cB = (float *) env->GetDirectBufferAddress(B);

    cblas_strsm(convertOrder(Order) , convertSide(Side), convertUplo(Uplo), convertTranspose(TransA) , convertDiag(Diag), M, N, alpha, cA, lda, cB, ldb);
};

JNIEXPORT void JNICALL Java_org_nd4j_linalg_cpu_nativecpu_CBLAS_dtrsm
        (JNIEnv *env, jclass clazz,
         jint Order, jint Side,
         jint Uplo, jint TransA, jint Diag,
         jint M, jint N,
         jdouble alpha,
         jobject A, jint lda,
         jobject B, jint ldb) {

    double *cA = (double *) env->GetDirectBufferAddress(A);
    double *cB = (double *) env->GetDirectBufferAddress(B);
    cblas_dtrsm(convertOrder(Order) , convertSide(Side), convertUplo(Uplo), convertTranspose(TransA) , convertDiag(Diag), M, N, alpha, cA, lda, cB, ldb);

};

#ifdef __cplusplus
}
#endif
#endif //NATIVEOPERATIONS_BLASIMPL_H
