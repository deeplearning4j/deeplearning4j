//
// Created by agibsonccc on 2/21/16.
//

#include "../NativeBlas.h"
#include <dll.h>
#include <cblas.h>
#include <pointercast.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif



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

int maxThreads = 8;
int vendor = 0;

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

typedef void* (*void_int)(int);
typedef int* (*int_int)(int);
typedef int* (*int_int_int)(int, int);

void blas_set_num_threads(int num) {
#ifdef __MKL
    // if we're linked against mkl - just go for it
    MKL_Set_Num_Threads(num);
    MKL_Domain_Set_Num_Threads(num, 0); // MKL_DOMAIN_ALL
    MKL_Domain_Set_Num_Threads(num, 1); // MKL_DOMAIN_BLAS
    MKL_Set_Num_Threads_Local(num);
#elif __OPENBLAS
#ifdef _WIN32
    // for win32 we just check for mkl_rt.dll
    HMODULE handle = LoadLibrary("mkl_rt.dll");
    if (handle != NULL) {
        void_int mkl_global = (void_int) GetProcAddress(handle, "MKL_Set_Num_Threads");
        if (mkl_global != NULL) {
            mkl_global(num);

            vendor = 3;

            int_int_int mkl_domain = (int_int_int) GetProcAddress(handle, "MKL_Domain_Set_Num_Threads");
            if (mkl_domain != NULL) {
                mkl_domain(num, 0); // DOMAIN_ALL
                mkl_domain(num, 1); // DOMAIN_BLAS
            }

            int_int mkl_local = (int_int) GetProcAddress(handle, "MKL_Set_Num_Threads_Local");
            if (mkl_local != NULL) {
                mkl_local(num);
            }
        } else {
            printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
        }
        //FreeLibrary(handle);
    } else {
      // OpenBLAS path
      handle = LoadLibrary("libopenblas.dll");
      if (handle != NULL) {
        void_int oblas = (void_int) GetProcAddress(handle, "openblas_set_num_threads");
        if (oblas != NULL) {
            vendor = 2;
            oblas(num);
        } else {
            printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
        }
        //FreeLibrary(handle);
      } else {
        printf("Unable to guess runtime. Please set OMP_NUM_THREADS manually.\n");
      }
    }
#elif __APPLE__
   // do nothing for MacOS
   printf("Unable to guess runtime. Please set OMP_NUM_THREADS or equivalent manually.\n");
#else
    // it's possible to have MKL being loaded at runtime
    void *handle = dlopen("libmkl_rt.so", RTLD_NOW|RTLD_GLOBAL);
    if (handle != NULL) {

        // we call for openblas only if libmkl isn't loaded, and openblas_set_num_threads exists
        void_int mkl_global = (void_int) dlsym(handle, "MKL_Set_Num_Threads");
        if (mkl_global != NULL) {
            // we're running against mkl
            mkl_global((int) num);

            vendor = 3;

            int_int_int mkl_domain = (int_int_int) dlsym(handle, "MKL_Domain_Set_Num_Threads");
            if (mkl_domain != NULL) {
                mkl_domain(num, 0); // DOMAIN_ALL
                mkl_domain(num, 1); // DOMAIN_BLAS
            }

            int_int mkl_local = (int_int) dlsym(handle, "MKL_Set_Num_Threads_Local");
            if (mkl_local != NULL) {
                mkl_local(num);
            }
        } else {
            printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
        }
        dlclose(handle);
    } else {
        // we're falling back to bundled OpenBLAS opening libnd4j.so
        handle = dlopen("libnd4j.so", RTLD_NOW|RTLD_GLOBAL);

        if (handle != NULL) {
            void_int oblas = (void_int) dlsym(handle, "openblas_set_num_threads");
            if (oblas != NULL) {
                vendor = 2;
                // we're running against openblas
                oblas((int) num);
            } else {
                printf("Unable to tune runtime. Please set OMP_NUM_THREADS manually.\n");
            }

            dlclose(handle);
        } else printf("Unable to guess runtime. Please set OMP_NUM_THREADS manually.\n");
    }
#endif

#else
    printf("Unable to guess runtime. Please set OMP_NUM_THREADS or equivalent manually.\n");
#endif
    fflush(stdout);
}


void Nd4jBlas::setMaxThreads(int num) {
    blas_set_num_threads(num);
    maxThreads = num;
}

int Nd4jBlas::getMaxThreads() {
    return maxThreads;
}

int Nd4jBlas::getVendor() {
    // 0 - Unknown
    // 1 - cuBLAS
    // 2 - openBLAS
    // 3 - MKL
    return vendor;
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
                       float *X, int incX,
                       float *Y, int incY) {
    return cblas_sdsdot(N,alpha,X,incX,Y,incY);

}

double Nd4jBlas::dsdot(Nd4jPointer *extraParams,int N,
                       float *X, int incX,
                       float *Y, int incY) {
    return cblas_dsdot(N,X,incX,Y,incY);
}

double Nd4jBlas::ddot(Nd4jPointer *extraParams,int N,
                      double *X, int incX,
                      double *Y, int incY) {
    return cblas_ddot(N,X,incX,Y,incY);
}

float Nd4jBlas::sdot(Nd4jPointer *extraParams,int N,
                     float *X, int incX,
                     float *Y, int incY) {
    return cblas_sdot(N,X,incX,Y,incY);
}

/*
 * ------------------------------------------------------
 * NRM2
 * ------------------------------------------------------
 */

float Nd4jBlas::snrm2(Nd4jPointer *extraParams,int N, float *X, int incX) {
    return cblas_snrm2(N,X,incX);

}

double Nd4jBlas::dnrm2(Nd4jPointer *extraParams,int N, double *X, int incX) {
    return cblas_dnrm2(N,X,incX);
}

/*
 * ------------------------------------------------------
 * ASUM
 * ------------------------------------------------------
 */

float Nd4jBlas::sasum(Nd4jPointer *extraParams,int N, float *X, int incX) {
    return cblas_sasum(N,X,incX);

}

double Nd4jBlas::dasum(Nd4jPointer *extraParams,int N, double *X, int incX) {
    return cblas_dasum(N,X,incX);

}

/*
 * ------------------------------------------------------
 * IAMAX
 * ------------------------------------------------------
 */

int Nd4jBlas::isamax(Nd4jPointer *extraParams,int N, float *X, int incX){
    return cblas_isamax(N,X,incX);

}

int Nd4jBlas::idamax(Nd4jPointer *extraParams,int N, double *X, int incX) {
    return cblas_idamax(N,X,incX);

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
                    float *X, int incX,
                    float *Y, int incY,
                    float c, float s) {
    cblas_srot(N,X,incX,Y,incY,c,s);

}

void Nd4jBlas::drot(Nd4jPointer *extraParams,int N,
                    double *X, int incX,
                    double *Y, int incY,
                    double c, double s) {
    cblas_drot(N,X,incX,Y,incY,c,s);
}

/*
 * ------------------------------------------------------
 * ROTG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotg(Nd4jPointer *extraParams,float *args) {
    return cblas_srotg(&args[0],&args[1],&args[2],&args[3]);
}

void Nd4jBlas::drotg(Nd4jPointer *extraParams,double *args) {
    cblas_drotg(&args[0],&args[1],&args[2],&args[3]);

}

/*
 * ------------------------------------------------------
 * ROTMG
 * ------------------------------------------------------
 */

void Nd4jBlas::srotmg(Nd4jPointer *extraParams,float *args,
                      float *P) {
    return cblas_srotmg(&args[0],&args[1],&args[2],args[3],P);

}

void Nd4jBlas::drotmg(Nd4jPointer *extraParams,double *args,
                      double *P) {
    cblas_drotmg(&args[0],&args[1],&args[2],args[3],P);

}

/*
 * ------------------------------------------------------
 * ROTM
 * ------------------------------------------------------
 */

void Nd4jBlas::srotm(Nd4jPointer *extraParams,int N,
                     float *X, int incX,
                     float *Y, int incY,
                     float *P) {
    cblas_srotm(N,X,incX,Y,incY,P);

}

void Nd4jBlas::drotm(Nd4jPointer *extraParams,int N,
                     double *X, int incX,
                     double *Y, int incY,
                     double *P) {
    cblas_drotm(N,X,incX,Y,incY,P);

}

/*
 * ------------------------------------------------------
 * SWAP
 * ------------------------------------------------------
 */

void Nd4jBlas::sswap(Nd4jPointer *extraParams,int N,
                     float *X, int incX,
                     float *Y, int incY) {
    cblas_sswap(N,X,incX,Y,incY);
}

void Nd4jBlas::dswap(Nd4jPointer *extraParams,int N,
                     double *X, int incX,
                     double *Y, int incY) {
    cblas_dswap(N,X,incX,Y,incY);
}

/*
 * ------------------------------------------------------
 * SCAL
 * ------------------------------------------------------
 */

void Nd4jBlas::sscal(Nd4jPointer *extraParams,int N, float alpha,
                     float *X, int incX){
    cblas_sscal(N,alpha,X,incX);

}

void Nd4jBlas::dscal(Nd4jPointer *extraParams,int N, double alpha,
                     double *X, int incX){
    cblas_dscal(N,alpha,X,incX);

}

/*
 * ------------------------------------------------------
 * SCOPY
 * ------------------------------------------------------
 */

void Nd4jBlas::scopy(Nd4jPointer *extraParams,int N,
                     float *X, int incX,
                     float *Y, int incY) {
    cblas_scopy(N,X,incX,Y,incY);
}

void Nd4jBlas::dcopy(Nd4jPointer *extraParams,int N,
                     double *X, int incX,
                     double *Y, int incY){
    cblas_dcopy(N,X,incX,Y,incY);
}

/*
 * ------------------------------------------------------
 * AXPY
 * ------------------------------------------------------
 */

void Nd4jBlas::saxpy(Nd4jPointer *extraParams,int N, float alpha,
                     float *X, int incX,
                     float *Y, int incY) {
    cblas_saxpy(N,alpha,X,incX,Y,incY);
}

void Nd4jBlas::daxpy(Nd4jPointer *extraParams,int N, double alpha,
                     double *X, int incX,
                     double *Y, int incY) {
    cblas_daxpy(N,alpha,X,incX,Y,incY);
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
                     float *A, int lda,
                     float *X, int incX,
                     float beta,
                     float *Y, int incY) {
    cblas_sgemv(convertOrder(Order),convertTranspose(TransA),M,N,alpha,A,lda,X,incX,beta,Y,incY);

}

void Nd4jBlas::dgemv(Nd4jPointer *extraParams,int Order, int TransA,
                     int M, int N,
                     double alpha,
                     double *A, int lda,
                     double *X, int incX,
                     double beta,
                     double *Y, int incY) {
    cblas_dgemv(convertOrder(Order),convertTranspose(TransA),M,N,alpha,A,lda,X,incX,beta,Y,incY);
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
                     float *A, int lda,
                     float *X, int incX,
                     float beta,
                     float *Y, int incY) {
    cblas_sgbmv(convertOrder(Order),convertTranspose(TransA),N,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY);
}

void Nd4jBlas::dgbmv(Nd4jPointer *extraParams,int Order, int TransA,
                     int M, int N,
                     int KL, int KU,
                     double alpha,
                     double *A, int lda,
                     double *X, int incX,
                     double beta,
                     double *Y, int incY) {
    cblas_dgbmv(convertOrder(Order),convertTranspose(TransA),M,N,KL,KU,alpha,A,lda,X,incX,beta,Y,incY);
}

/*
 * ------------------------------------------------------
 * SYMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     float *A, int lda,
                     float *X, int incX,
                     float beta,
                     float *Y, int incY) {
    cblas_ssymv(convertOrder(Order),convertUplo(Uplo),N,alpha,A,lda,X,incX,beta,Y,incY);
}

void Nd4jBlas::dsymv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     double *A, int lda,
                     double *X, int incX,
                     double beta,
                     double *Y, int incY) {
    cblas_dsymv(convertOrder(Order),convertUplo(Uplo),N,alpha,A,lda,X,incX,beta,Y,incY);
}

/*
 * ------------------------------------------------------
 * SBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::ssbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N, int K,
                     float alpha,
                     float *A, int lda,
                     float *X, int incX,
                     float beta,
                     float *Y, int incY) {
    cblas_ssbmv(convertOrder(Order),convertUplo(Uplo),N,K,alpha,A,lda,X,incX,beta,Y,incY);
}

void Nd4jBlas::dsbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N, int K,
                     double alpha,
                     double *A, int lda,
                     double *X, int incX,
                     double beta,
                     double *Y, int incY){
    cblas_dsbmv(convertOrder(Order),convertUplo(Uplo),N,K,alpha,A,lda,X,incX,beta,Y,incY);
}

/*
 * ------------------------------------------------------
 * SPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::sspmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     float *Ap,
                     float *X, int incX,
                     float beta,
                     float *Y, int incY){
    cblas_sspmv(convertOrder(Order),convertUplo(Uplo),N,alpha,Ap,X,incX,beta,Y,incY);
}

void Nd4jBlas::dspmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     double *Ap,
                     double *X, int incX,
                     double beta,
                     double *Y, int incY){
    cblas_dspmv(convertOrder(Order),convertUplo(Uplo),N,alpha,Ap,X,incX,beta,Y,incY);
}

/*
 * ------------------------------------------------------
 * TRMV
 * ------------------------------------------------------
 */

void Nd4jBlas::strmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,
                     int Diag,
                     int N, float alpha,
                     float *A, int lda,
                     float *X, int incX){
    cblas_strmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,A,lda,X,incX);
}

void Nd4jBlas::dtrmv(Nd4jPointer *extraParams,int Order, int Uplo, int TransA,int Diag,
                     int N, double alpha,
                     double *A, int lda,
                     double *X, int incX){
    cblas_dtrmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,A,lda,X,incX);
}

/*
 * ------------------------------------------------------
 * TBMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     float *A, int lda,
                     float *X, int incX){
    cblas_stbmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,A,lda,X,incX);
}

void Nd4jBlas::dtbmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     double *A, int lda,
                     double *X, int incX){
    cblas_dtbmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,A,lda,X,incX);
}

/*
 * ------------------------------------------------------
 * TPMV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     float *Ap,
                     float *X, int incX){
    cblas_stpmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,Ap,X,incX);
}

void Nd4jBlas::dtpmv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     double *Ap,
                     double *X, int incX) {
    cblas_dtpmv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,Ap,X,incX);

}

/*
 * ------------------------------------------------------
 * TRSV
 * ------------------------------------------------------
 */

void Nd4jBlas::strsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     float *A, int lda,
                     float *X, int incX){
    cblas_strsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,A,lda,X,incX);
}

void Nd4jBlas::dtrsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     double *A, int lda,
                     double *X, int incX){
    cblas_dtrsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,A,lda,X,incX);

}

/*
 * ------------------------------------------------------
 * TBSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stbsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     float *A, int lda,
                     float *X, int incX) {
    cblas_stbsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,A,lda,X,incX);

}

void Nd4jBlas::dtbsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N, int K,
                     double *A, int lda,
                     double *X, int incX){
    cblas_dtbsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,K,A,lda,X,incX);

}

/*
 * ------------------------------------------------------
 * TPSV
 * ------------------------------------------------------
 */

void Nd4jBlas::stpsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     float *Ap,
                     float *X, int incX){
    cblas_stpsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,Ap,X,incX);
}

void Nd4jBlas::dtpsv(Nd4jPointer *extraParams,int Order, int Uplo,
                     int TransA, int Diag,
                     int N,
                     double *Ap,
                     double *X, int incX){
    cblas_dtpsv(convertOrder(Order),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),N,Ap,X,incX);

}

/*
 * ------------------------------------------------------
 * GER
 * ------------------------------------------------------
 */

void Nd4jBlas::sger(Nd4jPointer *extraParams,int Order,
                    int M, int N,
                    float alpha,
                    float *X, int incX,
                    float *Y, int incY,
                    float *A, int lda){
    cblas_sger(convertOrder(Order),M,N,alpha,X,incX,Y,incY,A,lda);
}

void Nd4jBlas::dger(Nd4jPointer *extraParams,int Order,
                    int M, int N,
                    double alpha,
                    double *X, int incX,
                    double *Y, int incY,
                    double *A, int lda){
    cblas_dger(convertOrder(Order),M,N,alpha,X,incX,Y,incY,A,lda);
}

/*
 * ------------------------------------------------------
 * SYR
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr(Nd4jPointer *extraParams,int Order, int Uplo,
                    int N,
                    float alpha,
                    float *X, int incX,
                    float *A, int lda){
    cblas_ssyr(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,A,lda);
}

void Nd4jBlas::dsyr(Nd4jPointer *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    double *X, int incX,
                    double *A, int lda){
    cblas_dsyr(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,A,lda);

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
                    float *X,
                    int incX,
                    float *Ap){
    cblas_sspr(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,Ap);
}

void Nd4jBlas::dspr(Nd4jPointer *extraParams,int Order, int Uplo,
                    int N,
                    double alpha,
                    double *X, int incX,
                    double *Ap){
    cblas_dspr(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,Ap);

}

/*
 * ------------------------------------------------------
 * SYR2
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     float *X, int incX,
                     float *Y, int incY,
                     float *A, int lda) {
    cblas_ssyr2(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,Y,incY,A,lda);
}

void Nd4jBlas::dsyr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     double *X, int incX,
                     double *Y, int incY,
                     double *A, int lda){
    cblas_dsyr2(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,Y,incY,A,lda);
}

/*
 * ------------------------------------------------------
 * SPR2
 * ------------------------------------------------------
 */

void Nd4jBlas::sspr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     float alpha,
                     float *X, int incX,
                     float *Y, int incY,
                     float *Ap){
    cblas_sspr2(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,Y,incY,Ap);
}

void Nd4jBlas::dspr2(Nd4jPointer *extraParams,int Order, int Uplo,
                     int N,
                     double alpha,
                     double *X, int incX,
                     double *Y, int incY,
                     double *Ap){
    cblas_dspr2(convertOrder(Order),convertUplo(Uplo),N,alpha,X,incX,Y,incY,Ap);
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
void Nd4jBlas::hgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha,
                     float16 *A, int lda,
                     float16 *B, int ldb,
                     float beta,
                     float16 *C, int ldc) {
    // no-op
}

void Nd4jBlas::sgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     float alpha,
                     float *A, int lda,
                     float *B, int ldb,
                     float beta,
                     float *C, int ldc) {
    cblas_sgemm(convertOrder(Order),convertTranspose(TransA),convertTranspose(TransB),M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}

void Nd4jBlas::dgemm(Nd4jPointer *extraParams,int Order, int TransA, int TransB,
                     int M, int N, int K,
                     double alpha,
                     double *A, int lda,
                     double *B, int ldb,
                     double beta,
                     double *C, int ldc){
    cblas_dgemm(convertOrder(Order),convertTranspose(TransA),convertTranspose(TransB),M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}

/*
 * ------------------------------------------------------
 * SYMM
 * ------------------------------------------------------
 */

void Nd4jBlas::ssymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     float alpha,
                     float *A, int lda,
                     float *B, int ldb,
                     float beta,
                     float *C, int ldc){
    cblas_ssymm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),M,N,alpha,A,lda,B,ldb,beta,C,ldc);
}

void Nd4jBlas::dsymm(Nd4jPointer *extraParams,int Order, int Side, int Uplo,
                     int M, int N,
                     double alpha,
                     double *A, int lda,
                     double *B, int ldb,
                     double beta,
                     double *C, int ldc){
    cblas_dsymm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),M,N,alpha,A,lda,B,ldb,beta,C,ldc);
}

/*
 * ------------------------------------------------------
 * SYRK
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     float alpha,
                     float *A, int lda,
                     float beta,
                     float *C, int ldc){
    cblas_ssyrk(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,A,lda,beta,C,ldc);
}

void Nd4jBlas::dsyrk(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                     int N, int K,
                     double alpha,
                     double *A, int lda,
                     double beta,
                     double *C, int ldc){
    cblas_dsyrk(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,A,lda,beta,C,ldc);
}

/*
 * ------------------------------------------------------
 * SYR2K
 * ------------------------------------------------------
 */

void Nd4jBlas::ssyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      float alpha,
                      float *A, int lda,
                      float *B, int ldb,
                      float beta,
                      float *C, int ldc){
    cblas_ssyr2k(convertOrder(Trans),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}

void Nd4jBlas::dsyr2k(Nd4jPointer *extraParams,int Order, int Uplo, int Trans,
                      int N, int K,
                      double alpha,
                      double *A, int lda,
                      double *B, int ldb,
                      double beta,
                      double *C, int ldc) {
    cblas_dsyr2k(convertOrder(Order),convertUplo(Uplo),convertTranspose(Trans),N,K,alpha,A,lda,B,ldb,beta,C,ldc);
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
                     float *A, int lda,
                     float *B, int ldb){
    cblas_strmm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,A,lda,B,ldb);
}

void Nd4jBlas::dtrmm(Nd4jPointer *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     double *A, int lda,
                     double *B, int ldb){
    cblas_dtrmm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,A,lda,B,ldb);
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
                     float *A, int lda,
                     float *B, int ldb){
    cblas_strsm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,A,lda,B,ldb);
}

void Nd4jBlas::dtrsm(Nd4jPointer *extraParams,int Order, int Side,
                     int Uplo, int TransA, int Diag,
                     int M, int N,
                     double alpha,
                     double *A, int lda,
                     double *B, int ldb){
    cblas_dtrsm(convertOrder(Order),convertSide(Side),convertUplo(Uplo),convertTranspose(TransA),convertDiag(Diag),M,N,alpha,A,lda,B,ldb);
}

