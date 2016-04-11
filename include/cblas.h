//
// Created by agibsonccc on 1/26/16.
//

#ifndef NATIVEOPERATIONS_CBLAS_H
#define NATIVEOPERATIONS_CBLAS_H
#ifndef CBLAS_H
#include <dll.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifndef CBLAS_ENUM_DEFINED_H
#define CBLAS_ENUM_DEFINED_H
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
    AtlasConj=114};
enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG  {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE  {CblasLeft=141, CblasRight=142};
#endif

#ifndef CBLAS_ENUM_ONLY
#define CBLAS_H
#define CBLAS_INDEX int

int cblas_errprn(int ierr, int info, char *form, ...);
void cblas_xerbla(int p, const char *rout, const char *form, ...);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
float  cblas_sdsdot(int N, const float alpha, const float *X,
                    int incX, const float *Y, int incY);
double cblas_dsdot(int N, const float *X, int incX, const float *Y,
                   int incY);
float  cblas_sdot(int N, const float  *X, int incX,
                  const float  *Y, int incY);
double cblas_ddot(int N, const double *X, int incX,
                  const double *Y, int incY);
/*
 * Functions having prefixes Z and C only
 */
void   cblas_cdotu_sub(int N, const void *X, int incX,
                       const void *Y, int incY, void *dotu);
void   cblas_cdotc_sub(int N, const void *X, int incX,
                       const void *Y, int incY, void *dotc);

void   cblas_zdotu_sub(int N, const void *X, int incX,
                       const void *Y, int incY, void *dotu);
void   cblas_zdotc_sub(int N, const void *X, int incX,
                       const void *Y, int incY, void *dotc);


/*
 * Functions having prefixes S D SC DZ
 */
float  cblas_snrm2(int N, const float *X, int incX);
float  cblas_sasum(int N, const float *X, int incX);

double cblas_dnrm2(int N, const double *X, int incX);
double cblas_dasum(int N, const double *X, int incX);

float  cblas_scnrm2(int N, const void *X, int incX);
float  cblas_scasum(int N, const void *X, int incX);

double cblas_dznrm2(int N, const void *X, int incX);
double cblas_dzasum(int N, const void *X, int incX);


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX cblas_isamax(int N, const float  *X, int incX);
CBLAS_INDEX cblas_idamax(int N, const double *X, int incX);
CBLAS_INDEX cblas_icamax(int N, const void   *X, int incX);
CBLAS_INDEX cblas_izamax(int N, const void   *X, int incX);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void cblas_sswap(int N, float *X, int incX,
                 float *Y, int incY);
void cblas_scopy(int N, const float *X, int incX,
                 float *Y, int incY);
void cblas_saxpy(int N, const float alpha, const float *X,
                 int incX, float *Y, int incY);
void catlas_saxpby(int N, const float alpha, const float *X,
                   int incX, const float beta, float *Y, int incY);
void catlas_sset
        (int N, const float alpha, float *X, int incX);

void cblas_dswap(int N, double *X, int incX,
                 double *Y, int incY);
void cblas_dcopy(int N, const double *X, int incX,
                 double *Y, int incY);
void cblas_daxpy(int N, const double alpha, const double *X,
                 int incX, double *Y, int incY);
void catlas_daxpby(int N, const double alpha, const double *X,
                   int incX, const double beta, double *Y, int incY);
void catlas_dset
        (int N, const double alpha, double *X, int incX);

void cblas_cswap(int N, void *X, int incX,
                 void *Y, int incY);
void cblas_ccopy(int N, const void *X, int incX,
                 void *Y, int incY);
void cblas_caxpy(int N, const void *alpha, const void *X,
                 int incX, void *Y, int incY);
void catlas_caxpby(int N, const void *alpha, const void *X,
                   int incX, const void *beta, void *Y, int incY);
void catlas_cset
        (int N, const void *alpha, void *X, int incX);

void cblas_zswap(int N, void *X, int incX,
                 void *Y, int incY);
void cblas_zcopy(int N, const void *X, int incX,
                 void *Y, int incY);
void cblas_zaxpy(int N, const void *alpha, const void *X,
                 int incX, void *Y, int incY);
void catlas_zaxpby(int N, const void *alpha, const void *X,
                   int incX, const void *beta, void *Y, int incY);
void catlas_zset
        (int N, const void *alpha, void *X, int incX);


/*
 * Routines with S and D prefix only
 */
void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void cblas_srot(int N, float *X, int incX,
                float *Y, int incY, const float c, const float s);
void cblas_srotm(int N, float *X, int incX,
                 float *Y, int incY, const float *P);

void cblas_drotg(double *a, double *b, double *c, double *s);
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void cblas_drot(int N, double *X, int incX,
                double *Y, int incY, const double c, const double s);
void cblas_drotm(int N, double *X, int incX,
                 double *Y, int incY, const double *P);


/*
 * Routines with S D C Z CS and ZD prefixes
 */
void cblas_sscal(int N, const float alpha, float *X, int incX);
void cblas_dscal(int N, const double alpha, double *X, int incX);
void cblas_cscal(int N, const void *alpha, void *X, int incX);
void cblas_zscal(int N, const void *alpha, void *X, int incX);
void cblas_csscal(int N, const float alpha, void *X, int incX);
void cblas_zdscal(int N, const double alpha, void *X, int incX);

/*
 * Extra reference routines provided by ATLAS, but not mandated by the standard
 */
void cblas_crotg(void *a, void *b, void *c, void *s);
void cblas_zrotg(void *a, void *b, void *c, void *s);
void cblas_csrot(int N, void *X, int incX, void *Y, int incY,
                 const float c, const float s);
void cblas_zdrot(int N, void *X, int incX, void *Y, int incY,
                 const double c, const double s);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 const float alpha, const float *A, int lda,
                 const float *X, int incX, const float beta,
                 float *Y, int incY);
void cblas_sgbmv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, const float alpha,
                 const float *A, int lda, const float *X,
                 int incX, const float beta, float *Y, int incY);
void cblas_strmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const float *A, int lda,
                 float *X, int incX);
void cblas_stbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const float *A, int lda,
                 float *X, int incX);
void cblas_stpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const float *Ap, float *X, int incX);
void cblas_strsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const float *A, int lda, float *X,
                 int incX);
void cblas_stbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const float *A, int lda,
                 float *X, int incX);
void cblas_stpsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const float *Ap, float *X, int incX);

void cblas_dgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 const double alpha, const double *A, int lda,
                 const double *X, int incX, const double beta,
                 double *Y, int incY);
void cblas_dgbmv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, const double alpha,
                 const double *A, int lda, const double *X,
                 int incX, const double beta, double *Y, int incY);
void cblas_dtrmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const double *A, int lda,
                 double *X, int incX);
void cblas_dtbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const double *A, int lda,
                 double *X, int incX);
void cblas_dtpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const double *Ap, double *X, int incX);
void cblas_dtrsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const double *A, int lda, double *X,
                 int incX);
void cblas_dtbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const double *A, int lda,
                 double *X, int incX);
void cblas_dtpsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const double *Ap, double *X, int incX);

void cblas_cgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *X, int incX, const void *beta,
                 void *Y, int incY);
void cblas_cgbmv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, const void *alpha,
                 const void *A, int lda, const void *X,
                 int incX, const void *beta, void *Y, int incY);
void cblas_ctrmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *A, int lda,
                 void *X, int incX);
void cblas_ctbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const void *A, int lda,
                 void *X, int incX);
void cblas_ctpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *Ap, void *X, int incX);
void cblas_ctrsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *A, int lda, void *X,
                 int incX);
void cblas_ctbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const void *A, int lda,
                 void *X, int incX);
void cblas_ctpsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *Ap, void *X, int incX);

void cblas_zgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *X, int incX, const void *beta,
                 void *Y, int incY);
void cblas_zgbmv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, int M, int N,
                 int KL, int KU, const void *alpha,
                 const void *A, int lda, const void *X,
                 int incX, const void *beta, void *Y, int incY);
void cblas_ztrmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *A, int lda,
                 void *X, int incX);
void cblas_ztbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const void *A, int lda,
                 void *X, int incX);
void cblas_ztpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *Ap, void *X, int incX);
void cblas_ztrsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *A, int lda, void *X,
                 int incX);
void cblas_ztbsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, int K, const void *A, int lda,
                 void *X, int incX);
void cblas_ztpsv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 int N, const void *Ap, void *X, int incX);


/*
 * Routines with S and D prefixes only
 */
void cblas_ssymv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const float alpha, const float *A,
                 int lda, const float *X, int incX,
                 const float beta, float *Y, int incY);
void cblas_ssbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, int K, const float alpha, const float *A,
                 int lda, const float *X, int incX,
                 const float beta, float *Y, int incY);
void cblas_sspmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const float alpha, const float *Ap,
                 const float *X, int incX,
                 const float beta, float *Y, int incY);
void cblas_sger(const enum CBLAS_ORDER Order, int M, int N,
                const float alpha, const float *X, int incX,
                const float *Y, int incY, float *A, int lda);
void cblas_ssyr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const float alpha, const float *X,
                int incX, float *A, int lda);
void cblas_sspr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const float alpha, const float *X,
                int incX, float *Ap);
void cblas_ssyr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const float alpha, const float *X,
                 int incX, const float *Y, int incY, float *A,
                 int lda);
void cblas_sspr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const float alpha, const float *X,
                 int incX, const float *Y, int incY, float *A);

void cblas_dsymv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const double alpha, const double *A,
                 int lda, const double *X, int incX,
                 const double beta, double *Y, int incY);
void cblas_dsbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, int K, const double alpha, const double *A,
                 int lda, const double *X, int incX,
                 const double beta, double *Y, int incY);
void cblas_dspmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const double alpha, const double *Ap,
                 const double *X, int incX,
                 const double beta, double *Y, int incY);
void cblas_dger(const enum CBLAS_ORDER Order, int M, int N,
                const double alpha, const double *X, int incX,
                const double *Y, int incY, double *A, int lda);
void cblas_dsyr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const double alpha, const double *X,
                int incX, double *A, int lda);
void cblas_dspr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const double alpha, const double *X,
                int incX, double *Ap);
void cblas_dsyr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const double alpha, const double *X,
                 int incX, const double *Y, int incY, double *A,
                 int lda);
void cblas_dspr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const double alpha, const double *X,
                 int incX, const double *Y, int incY, double *A);


/*
 * Routines with C and Z prefixes only
 */
void cblas_chemv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const void *alpha, const void *A,
                 int lda, const void *X, int incX,
                 const void *beta, void *Y, int incY);
void cblas_chbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, int K, const void *alpha, const void *A,
                 int lda, const void *X, int incX,
                 const void *beta, void *Y, int incY);
void cblas_chpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const void *alpha, const void *Ap,
                 const void *X, int incX,
                 const void *beta, void *Y, int incY);
void cblas_cgeru(const enum CBLAS_ORDER Order, int M, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *A, int lda);
void cblas_cgerc(const enum CBLAS_ORDER Order, int M, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *A, int lda);
void cblas_cher(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const float alpha, const void *X, int incX,
                void *A, int lda);
void cblas_chpr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const float alpha, const void *X,
                int incX, void *A);
void cblas_cher2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *A, int lda);
void cblas_chpr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *Ap);

void cblas_zhemv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const void *alpha, const void *A,
                 int lda, const void *X, int incX,
                 const void *beta, void *Y, int incY);
void cblas_zhbmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, int K, const void *alpha, const void *A,
                 int lda, const void *X, int incX,
                 const void *beta, void *Y, int incY);
void cblas_zhpmv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 int N, const void *alpha, const void *Ap,
                 const void *X, int incX,
                 const void *beta, void *Y, int incY);
void cblas_zgeru(const enum CBLAS_ORDER Order, int M, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *A, int lda);
void cblas_zgerc(const enum CBLAS_ORDER Order, int M, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *A, int lda);
void cblas_zher(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const double alpha, const void *X, int incX,
                void *A, int lda);
void cblas_zhpr(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                int N, const double alpha, const void *X,
                int incX, void *A);
void cblas_zher2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *A, int lda);
void cblas_zhpr2(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, int N,
                 const void *alpha, const void *X, int incX,
                 const void *Y, int incY, void *Ap);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, const float alpha, const float *A,
                 int lda, const float *B, int ldb,
                 const float beta, float *C, int ldc);
void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, int M, int N,
                 const float alpha, const float *A, int lda,
                 const float *B, int ldb, const float beta,
                 float *C, int ldc);
void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const float alpha, const float *A, int lda,
                 const float beta, float *C, int ldc);
void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const float alpha, const float *A, int lda,
                  const float *B, int ldb, const float beta,
                  float *C, int ldc);
void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const float alpha, const float *A, int lda,
                 float *B, int ldb);
void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const float alpha, const float *A, int lda,
                 float *B, int ldb);

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, const double alpha, const double *A,
                 int lda, const double *B, int ldb,
                 const double beta, double *C, int ldc);
void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, int M, int N,
                 const double alpha, const double *A, int lda,
                 const double *B, int ldb, const double beta,
                 double *C, int ldc);
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const double alpha, const double *A, int lda,
                 const double beta, double *C, int ldc);
void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const double alpha, const double *A, int lda,
                  const double *B, int ldb, const double beta,
                  double *C, int ldc);
void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const double alpha, const double *A, int lda,
                 double *B, int ldb);
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const double alpha, const double *A, int lda,
                 double *B, int ldb);

void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, const void *alpha, const void *A,
                 int lda, const void *B, int ldb,
                 const void *beta, void *C, int ldc);
void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *B, int ldb, const void *beta,
                 void *C, int ldc);
void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const void *alpha, const void *A, int lda,
                 const void *beta, void *C, int ldc);
void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void *alpha, const void *A, int lda,
                  const void *B, int ldb, const void *beta,
                  void *C, int ldc);
void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const void *alpha, const void *A, int lda,
                 void *B, int ldb);
void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const void *alpha, const void *A, int lda,
                 void *B, int ldb);

void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, const void *alpha, const void *A,
                 int lda, const void *B, int ldb,
                 const void *beta, void *C, int ldc);
void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *B, int ldb, const void *beta,
                 void *C, int ldc);
void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const void *alpha, const void *A, int lda,
                 const void *beta, void *C, int ldc);
void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void *alpha, const void *A, int lda,
                  const void *B, int ldb, const void *beta,
                  void *C, int ldc);
void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const void *alpha, const void *A, int lda,
                 void *B, int ldb);
void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, int M, int N,
                 const void *alpha, const void *A, int lda,
                 void *B, int ldb);


/*
 * Routines with prefixes C and Z only
 */
void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *B, int ldb, const void *beta,
                 void *C, int ldc);
void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const float alpha, const void *A, int lda,
                 const float beta, void *C, int ldc);
void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void *alpha, const void *A, int lda,
                  const void *B, int ldb, const float beta,
                  void *C, int ldc);
void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, int M, int N,
                 const void *alpha, const void *A, int lda,
                 const void *B, int ldb, const void *beta,
                 void *C, int ldc);
void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, int N, int K,
                 const double alpha, const void *A, int lda,
                 const double beta, void *C, int ldc);
void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, int N, int K,
                  const void *alpha, const void *A, int lda,
                  const void *B, int ldb, const double beta,
                  void *C, int ldc);

int cblas_errprn(int ierr, int info, char *form, ...);
#ifdef __cplusplus
}
#endif
#endif  /* end #ifdef CBLAS_ENUM_ONLY */
#endif
#endif //NATIVEOPERATIONS_CBLAS_H
