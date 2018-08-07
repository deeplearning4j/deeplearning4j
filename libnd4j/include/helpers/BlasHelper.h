/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_BLAS_HELPER_H
#define LIBND4J_BLAS_HELPER_H

#include <pointercast.h>
#include <types/float16.h>
#include <cblas.h>
#include <helpers/logger.h>

#ifdef _WIN32
#define CUBLASWINAPI __stdcall
#define CUSOLVERAPI __stdcall
#else
#define CUBLASWINAPI 
#define CUSOLVERAPI 
#endif

namespace nd4j {
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
        CUBLAS_OP_N=0,  
        CUBLAS_OP_T=1,  
        CUBLAS_OP_C=2  
    } cublasOperation_t;

    struct cublasContext;
    typedef struct cublasContext *cublasHandle_t;

    typedef enum
    {
	    CUDA_R_16F= 2,  /* real as a half */
	    CUDA_C_16F= 6,  /* complex as a pair of half numbers */
	    CUDA_R_32F= 0,  /* real as a float */
	    CUDA_C_32F= 4,  /* complex as a pair of float numbers */
	    CUDA_R_64F= 1,  /* real as a double */
	    CUDA_C_64F= 5,  /* complex as a pair of double numbers */
	    CUDA_R_8I = 3,  /* real as a signed char */
	    CUDA_C_8I = 7,  /* complex as a pair of signed char numbers */
	    CUDA_R_8U = 8,  /* real as a unsigned char */
	    CUDA_C_8U = 9,  /* complex as a pair of unsigned char numbers */
	    CUDA_R_32I= 10, /* real as a signed int */
	    CUDA_C_32I= 11, /* complex as a pair of signed int numbers */
	    CUDA_R_32U= 12, /* real as a unsigned int */
	    CUDA_C_32U= 13  /* complex as a pair of unsigned int numbers */
    } cublasDataType_t; 

    typedef void (*CblasSgemv)(CBLAS_ORDER Layout,
                 CBLAS_TRANSPOSE TransA, int M, int N,
                 float alpha, float *A, int lda,
                 float *X, int incX, float beta,
                 float *Y, int incY);

    typedef void (*CblasDgemv)(CBLAS_ORDER Layout,
                 CBLAS_TRANSPOSE TransA, int M, int N,
                 double alpha, double *A, int lda,
                 double *X, int incX, double beta,
                 double *Y, int incY);


    typedef void (*CblasSgemm)(CBLAS_ORDER Layout, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB, int M, int N,
                     int K, float alpha, float *A,
                     int lda, float *B, int ldb,
                     float beta, float *C, int ldc);

    typedef void (*CblasDgemm)(CBLAS_ORDER Layout, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB, int M, int N,
                     int K, double alpha, double *A,
                     int lda, double *B, int ldb,
                     double beta, double *C, int ldc);

    typedef void (*CblasSgemmBatch)(CBLAS_ORDER Layout, CBLAS_TRANSPOSE *TransA_Array,
                           CBLAS_TRANSPOSE *TransB_Array, int *M_Array, int *N_Array,
                           int *K_Array, float *alpha_Array, float **A_Array,
                           int *lda_Array, float **B_Array, int *ldb_Array,
                           float *beta_Array, float **C_Array, int *ldc_Array,
                           int group_count, int *group_size);

    typedef void (*CblasDgemmBatch)(CBLAS_ORDER Layout, CBLAS_TRANSPOSE *TransA_Array,
                           CBLAS_TRANSPOSE *TransB_Array, int *M_Array, int *N_Array,
                           int *K_Array, double *alpha_Array, double **A_Array,
                           int *lda_Array, double **B_Array, int* ldb_Array,
                           double *beta_Array, double **C_Array, int *ldc_Array,
                           int group_count, int *group_size);

    enum LAPACK_LAYOUT { LAPACK_ROW_MAJOR=101, LAPACK_COL_MAJOR=102 };

    typedef int (*LapackeSgesvd)(LAPACK_LAYOUT matrix_layout, char jobu, char jobvt,
                           int m, int n, float* a, int lda,
                           float* s, float* u, int ldu, float* vt,
                           int ldvt, float* superb);

    typedef int (*LapackeDgesvd)(LAPACK_LAYOUT matrix_layout, char jobu, char jobvt,
                           int m, int n, double* a,
                           int lda, double* s, double* u, int ldu,
                           double* vt, int ldvt, double* superb);

    typedef int (*LapackeSgesdd)(LAPACK_LAYOUT matrix_layout, char jobz, int m,
                           int n, float* a, int lda, float* s,
                           float* u, int ldu, float* vt,
                           int ldvt);
    typedef int (*LapackeDgesdd)(LAPACK_LAYOUT matrix_layout, char jobz, int m,
                           int n, double* a, int lda, double* s,
                           double* u, int ldu, double* vt,
                           int ldvt);

    typedef cublasStatus_t (CUBLASWINAPI *CublasSgemv)(cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m, 
                                                      int n, 
                                                      float *alpha, /* host or device pointer */
                                                      float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx, 
                                                      float *beta,  /* host or device pointer */
                                                      float *y, 
                                                      int incy);  
 
    typedef cublasStatus_t (CUBLASWINAPI *CublasDgemv)(cublasHandle_t handle, 
                                                      cublasOperation_t trans, 
                                                      int m,
                                                      int n,
                                                      double *alpha, /* host or device pointer */ 
                                                      double *A,
                                                      int lda,
                                                      double *x,
                                                      int incx,
                                                      double *beta, /* host or device pointer */
                                                      double *y, 
                                                      int incy);

    typedef cublasStatus_t (CUBLASWINAPI *CublasHgemm)(cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      __half *alpha, /* host or device pointer */  
                                                      __half *A, 
                                                      int lda,
                                                      __half *B,
                                                      int ldb, 
                                                      __half *beta, /* host or device pointer */  
                                                      __half *C,
                                                      int ldc);             

    typedef cublasStatus_t (CUBLASWINAPI *CublasSgemm)(cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      float *alpha, /* host or device pointer */  
                                                      float *A, 
                                                      int lda,
                                                      float *B,
                                                      int ldb, 
                                                      float *beta, /* host or device pointer */  
                                                      float *C,
                                                      int ldc);
    
    typedef cublasStatus_t (CUBLASWINAPI *CublasDgemm)(cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      double *alpha, /* host or device pointer */  
                                                      double *A, 
                                                      int lda,
                                                      double *B,
                                                      int ldb, 
                                                      double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc);

    typedef cublasStatus_t (CUBLASWINAPI *CublasSgemmEx)(cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      float *alpha, /* host or device pointer */  
                                                      void *A, 
                                                      cublasDataType_t Atype,
                                                      int lda,
                                                      void *B,
                                                      cublasDataType_t Btype,
                                                      int ldb, 
                                                      float *beta, /* host or device pointer */  
                                                      void *C,
                                                      cublasDataType_t Ctype,
                                                      int ldc); 

    typedef cublasStatus_t (CUBLASWINAPI *CublasHgemmBatched)(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          __half *alpha,  /* host or device pointer */  
                                                          __half *Aarray[], 
                                                          int lda,
                                                          __half *Barray[],
                                                          int ldb, 
                                                          __half *beta,   /* host or device pointer */  
                                                          __half *Carray[],
                                                          int ldc,
                                                          int batchCount);

    typedef cublasStatus_t (CUBLASWINAPI *CublasSgemmBatched)(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          float *alpha,  /* host or device pointer */  
                                                          float *Aarray[], 
                                                          int lda,
                                                          float *Barray[],
                                                          int ldb, 
                                                          float *beta,   /* host or device pointer */  
                                                          float *Carray[],
                                                          int ldc,
                                                          int batchCount);

    typedef cublasStatus_t (CUBLASWINAPI *CublasDgemmBatched)(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb, 
                                                          int m,
                                                          int n,
                                                          int k,
                                                          double *alpha,  /* host or device pointer */ 
                                                          double *Aarray[], 
                                                          int lda,
                                                          double *Barray[],
                                                          int ldb, 
                                                          double *beta,  /* host or device pointer */ 
                                                          double *Carray[],
                                                          int ldc,
                                                          int batchCount);

    typedef enum{
        CUSOLVER_STATUS_SUCCESS=0,
        CUSOLVER_STATUS_NOT_INITIALIZED=1,
        CUSOLVER_STATUS_ALLOC_FAILED=2,
        CUSOLVER_STATUS_INVALID_VALUE=3,
        CUSOLVER_STATUS_ARCH_MISMATCH=4,
        CUSOLVER_STATUS_MAPPING_ERROR=5,
        CUSOLVER_STATUS_EXECUTION_FAILED=6,
        CUSOLVER_STATUS_INTERNAL_ERROR=7,
        CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
        CUSOLVER_STATUS_NOT_SUPPORTED = 9,
        CUSOLVER_STATUS_ZERO_PIVOT=10,
        CUSOLVER_STATUS_INVALID_LICENSE=11
    } cusolverStatus_t;

    typedef enum {
        CUSOLVER_EIG_TYPE_1=1,
        CUSOLVER_EIG_TYPE_2=2,
        CUSOLVER_EIG_TYPE_3=3
    } cusolverEigType_t ;

    typedef enum {
        CUSOLVER_EIG_MODE_NOVECTOR=0,
        CUSOLVER_EIG_MODE_VECTOR=1
    } cusolverEigMode_t ;

    struct cusolverDnContext;
    typedef struct cusolverDnContext *cusolverDnHandle_t;

    typedef cusolverStatus_t (CUSOLVERAPI *CusolverDnSgesvdBufferSize)(
        cusolverDnHandle_t handle,
        int m,
        int n,
        int *lwork);

    typedef cusolverStatus_t (CUSOLVERAPI *CusolverDnDgesvdBufferSize)(
        cusolverDnHandle_t handle,
        int m,
        int n,
        int *lwork);

    typedef cusolverStatus_t (CUSOLVERAPI *CusolverDnSgesvd)(
        cusolverDnHandle_t handle, 
        signed char jobu, 
        signed char jobvt, 
        int m, 
        int n, 
        float *A, 
        int lda, 
        float *S, 
        float *U, 
        int ldu, 
        float *VT, 
        int ldvt, 
        float *work, 
        int lwork, 
        float *rwork, 
        int  *info);

    typedef cusolverStatus_t (CUSOLVERAPI *CusolverDnDgesvd)(
        cusolverDnHandle_t handle, 
        signed char jobu, 
        signed char jobvt, 
        int m, 
        int n, 
        double *A, 
        int lda, 
        double *S, 
        double *U, 
        int ldu, 
        double *VT, 
        int ldvt, 
        double *work,
        int lwork, 
        double *rwork, 
        int *info);


    enum BlasFunctions {
        GEMV = 0,
        GEMM = 1,
    };

    class BlasHelper {
    private:
        static BlasHelper* _instance;

		bool _hasHgemv = false;
		bool _hasHgemm = false;
		bool _hasHgemmBatch = false;

        bool _hasSgemv = false;
        bool _hasSgemm = false;
        bool _hasSgemmBatch = false;

        bool _hasDgemv = false;
        bool _hasDgemm = false;
        bool _hasDgemmBatch = false;
        
        CblasSgemv cblasSgemv;
        CblasDgemv cblasDgemv;
        CblasSgemm cblasSgemm;
        CblasDgemm cblasDgemm;
        CblasSgemmBatch cblasSgemmBatch;
        CblasDgemmBatch cblasDgemmBatch;
        LapackeSgesvd lapackeSgesvd;
        LapackeDgesvd lapackeDgesvd;
        LapackeSgesdd lapackeSgesdd;
        LapackeDgesdd lapackeDgesdd;

        CublasSgemv cublasSgemv;
        CublasDgemv cublasDgemv;
        CublasHgemm cublasHgemm;
        CublasSgemm cublasSgemm;
        CublasDgemm cublasDgemm;
        CublasSgemmEx cublasSgemmEx;
        CublasHgemmBatched cublasHgemmBatched;
        CublasSgemmBatched cublasSgemmBatched;
        CublasDgemmBatched cublasDgemmBatched;
        CusolverDnSgesvdBufferSize cusolverDnSgesvdBufferSize;
        CusolverDnDgesvdBufferSize cusolverDnDgesvdBufferSize;
        CusolverDnSgesvd cusolverDnSgesvd;
        CusolverDnDgesvd cusolverDnDgesvd;

    public:
        static BlasHelper* getInstance();

        void initializeFunctions(Nd4jPointer *functions);
		void initializeDeviceFunctions(Nd4jPointer *functions);

        template <typename T>
        bool hasGEMV();

        template <typename T>
        bool hasGEMM();

        template <typename T>
        bool hasBatchedGEMM();

        CblasSgemv sgemv();
        CblasDgemv dgemv();

        CblasSgemm sgemm();
        CblasDgemm dgemm();

        CblasSgemmBatch sgemmBatched();
        CblasDgemmBatch dgemmBatched();

        LapackeSgesvd sgesvd();
        LapackeDgesvd dgesvd();

        LapackeSgesdd sgesdd();
        LapackeDgesdd dgesdd();
        
        // destructor
        ~BlasHelper() noexcept; 
    };
}

#endif
