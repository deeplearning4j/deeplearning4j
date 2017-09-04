//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GEMM_H
#define LIBND4J_GEMM_H

#include <cblas.h>
#include <templatemath.h>


namespace nd4j {
     namespace blas {




        template <typename T>
        class GEMM {
        protected:

            static inline int linearIndexC(int rows, int cols, int r, int c) {
                return (r * cols + c);
            }

            static inline int linearIndexF(int rows, int cols, int r, int c) {
                return (c * rows + r);
            }

            static T* transpose(int orderSource, int orderTarget, int rows, int cols, T *source) {
                T *ret = new T[rows * cols];

                // handle transpose in parallel
#pragma omp parallel for proc_bind(close)
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        int zIdx = orderTarget == CblasRowMajor ? linearIndexC(rows, cols, r, c) : linearIndexF(rows, cols, r, c);
                        int xIdx = orderSource == CblasColMajor ? linearIndexF(rows, cols, r, c) : linearIndexC(rows, cols, r, c);

                        ret[zIdx] = source[xIdx];
                    }
                }

                return ret;
            }

        public:

            static void op(int Order, int TransA, int TransB,
                      int M, int N, int K,
                      T alpha,
                      T *A, int lda,
                      T *B, int ldb,
                      T beta,
                      T *C, int ldc) {

                // optionally handle transpose
                // we want C always here
                T *aT = TransA != CblasTrans ? transpose(CblasColMajor, CblasRowMajor, M, K, A) : A;

                // we want F always here
                T *bT = TransB == CblasTrans ? transpose(CblasRowMajor, CblasColMajor, K, N, B) : B;


                if (beta == (T) 0.0f) {
                    int length = M*N;
                    if (length <= 8192) {
#pragma omp simd
                        for (int r = 0; r < length; r++)
                            C[r] = (T) 0.0f;
                    } else {
#pragma omp parallel for simd
                        for (int r = 0; r < length; r++)
                            C[r] = (T) 0.0f;
                    }
                }


#pragma omp parallel for proc_bind(spread)
                for (int r = 0; r < M; r++) {

                    int aIdx = linearIndexC(M, K, r, 0);
                    T *aX = aT + aIdx;

                    for (int c = 0; c < N; c++) {
                        int zIdx = linearIndexF(M, N, r, c);

                        T dot = (T) 0.0f;

                        if (alpha != (T) 0.0f) {
                            int bIdx = linearIndexF(K, N, 0, c);

                            T *bX = bT + bIdx;

                            dot = nd4j::math::nd4j_dot<T>(aX, bX, K) * alpha;
                        }

                        if (beta != (T) 0.0f) {
                            C[zIdx] = dot + beta * C[zIdx];
                        } else {
                            C[zIdx] = dot;
                        }
                    }
                }


                // if transpose was applied - dismiss transposed arrays
                if (TransA != CblasTrans)
                    delete[] aT;

                if (TransB == CblasTrans)
                    delete[] bT;
            }
        };

         template <typename T>
         class GEMV : public nd4j::blas::GEMM<T>{

         public:
             static void op(int TRANS, int M, int N,
                            T alpha,
                            T* A,
                            int lda,
                            T* X,
                            int incx,
                            T beta,
                            T* Y,
                            int incy ) {

                 T *aT = TRANS == CblasTrans ? GEMM<T>::transpose(CblasColMajor, CblasRowMajor, M, N, A) : A;

#pragma omp parallel for proc_bind(close)
                 for (int r = 0; r < M; r++) {
                     int aIdx = GEMM<T>::linearIndexC(M, N, r, 0);
                     T *aX = aT + aIdx;


                     T dot = nd4j::math::nd4j_dot<T>(aX, X, lda) * alpha;
                     Y[r] =  beta == (T) 0.0f ? dot : dot + beta * Y[r];
                 }

                 if (TRANS == CblasTrans)
                     delete[] aT;
             }
         };
    }
}

#endif //LIBND4J_GEMM_H
