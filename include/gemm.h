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
        private:

            static inline int linearIndexC(int rows, int cols, int r, int c) {
                return (r * cols + c);
            }

            static inline int linearIndexF(int rows, int cols, int r, int c) {
                return (c * rows + r);
            }

            static T* transpose(int order, int rows, int cols, T *source) {
                T *ret = new T[rows * cols];

                // handle transpose
#pragma omp parallel for proc_bind(close)
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        int zIdx = order == CblasColMajor ? linearIndexF(rows, cols, c, r) : linearIndexC(rows, cols, c, r);
                        int xIdx = order == CblasColMajor ? linearIndexF(rows, cols, r, c) : linearIndexC(rows, cols, r, c);

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
                T *aT = TransA == CblasTrans ? transpose(Order, M, N, A) : A;
                T *bT = TransA == CblasTrans ? transpose(Order, N, K, B) : B;


#pragma omp parallel for proc_bind(close)
                for (int r = 0; r < N; r++) {

                    for (int c = 0; c < K; c++) {
                        int zIdx = order == CblasColMajor ? linearIndexF(rows, cols, c, r) : linearIndexC(rows, cols, c, r);

                        int rAi = order == CblasColMajor ? linearIndexF(M, N, r, 0) : linearIndexC(M, N, r, 0);
                        int rBi = order == CblasColMajor ? linearIndexF(N, K, r, 0) : linearIndexC(N, K, r, 0);
                        T *rA = &A[rAi];
                        T *rB = &B[rBi];

                        C[zIdx] = nd4j::math::nd4j_dot(rA, rB, M);
                    }
                }


                // if transpose was applied - dismiss transposed arrays
                if (TransA == CblasTrans)
                    delete[] aT;

                if (TransB == CblasTrans)
                    delete[] bT;
            }
        };
    }
}

#endif //LIBND4J_GEMM_H
