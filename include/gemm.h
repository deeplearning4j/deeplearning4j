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
                        int zIdx = order == CblasColMajor ? linearIndexC(rows, cols, r, c) : linearIndexF(rows, cols, r, c);
                        int xIdx = order == CblasColMajor ? linearIndexF(rows, cols, r, c) : linearIndexC(rows, cols, r, c);

                        //printf("zIdx: [%i]; xIdx: [%i]; x.y: [%i.%i]\n", zIdx, xIdx, r, c);
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
                int aRows = M;
                int aColumns = K;
                int bRows = K;
                int bColumns = N;

                // optionally handle transpose
                T *aT = A; //TransA != CblasTrans ? transpose(Order, M, K, A) : A;
                T *bT = B; //TransB != CblasTrans ? transpose(Order, K, N, B) : B;

                if (beta == (T) 0.0f) {
#pragma omp simd
                    for (int r = 0; r < M * N; r++) {
                        C[r] = (T) 0.0f;
                    }
                }


#pragma omp parallel for proc_bind(close)
                for (int r = 0; r < M; r++) {

                    for (int c = 0; c < N; c++) {
                        int zIdx = linearIndexF(M, N, r, c);

                        T dot = (T) 0.0f;

                        if (alpha != (T) 0.0f) {
                            for (int k = 0; k < K; k++) {
                                int aIdx = TransA == CblasTrans ? linearIndexC(M, K, r, k) : linearIndexF(M, K, r, k);
                                int bIdx = TransB == CblasTrans ? linearIndexC(K, N, k, c) : linearIndexF(K, N, k, c);

                                //alpha * aT[aIdx] * bT[bIdx] + beta * C[zIdx];
                                dot += aT[aIdx] * bT[bIdx];
                            }

                            dot *= alpha;
                        }

                        if (beta != (T) 0.0f) {
                            C[zIdx] = dot + beta * C[zIdx];
                        } else {
                            C[zIdx] = dot;
                        }
                    }
                }


                // if transpose was applied - dismiss transposed arrays
                //if (TransA == CblasTrans)
                //    delete[] aT;

                //if (TransB == CblasTrans)
                //    delete[] bT;
            }
        };
    }
}

#endif //LIBND4J_GEMM_H
