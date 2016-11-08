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

            static T* transpose(int rows, int cols, T *source) {
                T *ret = new T[rows * cols];

                // handle transpose

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
                T *aT = TransA == CblasTrans ? transpose(M, N, A) : A;
                T *bT = TransA == CblasTrans ? transpose(N, K, B) : B;


#pragma omp parallel for proc_bind(close)
                for (int r = 0; r < M; r++) {

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
