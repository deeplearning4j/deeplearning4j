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
            static inline int linearIndexC(int rows, int cols, int r, int c);
            static inline int linearIndexF(int rows, int cols, int r, int c);
            static T* transpose(int orderSource, int orderTarget, int rows, int cols, T *source);


        public:
            static void op(int Order, int TransA, int TransB, int M, int N, int K, T alpha, T *A, int lda, T *B, int ldb, T beta, T *C, int ldc);
        };

         template <typename T>
         class GEMV : public nd4j::blas::GEMM<T>{

         public:
             static void op(int TRANS, int M, int N, T alpha, T* A, int lda, T* X, int incx, T beta, T* Y, int incy );
         };
    }
}

#endif //LIBND4J_GEMM_H
