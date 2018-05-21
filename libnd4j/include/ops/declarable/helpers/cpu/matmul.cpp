//
// Created by raver119 on 20.12.17.
//

#include <ops/declarable/helpers/matmul.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _matmul(NDArray<T> *vA, NDArray<T> *vB, NDArray<T> *vC, int transA, int transB, T alpha, T beta) {
                CBLAS_TRANSPOSE tA = (CBLAS_TRANSPOSE) transA;
                CBLAS_TRANSPOSE tB = (CBLAS_TRANSPOSE) transB;

                int M = vA->sizeAt(0);
                int N = vB->sizeAt(1);
                int K = vA->sizeAt(1);

                int ldA = transA == CblasNoTrans ? M : K;
                int ldB = transB == CblasNoTrans ? K : N;
                int ldC = M;

                auto A = vA->buffer();
                auto B = vB->buffer();
                auto C = vC->buffer();

#pragma omp parallel for simd collapse(2)
                for (int m = 0; m < M; ++m) {
                    for (int n = 0; n < N; ++n) {
                        T c_mnp = 0;

                        for (int k = 0; k < K; ++k)
                            c_mnp += A[tA == CblasNoTrans ? (m + k * ldA) : (m * ldA + k)] * B[tB == CblasNoTrans ? (k + n * ldB) : (k * ldB + n)];

                        C[m + n * ldC] = alpha * c_mnp + beta * C[m + n * ldC];
                    }
                }
            };

            template void _matmul<float>(NDArray<float> *A, NDArray<float> *B, NDArray<float> *C, int transA, int transB, float alpha, float beta);
            template void _matmul<float16>(NDArray<float16> *A, NDArray<float16> *B, NDArray<float16> *C, int transA, int transB, float16 alpha, float16 beta);
            template void _matmul<double>(NDArray<double> *A, NDArray<double> *B, NDArray<double> *C, int transA, int transB, double alpha, double beta);
        }
    }
}
