//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <types/float16.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <helpers/BlasHelper.h>


namespace nd4j {
    namespace ops {
        namespace helpers {
        


            template <typename T>
            void _bgemm(std::vector<NDArray<T>*>& vA, std::vector<NDArray<T>*>& vB, std::vector<NDArray<T>*>& vC, NDArray<T>* alphas, NDArray<T>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC) {
                int batchSize = vA.size();
                if (BlasHelper::getInstance()->hasBatchedGEMM<T>()) {
                    auto arr = vA.at(0);
                    CBLAS_TRANSPOSE *_tA, *_tB;
                    int *_M, *_N, *_K, *_ldA, *_ldB, *_ldC, *_size;
                    // mkl requires mnk etc as arrays, cuda doesn't
                    ALLOCATE(_tA, arr->getWorkspace(), batchSize, CBLAS_TRANSPOSE);
                    ALLOCATE(_tB, arr->getWorkspace(), batchSize, CBLAS_TRANSPOSE);
                    ALLOCATE(_M, arr->getWorkspace(), batchSize, int);
                    ALLOCATE(_N, arr->getWorkspace(), batchSize, int);
                    ALLOCATE(_K, arr->getWorkspace(), batchSize, int);
                    ALLOCATE(_ldA, arr->getWorkspace(), batchSize, int);
                    ALLOCATE(_ldB, arr->getWorkspace(), batchSize, int);
                    ALLOCATE(_ldC, arr->getWorkspace(), batchSize, int);
                    ALLOCATE(_size, arr->getWorkspace(), batchSize, int);

                    shape::fill(_tA, (CBLAS_TRANSPOSE) transA, batchSize);
                    shape::fill(_tB, (CBLAS_TRANSPOSE) transB, batchSize);

                    shape::fill(_M, M, batchSize);
                    shape::fill(_N, N, batchSize);
                    shape::fill(_K, K, batchSize);
                    shape::fill(_ldA, ldA, batchSize);
                    shape::fill(_ldB, ldB, batchSize);
                    shape::fill(_ldC, ldC, batchSize);
                    shape::fill(_size, 1, batchSize);

                    std::vector<T*> buffersA(batchSize);
                    std::vector<T*> buffersB(batchSize);
                    std::vector<T*> buffersC(batchSize);

                    for (int e = 0; e < batchSize; e++) {
                        buffersA[e] = vA[e]->buffer();
                        buffersB[e] = vB[e]->buffer();
                        buffersC[e] = vC[e]->buffer();
                    }

                    if (sizeof(T) == 8) {
                        BlasHelper::getInstance()->dgemmBatched()(CblasColMajor, _tA, _tB, _M, _N, _K, (double *) alphas->buffer(), (double **) buffersA.data(), _ldA, (double **) buffersB.data(), _ldB, (double *) betas->buffer(),(double **)  buffersC.data(), _ldC, vA.size(), _size);
                    } else if (sizeof(T) == 4) {
                        BlasHelper::getInstance()->sgemmBatched()(CblasColMajor, _tA, _tB, _M, _N, _K, (float *) alphas->buffer(), (float **) buffersA.data(), _ldA, (float **) buffersB.data(), _ldB, (float *) betas->buffer(), (float **) buffersC.data(), _ldC, vA.size(), _size);
                    }

                    // release temporary arrays
                    RELEASE(_tA, arr->getWorkspace());
                    RELEASE(_tB, arr->getWorkspace());
                    RELEASE(_M, arr->getWorkspace());
                    RELEASE(_N, arr->getWorkspace());
                    RELEASE(_K, arr->getWorkspace());
                    RELEASE(_ldA, arr->getWorkspace());
                    RELEASE(_ldB, arr->getWorkspace());
                    RELEASE(_ldC, arr->getWorkspace());
                    RELEASE(_size, arr->getWorkspace());
                } else {
                    CBLAS_TRANSPOSE tA = (CBLAS_TRANSPOSE) transA;
                    CBLAS_TRANSPOSE tB = (CBLAS_TRANSPOSE) transB;

//#pragma omp parallel for                   
                    for (int p = 0; p < vA.size(); ++p) {
                        auto A = vA.at(p)->buffer();
                        auto B = vB.at(p)->buffer();
                        auto C = vC.at(p)->buffer();
                        auto alpha = alphas->getScalar(p);
                        auto beta = betas->getScalar(p);
                        for (int m = 0; m < M; ++m) {
                            for (int n = 0; n < N; ++n) {
                                T c_mnp = 0;

//                                #pragma omp simd
                                for (int k = 0; k < K; ++k)
                                    c_mnp += A[tA == CblasNoTrans ? (m + k * ldA) : (m * ldA + k)] * B[tB == CblasNoTrans ? (k + n * ldB) : (k * ldB + n)];

                                C[m + n * ldC] = alpha * c_mnp + beta * C[m + n * ldC];
                            } 
                        } 
                    }
                }
            };

            template void _bgemm<float>(std::vector<NDArray<float>*>& vA, std::vector<NDArray<float>*>& vB, std::vector<NDArray<float>*>& vC, NDArray<float>* alphas, NDArray<float>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC);
            template void _bgemm<double>(std::vector<NDArray<double>*>& vA, std::vector<NDArray<double>*>& vB, std::vector<NDArray<double>*>& vC, NDArray<double>* alphas, NDArray<double>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC);
            template void _bgemm<float16>(std::vector<NDArray<float16>*>& vA, std::vector<NDArray<float16>*>& vB, std::vector<NDArray<float16>*>& vC, NDArray<float16>* alphas, NDArray<float16>* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC);
        }
    }
}