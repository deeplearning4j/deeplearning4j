//
// @author raver119@gmail.com
//
#ifndef LIBND4J_AGGREGATE_OPS_H
#define LIBND4J_AGGREGATE_OPS_H

#include <ops/ops.h>
#include <templatemath.h>

#define HS_MAX_EXP 6.0f

#ifdef __CUDACC__
#define aggregate_def __device__ inline static
#else
#include <ops/gemm.h>
#define aggregate_def inline static
#endif
/*
 *
 *
 * Aggregate Ops are special things suited for CUDA mostly. They are meant to be executed within single block ONLY.
 * So, when batched, they should provide proper parallelism levels on poorly parallel tasks otherwise.
 *
 * On CPU aggregate ops are trying to minimize OpenMP multi-threading use, only SIMD is enforced
 *
 *
 */
namespace aggregateOps {

    template<typename T>
    class GEMM {
    public:
#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            // no-op
        }
#endif

#ifndef __CUDACC__
        static CBLAS_ORDER  convertOrder(int from) {
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


        static CBLAS_TRANSPOSE convertTranspose(int from) {
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
#endif

#ifndef __CUDACC__
        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            int M = indexArguments[0];
            int N = indexArguments[1];
            int K = indexArguments[2];
            int lda = indexArguments[3];
            int ldb = indexArguments[4];
            int ldc = indexArguments[5];
            int TransA = indexArguments[6];
            int TransB = indexArguments[7];
            int Order = indexArguments[8];

            T alpha = realArguments[0];
            T beta = realArguments[1];

            T *A = arguments[0];
            T *B = arguments[1];
            T *C = arguments[2];

            nd4j::blas::GEMM<T>::op(convertOrder(Order), convertTranspose(TransA), convertTranspose(TransB),M,N,K,(T) alpha,A,lda,B,ldb,(T) beta,C,ldc);
        }
#else
        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            // stub for nvcc
        }
#endif
    };

    /**
     * We don't include this class into ops directly, since it won't be ever used directly,
     * Only as part of SkipGram or CBOW
     */
    template<typename T>
    class HierarchicSoftmax {
        private:

        public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            int vectorLength = indexArguments[0];
            int expLength = indexArguments[1];
            int code = indexArguments[2];
            int isInference = indexArguments[3];

            T *syn0 = arguments[0]; // we pass row pointer here
            T *syn1 = arguments[1]; // we pass row pointer here
            T *expTable = arguments[2];
            T *neu1e = arguments[3];


            T dot(0.0f);
            T g(0.0f);
            T f(0.0f);
            T alpha = realArguments[0];

            //nd4j_printf("Vector length: [%i]; expLength: [%i]; Code: [%i]; Inf: [%i]\n", vectorLength, expLength, code, isInference);


//            shape::printArray<T>(syn0, vectorLength, "syn0");
//            shape::printArray<T>(syn1, vectorLength, "syn1");
//            shape::printArray<T>(neu1e, vectorLength, "neu1e");

            // dot
//#pragma omp simd reduction(sumT:dot)
            for (int x = 0; x < vectorLength; x++) {
                dot += syn0[x] * syn1[x];
            }

            // gradient
            if (dot < (T) - HS_MAX_EXP || dot >= (T) HS_MAX_EXP) {
                return;
            }

            int idx = static_cast<int>((dot + HS_MAX_EXP) * ((T) expLength / HS_MAX_EXP / 2.0f));

            if (idx >= expLength || idx < 0) {
                return;
            }

            f = expTable[idx];
            g = (static_cast<T>(1.0f) - static_cast<T>(code) - f) * alpha;

            //nd4j_printf("dot: [%f]; idx: [%i]; f: [%f]; g: [%f]\n", (float) dot, idx, (float) f, (float) g);

            // axpy1
#pragma omp simd
            for (int x = 0; x < vectorLength; x++) {
                neu1e[x] = g * syn1[x] + neu1e[x];
            }

            // axpy2
            if (!isInference) {
#pragma omp simd
                for (int x = 0; x < vectorLength; x++) {
                    syn1[x] = g * syn0[x] + syn1[x];
                }
            }
        }

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            /*
                We know that syn0 & syn1 are 2D matrices, so we can just use offsets here
            */
            __shared__ int vectorLength;
            __shared__ int expLength;
            __shared__ int code;
            __shared__ int isInference;

            T *syn0 = arguments[0];
            T *syn1 = arguments[1];
            T *expTable = arguments[2];
            T *neu1e = arguments[3];

            __shared__ T dot;
            __shared__ T g;
            __shared__ T f;
            __shared__ T alpha;

            if (threadIdx.x == 0) {
                vectorLength = indexArguments[0];
                expLength = indexArguments[1];
                code = indexArguments[2];
                isInference = indexArguments[3];

                dot = (T) 0.0f;

                alpha = realArguments[0];
            }
            __syncthreads();


            // TODO: it would be great to implement dot without atomicAdd call. like aggregateParticles, or something like that
            // dot
            for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                T prod = syn0[x] * syn1[x];
                nd4j::math::atomics::nd4j_atomicAdd<T>(&dot, prod);
            }


            // gradient
            __syncthreads();

            if (dot < - (T) HS_MAX_EXP || dot >= (T) HS_MAX_EXP)
                return;

            int idx = (int) ((dot + HS_MAX_EXP) * ((T) expLength / (T) HS_MAX_EXP / 2.0));

            if (idx >= expLength)
                return;


            if (threadIdx.x == 0) {
                // gradient calculation
                f = expTable[idx];
                g = ((T) 1.0f - (T) code - f) * alpha;
            }
            __syncthreads();

            // axpy1
            for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                neu1e[x] = g * syn1[x] + neu1e[x];
            }

            // axpy2
            if (!isInference)
                for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                    syn1[x] = g * syn0[x] + syn1[x];
                }
        }
#endif
    };

    /**
     * We don't include this class into ops directly, since it won't be ever used directly,
     * Only as part of SkipGram or CBOW
     */
    template<typename T>
    class NegativeSampling {
    public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            int vectorLength = indexArguments[0];
            int expLength = indexArguments[1];
            int code = indexArguments[2];
            int isInference = indexArguments[3];

            T *syn0 = arguments[0]; // we pass row pointer here
            T *syn1Neg = arguments[1]; // we pass row pointer here
            T *expTable = arguments[2];
            T *neu1e = arguments[3];

            T dot = (T) 0.0f;
            T g = (T) 0.0f;
            T alpha = realArguments[0];

            // dot
#pragma omp simd reduction(sumT:dot)
            for (int x = 0; x < vectorLength; x++) {
                dot += syn0[x] * syn1Neg[x];
            }

            if (dot > HS_MAX_EXP)
                g = (code - 1) * alpha;
            else if (dot < (T) - HS_MAX_EXP)
                g = (code - 0) * alpha;
            else {
                int idx = (int) ((dot + (T) HS_MAX_EXP) * ((T) expLength / HS_MAX_EXP / 2.0));
                if (idx >= expLength)
                    return;

                if (idx < 0)
                    return;

                g = ((T) code - expTable[idx]) * alpha;
            }

            // axpy1
#pragma omp simd
            for (int x = 0; x < vectorLength; x++) {
                neu1e[x] = g * syn1Neg[x] + neu1e[x];
            }

            // axpy2
            if (!isInference) {
#pragma omp simd
                for (int x = 0; x < vectorLength; x++) {
                    syn1Neg[x] = g * syn0[x] + syn1Neg[x];
                }
            }
        }

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            /*
                We know that syn0 & syn1 are 2D matrices, so we can just use offsets here
            */
            __shared__ int vectorLength;
            __shared__ int expLength;
            __shared__ int code;
            __shared__ int isInference;

            T *syn0 = arguments[0];
            T *syn1Neg = arguments[1];
            T *expTable = arguments[2];
            T *neu1e = arguments[3];

            __shared__ T dot;
            __shared__ T g;
            __shared__ T alpha;

            if (threadIdx.x == 0) {
                vectorLength = indexArguments[0];
                expLength = indexArguments[1];
                code = indexArguments[2];
                isInference = indexArguments[3];

                dot = (T) 0.0f;

                alpha = realArguments[0];
            }
            __syncthreads();


            // TODO: it would be great to implement dot without atomicAdd call. like aggregateParticles, or something like that
            // dot
            for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                T prod = syn0[x] * syn1Neg[x];
                nd4j::math::atomics::nd4j_atomicAdd<T>(&dot, prod);
            }


            // gradient
            __syncthreads();


            int idx = (int) ((dot + (T) HS_MAX_EXP) * ((T) expLength / (T) HS_MAX_EXP / 2.0));
            if (idx >= expLength && dot <= (T) HS_MAX_EXP && dot >= (T) -HS_MAX_EXP)
                return;


            if (threadIdx.x == 0) {
                // gradient calculation
                if (dot > (T) HS_MAX_EXP)
                    g = (code - 1) * alpha;
                else if (dot < (T) - HS_MAX_EXP)
                    g = (code - 0) * alpha;
                else {


                    g = ((T) code - expTable[idx]) * alpha;
                }

            //    printf("dot: [%f]; g: [%f]\n", dot, g);
            }
            __syncthreads();

           // printf("before syn1Neg[%i]: [%f], dot: [%f]; g: [%f]; vectorLength: [%i]\n", threadIdx.x, syn1Neg[threadIdx.x], dot, g, vectorLength);

            // axpy1
            for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                neu1e[x] = g * syn1Neg[x] + neu1e[x];
            }

            // axpy2
            if (!isInference)
                for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                    syn1Neg[x] = g * syn0[x] + syn1Neg[x];
                }

        //    printf("after syn1Neg[%i]: [%f]\n", threadIdx.x, syn1Neg[threadIdx.x]);

        }
#endif
    };

    template<typename T>
    class Dot {
    public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            T *vecX = arguments[0];
            T *vecY = arguments[1];
            T *vecZ = arguments[2];

            T dot = (T) 0.0f;

            int vectorLength = indexArguments[0];

#pragma omp simd reduction(sumT:dot)
            for (int x = 0; x < vectorLength; x++) {
                dot += vecX[x] * vecY[x];
            }

            vecZ[0] = dot;
        };

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            T *vecX = arguments[0];
            T *vecY = arguments[1];
            T *vecZ = arguments[2];

            int vectorLength = indexArguments[0];

            __shared__ T dot;
            if (threadIdx.x == 0)
                dot = (T) 0.0f;
            __syncthreads();

            for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                T prod = vecX[x] * vecY[x];
                nd4j::math::atomics::nd4j_atomicAdd<T>(&dot, prod);
            }
            __syncthreads();

            if (threadIdx.x == 0)
                vecZ[0] = dot;
        }
#endif
    };

    template<typename T>
    class Axpy {
    public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            T *vecX = arguments[0];
            T *vecY = arguments[1];

            T alpha = realArguments[0];

            int vectorLength = indexArguments[0];

#pragma omp simd
            for (int x = 0; x < vectorLength; x++) {
                vecY[x] = alpha * vecX[x] + vecY[x];
            }
        };

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            T *vecX = arguments[0];
            T *vecY = arguments[1];

            T alpha = realArguments[0];

            int vectorLength = indexArguments[0];

            for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                vecY[x] = alpha * vecX[x] + vecY[x];
            }
            __syncthreads();
        }
#endif
    };


    template<typename T>
    class SkipGram {
    public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            int syn0Row = indexArguments[0];
            int vectorLength = indexArguments[1];
            int hsRounds = indexArguments[2];
            int ngRounds = indexArguments[3];
            int expLength = indexArguments[4];
            int vocabSize = indexArguments[5];
            int ngStarter = indexArguments[6];
            int negTableLength = indexArguments[7];
            int isInference = indexArguments[8];


            auto neu1e = new T[vectorLength];
            std::memset(neu1e, 0, sizeof(T) * vectorLength);

            T *args[4];
            int idxArgs[4];

            args[1] = arguments[1]; // syn1
            args[2] = arguments[2]; // expTable
            args[3] = neu1e;


            idxArgs[0] = vectorLength; // vectorLength
            idxArgs[1] = expLength; // expLength
            idxArgs[3] = isInference;

            T *syn1Neg = arguments[3];
            T *negTable = arguments[4];
            T *inferenceVector = arguments[5];

            T *syn0 = isInference == 1 ? inferenceVector : arguments[0] + (syn0Row * vectorLength);

            args[0] = syn0;// syn0

            int *idxSyn1 = intArrays[0];
            int *codes = intArrays[1];

            //nd4j_printf("syn0Row: [%i]; vecLen: [%i]; hsRounds: [%i]; ngRounds: [%i]; expLength: [%i]; vocabSize: [%i]; ngStarter: [%i]; negTableLength: [%i]; isInf: [%i]\n", syn0Row, vectorLength, hsRounds, ngRounds, expLength, vocabSize, ngStarter, negTableLength, isInference);

            auto next_random = static_cast<unsigned long long>(realArguments[1]);

            if (hsRounds > 0) {
                for (int r = 0; r < hsRounds; r++) {
                    args[1] = arguments[1] + (idxSyn1[r] * vectorLength); // syn1 row
                    idxArgs[2] = codes[r];  // code for row

                    //nd4j_printf("idx syn1: [%i]; code: [%i]\n", idxSyn1[r], idxArgs[2]);

                    HierarchicSoftmax<T>::executeAggregate(args, 4, nullptr, 0, idxArgs, 5, nullptr, 0, realArguments, 1);
                }
            }



            int target = ngStarter;
            if (ngRounds > 0) {
                for (int r = 0; r < ngRounds + 1; r++) {
                    if (r == 0) {
                        idxArgs[2] = 1;
                    } else {
                        next_random = next_random * (unsigned long long) 25214903917 + 11;
                        target = negTable[(next_random >> 16) % negTableLength];

                        if (target <= 0 || target >= vocabSize) target = next_random % (vocabSize - 1) + 1;
                        if (target == ngStarter)
                            continue;

                        idxArgs[2] = 0;
                    }

                    args[1] = syn1Neg + (target * vectorLength); // syn1Neg instead of syn1

                    NegativeSampling<T>::executeAggregate(args, 4, nullptr, 0, idxArgs, 5, nullptr, 0, realArguments, 1);
                }
            }

            //nd4j_printf("applying...\n","");

            if (!isInference) {
#pragma omp simd
                for (int x = 0; x < vectorLength; x++) {
                    syn0[x] += neu1e[x];
                }
            } else {
#pragma omp simd
                for (int x = 0; x < vectorLength; x++) {
                    inferenceVector[x] += neu1e[x];
                }
            }

            delete[] neu1e;
        }

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
            __shared__ int syn0Row;
            __shared__ int vectorLength;
            __shared__ int hsRounds;
            __shared__ int ngRounds;
            __shared__ int expLength;
            __shared__ int vocabSize;
            __shared__ int ngStarter;
            __shared__ int negTableLength;
            __shared__ int isInference;

            __shared__ T *neu1e;

            __shared__ T *args[4];
            __shared__ int idxArgs[4];


            __shared__ unsigned long long next_random;

            __shared__ T *negTable;
            T *syn1Neg = arguments[3];
            __shared__ T *inferenceVector;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                neu1e = (T *) shmem;

                syn0Row = indexArguments[0];
                vectorLength = indexArguments[1];
                hsRounds = indexArguments[2];
                ngRounds = indexArguments[3];
                expLength = indexArguments[4];
                vocabSize = indexArguments[5];
                ngStarter = indexArguments[6];
                negTableLength = indexArguments[7];
                isInference = indexArguments[8];

                inferenceVector = arguments[5];

                next_random = (unsigned long long) realArguments[1];

                args[0] = isInference == 1 ? inferenceVector : arguments[0] + (syn0Row * vectorLength); // syn0
                args[1] = arguments[1]; // syn1
                args[2] = arguments[2]; // expTable
                args[3] = neu1e;

                negTable = arguments[4];

                idxArgs[0] = vectorLength; // vectorLength
                idxArgs[1] = expLength; // expLength
                idxArgs[3] = isInference;
            }
            __syncthreads();

            T *syn0 = isInference ? inferenceVector : arguments[0] + (syn0Row * vectorLength);

            for (int i = threadIdx.x; i < vectorLength; i+=blockDim.x) {
                neu1e[i] = (T) 0.0f;
            }

            int *idxSyn1 = intArrays[0];
            int *codes = intArrays[1];


            for (int r = 0; r < hsRounds; r++) {
                if (threadIdx.x == 0) {
                    args[1] = arguments[1] + (idxSyn1[r] * vectorLength);// syn1 row
                    idxArgs[2] = codes[r];  // code for row
                }
                __syncthreads();

                HierarchicSoftmax<T>::executeAggregateCuda(args, 4, nullptr, 0, idxArgs, 3, nullptr, 0,  realArguments, 1);
            }
            __syncthreads();


            __shared__ int target;
            if (ngRounds > 0)
                for (int r = 0; r < ngRounds + 1; r++) {
                    if (threadIdx.x == 0) {
                        if (r == 0) {
                            // this line isn't a mistake
                            target = ngStarter;

                            idxArgs[2] = 1;
                        } else {
                            next_random = next_random * (unsigned long long)25214903917 + 11 + blockIdx.x;
                            target = negTable[(next_random >> 16) % negTableLength];

                            if (target <= 0 || target >= vocabSize) target = next_random % (vocabSize - 1) + 1;

                            idxArgs[2] = 0;
                        }

                        args[1] = syn1Neg + (target * vectorLength);
                    }
                    __syncthreads();

                    // we put it here, to make sure all threads pick up continue call
                    if (r != 0 && target == ngStarter)
                        continue;

                    NegativeSampling<T>::executeAggregateCuda(args, 4, nullptr, 0, idxArgs, 3, nullptr, 0, realArguments, 1);
                }



            // final axpy with 1.0f as alpha
            if (!isInference)
                for (int x = threadIdx.x; x < vectorLength; x+= blockDim.x) {
                    syn0[x] += neu1e[x];
                }
            else
                for (int x = threadIdx.x; x < vectorLength; x+= blockDim.x) {
                    inferenceVector[x] += neu1e[x];
                }
        }
#endif
    };

    template<typename T>
    class CBOW {
    public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments,
                         int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,
                         T *realArguments, int numRealArguments) {
            int vectorLength = indexArguments[0];
            int hsRounds = indexArguments[1];
            int ngRounds = indexArguments[2];
            int expLength = indexArguments[3];
            int vocabSize = indexArguments[4];
            int ngStarter = indexArguments[5];
            int negTableLength = indexArguments[6];
            int idxSyn0Length = indexArguments[7];
            //int initialIdx = indexArguments[8];
            int numLabels = indexArguments[9];
            int trainWords = indexArguments[10];
            int isInference = indexArguments[11];


            int *idxSyn0 = intArrays[0];
            int *idxSyn1 = intArrays[1];
            int *codes = intArrays[2];


            T *neu1 = new T[vectorLength];
            T *neu1e = new T[vectorLength];
            std::memset(neu1, 0, sizeof(T) * vectorLength);
            std::memset(neu1e, 0, sizeof(T) * vectorLength);

            T *syn0 = arguments[0];
            T *syn1 = arguments[1];
            T *expTable = arguments[2];
            T *syn1Neg = arguments[3];
            T *negTable = arguments[4];
            T *inferenceVector = arguments[5];

            T *args[4];
            int idxArgs[4];
            idxArgs[0] = vectorLength; // vectorLength
            idxArgs[1] = expLength; // expLength
            idxArgs[3] = isInference;

            unsigned long long next_random = (unsigned long long) realArguments[1];

            // building neu1 for current window
            for (int c = 0; c < idxSyn0Length; c++) {
                T *syn0word = syn0 + (idxSyn0[c] * vectorLength);

#pragma omp simd
                for (int i = 0; i < vectorLength; i++) {
                    neu1[i] += syn0word[i];
                }
            }

            // for inference we use additional inference vector
            if (isInference) {
#pragma omp simd
                for (int i = 0; i < vectorLength; i++) {
                    neu1[i] += inferenceVector[i];
                }
            }


            // average neu1
            if (idxSyn0Length > 0) {
#pragma omp simd
                for (int i = 0; i < vectorLength; i++) {
                    neu1[i] /= idxSyn0Length + isInference;
                }
            }

            args[0] = neu1;
            args[2] = expTable;
            args[3] = neu1e;

            if (hsRounds > 0)
                for (int i = 0; i < hsRounds; i++) {
                    args[1] = syn1 + (idxSyn1[i] * vectorLength);
                    idxArgs[2] = codes[i];

                    HierarchicSoftmax<T>::executeAggregate((T **)args, 4, nullptr, 0, idxArgs, 3, nullptr, 0, realArguments, 2);
                }

            int target = ngStarter;
            if (ngRounds > 0)
                for (int i = 0; i < ngRounds + 1; i++) {
                    if (i == 0) {
                        idxArgs[2] = 1;
                    } else {
                        next_random = next_random * (unsigned long long) 25214903917 + 11;
                        target = negTable[(next_random >> 16) % negTableLength];

                        if (target <= 0 || target >= vocabSize) target = next_random % (vocabSize - 1) + 1;
                        if (target == ngStarter)
                            continue;

                        idxArgs[2] = 0;
                    }

                    args[1] = syn1Neg + (target * vectorLength); // syn1Neg instead of syn1

                    //printf("Negative round: target: [%i]; code: [%i]; neu1e[0]: [%f]\n", target, idxArgs[4], neu1e[0]);

                    NegativeSampling<T>::executeAggregate((T **)args, 4, nullptr, 0, idxArgs, 3, nullptr, 0, realArguments, 2);
                }


            // if we don't train words - we skip start of idxSyn0
            int starter = trainWords == 1 ? 0 : idxSyn0Length - numLabels;

            // propagate neu1e -> syn0
            if (!isInference) {
                for (int c = starter; c < idxSyn0Length; c++) {
                    T *syn0word = arguments[0] + (idxSyn0[c] * vectorLength);
#pragma omp simd
                    for (int i = 0; i < vectorLength; i++) {
                        syn0word[i] += neu1e[i];
                    }
                }
            } else {
#pragma omp simd
                for (int i = 0; i < vectorLength; i++) {
                    inferenceVector[i] += neu1e[i];
                }
            }



            delete[] neu1;
            delete[] neu1e;
        }


#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, Nd4jLong **shapeArguments, int numShapeArguments,
                         int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,
                         T *realArguments, int numRealArguments) {
            __shared__ int vectorLength;
            __shared__ int hsRounds;
            __shared__ int ngRounds;
            __shared__ int expLength;
            __shared__ int vocabSize;
            __shared__ int ngStarter;
            __shared__ int negTableLength;
            __shared__ int idxSyn0Length;
            __shared__ int initialIdx;
            __shared__ int numLabels;
            __shared__ int trainWords;
            __shared__ int isInference;

            int *idxSyn0 = intArrays[0];
            int *idxSyn1 = intArrays[1];
            int *codes = intArrays[2];

            __shared__ T *neu1;
            __shared__ T *neu1e;

            __shared__ T *args[5];
            __shared__ int idxArgs[4];

            T *syn0 = arguments[0];
            T *syn1 = arguments[1];
            //T *expTable = arguments[2];
            T *syn1Neg = arguments[3];
            T *negTable = arguments[4];
            T *inferenceVector = arguments[5];

            if (threadIdx.x == 0) {
                vectorLength = indexArguments[0];
                hsRounds = indexArguments[1];
                ngRounds = indexArguments[2];
                expLength = indexArguments[3];
                vocabSize = indexArguments[4];
                ngStarter = indexArguments[5];
                negTableLength = indexArguments[6];
                idxSyn0Length = indexArguments[7];
                initialIdx = indexArguments[8];
                numLabels = indexArguments[9];
                trainWords = indexArguments[10];
                isInference = indexArguments[11];

                extern __shared__ unsigned char shmem[];
                neu1 = (T *) shmem;
                neu1e = neu1 + vectorLength;

                args[0] = neu1;
                args[2] = arguments[2]; //expTable
                args[3] = neu1e;

                idxArgs[0] = vectorLength; // vectorLength
                idxArgs[1] = expLength; // expLength
                idxArgs[3] = isInference;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < vectorLength; i += blockDim.x) {
                neu1[i] = (T) 0.0f;
                neu1e[i] = (T) 0.0f;
            }

            unsigned long long next_random = (unsigned long long) realArguments[1];
            for (int c = 0; c < idxSyn0Length; c++) {
                T *syn0word = syn0 + (idxSyn0[c] * vectorLength);

                for (int i = threadIdx.x; i < vectorLength; i += blockDim.x) {
                    neu1[i] += syn0word[i];
                }
            }

            if (isInference)
                for (int i = threadIdx.x; i < vectorLength; i += blockDim.x) {
                    neu1[i] += inferenceVector[i];
                }

            // average neu1
            if (idxSyn0Length > 0) {
                for (int i = threadIdx.x; i < vectorLength; i += blockDim.x) {
                    neu1[i] /= idxSyn0Length + + isInference;
                }
            }
            __syncthreads();



            if (hsRounds > 0)
                for (int i = 0; i < hsRounds; i++) {
                    if (threadIdx.x == 0) {
                        args[1] = syn1 + (idxSyn1[i] * vectorLength);
                        idxArgs[2] = codes[i];
                    }
                    __syncthreads();

                    HierarchicSoftmax<T>::executeAggregateCuda(args, 4, nullptr, 0, idxArgs, 3, nullptr, 0, realArguments, 2);
                }

            __shared__ int target;
            if (ngRounds > 0)
                for (int i = 0; i < ngRounds + 1; i++) {
                    if (threadIdx.x == 0) {
                        if (i == 0) {
                            target = ngStarter;
                        } else {
                            next_random = next_random * (unsigned long long) 25214903917 + 11;
                            target = negTable[(next_random >> 16) % negTableLength];

                            if (target <= 0 || target >= vocabSize) target = next_random % (vocabSize - 1) + 1;
                        }

                        args[1] = syn1Neg + (target * vectorLength); // syn1Neg instead of syn1
                        idxArgs[2] = i == 0 ? 1 : 0;
                    }
                    __syncthreads();

                    if (i != 0 && target == ngStarter)
                            continue;


                    NegativeSampling<T>::executeAggregateCuda(args, 4, nullptr, 0, idxArgs, 3, nullptr, 0, realArguments, 2);

                    //printf("Negative round: target: [%i]; code: [%i]; neu1[%i]: [%f]; neu1e[%i]: [%f]\n", target, idxArgs[2], threadIdx.x, neu1[threadIdx.x], threadIdx.x, neu1e[threadIdx.x]);
                }


            // if we don't train words - we skip start of idxSyn0
            int starter = trainWords == 1 ? 0 : idxSyn0Length - numLabels;

            if (!isInference)
                for (int c = starter; c < idxSyn0Length; c++) {
                    T *syn0word = arguments[0] + (idxSyn0[c] * vectorLength);

                    for (int i = threadIdx.x; i < vectorLength; i += blockDim.x) {
                        syn0word[i] += neu1e[i];
                    }
                }
            else {
                for (int i = threadIdx.x; i < vectorLength; i += blockDim.x) {
                        inferenceVector[i] += neu1e[i];
                }
            }

        }
#endif
    };

}

#endif //LIBND4J_AGGREGATE_OPS_H
