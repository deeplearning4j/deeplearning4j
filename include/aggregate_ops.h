//
// @author raver119@gmail.com
//
#ifndef LIBND4J_AGGREGATE_OPS_H
#define LIBND4J_AGGREGATE_OPS_H

#include <ops.h>
#include <templatemath.h>

#define HS_MAX_EXP 6.0f

#ifdef __CUDACC__
#define aggregate_def __device__ inline static
#else
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
    class HierarchicSoftmax {
        public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
            int idxSyn0 = indexArguments[0];
            int idxSyn1 = indexArguments[1];
            int vectorLength = indexArguments[2];
            int expLength = indexArguments[3];
            int code = indexArguments[4];

            T *syn0 = arguments[0] + (idxSyn0 * vectorLength);
            T *syn1 = arguments[1] + (idxSyn1 * vectorLength);
            T *expTable = arguments[2];
            T *neu1e = arguments[3];

            T dot = (T) 0.0f;
            T g = (T) 0.0f;
            T f = (T) 0.0f;
            T alpha = realArguments[0];

            // dot
#pragma omp simd reduction(+:dot)
            for (int x = 0; x < vectorLength; x++) {
                dot += syn0[x] * syn1[x];
            }

            // gradient
            if (dot < - HS_MAX_EXP || dot >= HS_MAX_EXP)
                return;

            int idx = (int) ((dot + HS_MAX_EXP) * ((T) expLength / HS_MAX_EXP / 2.0));

            if (idx >= expLength)
                return;

            f = expTable[idx];
            g = (1 - code - f) * alpha;

            // axpy1
#pragma omp simd
            for (int x = 0; x < vectorLength; x++) {
                neu1e[x] = g * syn1[x] + neu1e[x];
            }

            // axpy2
#pragma omp simd
            for (int x = 0; x < vectorLength; x++) {
                syn1[x] = g * syn0[x] + syn1[x];
            }
        }

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
            /*
                We know that syn0 & syn1 are 2D matrices, so we can just use offsets here
            */

            __shared__ int idxSyn0;
            __shared__ int idxSyn1;
            __shared__ int vectorLength;
            __shared__ int expLength;
            __shared__ int code;

            __shared__ T *syn0;
            __shared__ T *syn1;
            __shared__ T *expTable;

            __shared__ T *neu1e;
            __shared__ T dot;
            __shared__ T g;
            __shared__ T f;
            __shared__ T alpha;

            if (threadIdx.x == 0) {
                idxSyn0 = indexArguments[0];
                idxSyn1 = indexArguments[1];
                vectorLength = indexArguments[2];
                expLength = indexArguments[3];
                code = indexArguments[4];

                syn0 = arguments[0] + (idxSyn0 * vectorLength);
                syn1 = arguments[1] + (idxSyn1 * vectorLength);
                expTable = arguments[2];
                neu1e = arguments[3];

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

            int idx = (int) ((dot + HS_MAX_EXP) * ((T) expLength / HS_MAX_EXP / 2.0));

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
            __syncthreads();

            // axpy2
            for (int x = threadIdx.x; x < vectorLength; x+=blockDim.x) {
                syn1[x] = g * syn0[x] + syn1[x];
            }
            __syncthreads();

        }
#endif
    };

    template<typename T>
    class Dot {
    public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
            T *vecX = arguments[0];
            T *vecY = arguments[1];
            T *vecZ = arguments[2];

            T dot = (T) 0.0f;

            int vectorLength = indexArguments[0];

#pragma omp simd reduction(+:dot)
            for (int x = 0; x < vectorLength; x++) {
                dot += vecX[x] * vecY[x];
            }

            vecZ[0] = dot;
        };

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
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

        aggregate_def void executeAggregate(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
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
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
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

        aggregate_def void
        executeAggregate(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments,
                         int numRealArguments) {
            int syn0Row = indexArguments[0];
            int vectorLength = indexArguments[1];
            int hsRounds = indexArguments[2];
            int ngRounds = indexArguments[3];
            int expLength = indexArguments[4];
            int vocabSize = indexArguments[5];
            int ngStarter = indexArguments[6];



            T *neu1e = new T[vectorLength];
            std::memset(neu1e, 0, sizeof(T) * vectorLength);

            T **args = new T *[4];
            int *idxArgs = new int[5];
            args[0] = arguments[0]; // syn0
            args[1] = arguments[1]; // syn1
            args[2] = arguments[2]; // expTable
            args[3] = neu1e;


            idxArgs[0] = indexArguments[0]; // syn0 row
            idxArgs[2] = indexArguments[1]; // vectorLength
            idxArgs[3] = indexArguments[4]; // expLength

            T *syn0 = arguments[0] + (syn0Row * vectorLength);

            int *idxSyn1 = shapeArguments[0];
            int *codes = shapeArguments[1];

            for (int r = 0; r < hsRounds; r++) {
                idxArgs[1] = idxSyn1[r]; // syn1 row
                idxArgs[4] = codes[r];  // code for row

                HierarchicSoftmax<T>::executeAggregate(args, 4, nullptr, 0, idxArgs, 5, realArguments, 1);
            }

            args[1] = arguments[3]; // syn1Neg instead of syn1

            if (ngRounds > 0)
                for (int r = 0; r < ngRounds + 1; r++) {
                    if (r == 0) {
                        idxArgs[1] = ngStarter;
                        idxArgs[4] = 1;
                    } else {
                        int target;


                        if (target == ngStarter)
                            continue;

                        idxArgs[1] = 1;
                        idxArgs[4] = 0;
                    }
                    HierarchicSoftmax<T>::executeAggregate(args, 4, nullptr, 0, idxArgs, 5, realArguments, 1);
                }

#pragma omp simd
            for (int x = 0; x < vectorLength; x++) {
                syn0[x] += neu1e[x];
            }

            delete[] neu1e;
            delete[] args;
            delete[] idxArgs;
        }

#ifdef __CUDACC__
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {

        }
#endif
    };
}

#endif //LIBND4J_AGGREGATE_OPS_H
