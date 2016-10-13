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
 * So, when batched, they should provide proper parallelism levels on poorly parallel tasks otherwise
 *
 *
 */
namespace aggregateOps {

    template<typename T>
    class HierarchicSoftmax {
        public:

        aggregate_def void executeAggregate(T **arguments, int numArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
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
// TODO: simd reduction required here
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
        aggregate_def void executeAggregateCuda(T **arguments, int numArguments, int *indexArguments, int numIndexArguments, T *realArguments, int numRealArguments) {
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
}

#endif //LIBND4J_AGGREGATE_OPS_H
