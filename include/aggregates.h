//
// @author raver119@gmail.com
//

#ifndef LIBND4J_AGGREGATES_H
#define LIBND4J_AGGREGATES_H

#include <aggregate_ops.h>
#include <helper_ptrmap.h>

#define AGGREGATE_OPS \
        (0, aggregateOps::HierarchicSoftmax) ,\
        (1, aggregateOps::Dot) ,\
        (2, aggregateOps::Axpy) ,\
        (3, aggregateOps::SkipGram)


namespace functions {
    namespace aggregate {

        template<typename T>
        class AggregatedFunction {

        public:
#ifdef __CUDACC__
            template<typename OpClass>
            __device__ inline static void execCuda(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  T *realArguments, int numRealArguments) {
                OpClass::executeAggregateCuda(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
            }
#endif

            template<typename OpClass>
            inline static void exec(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,  T *realArguments, int numRealArguments) {
                OpClass::executeAggregate(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
            }

            inline static void exec(int opNum, T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
                DISPATCH_BY_OPNUM(exec, PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), AGGREGATE_OPS);
            }
		};
    }
}

#ifdef __CUDACC__

template <typename T, typename OpClass>
__device__ void aggregateGeneric(T **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, T *realArguments, int numRealArguments) {
    functions::aggregate::AggregatedFunction<T>:: template execCuda<OpClass>(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments);
};


template <typename T, typename OpClass>
__device__ void aggregateBatchGeneric(int numAggregates, int opNum, void *ptrToArguments) {

    // helper should be in __shared__ memory probably, no sense using stack here
    nd4j::PointersHelper<T> helper(ptrToArguments, numAggregates);

    for(int r = blockIdx.x; r < numAggregates; r += gridDim.x) {
        T **arguments = helper.getArguments(r);
        int **shapes = helper.getShapeArguments(r);
        int *idxArg = helper.getIndexArguments(r);
        T *realArg = helper.getRealArguments(r);

        //functions::aggregate::AggregatedFunction<T>:: template execCuda<OpClass>(arguments, numArguments[r], shapes, numShapes[r], idxArg, numIndexArguments[r], realArg, numRealArguments[r]);
    }
};

// simple aggregates
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, float, INPUT(float **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, float *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, double, INPUT(double **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays, double *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, intArrays, numIntArrays, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
//DISPATCH_KERNEL_SIMPLE(aggregateSimple_, aggregateGeneric, float16, INPUT(float16 **arguments, int numArguments, int **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments, float16 *realArguments, int numRealArguments), PARAMS(arguments, numArguments, shapeArguments, numShapeArguments, indexArguments, numIndexArguments, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

// batched aggregates
DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, float, INPUT(int numAggregates, int ops, void *ptrToArguments), PARAMS(numAggregates, ops, ptrToArguments), OPS_A(AGGREGATE_OPS))
//DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, double, INPUT(int numAggregates, int *ops, Nd4jPointer *ptrToArguments, int *numArguments, Nd4jPointer *ptrToShapes, int *numShapes, int **indexArguments, int *numIndexArguments, double **realArguments, int *numRealArguments), PARAMS(numAggregates, ops, ptrToArguments, numArguments, ptrToShapes, numShapes, indexArguments, numIndexArguments, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))
//DISPATCH_KERNEL_SIMPLE(aggregateBatchSimple_, aggregateBatchGeneric, float16, INPUT(int numAggregates, int *ops, Nd4jPointer *ptrToArguments, int *numArguments, Nd4jPointer *ptrToShapes, int *numShapes, int **indexArguments, int *numIndexArguments, float16 **realArguments, int *numRealArguments), PARAMS(numAggregates, ops, ptrToArguments, numArguments, ptrToShapes, numShapes, indexArguments, numIndexArguments, realArguments, numRealArguments), OPS_A(AGGREGATE_OPS))

#endif

#endif //LIBND4J_AGGREGATES_H
