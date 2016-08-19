//
// @author raver119@gmail.com
//
//
#ifndef LIBND4J_GRID_H
#define LIBND4J_GRID_H


template<typename OpTypeA, typename OpTypeB>
static inline __device__ void transformCuda(
        Nd4jIndex n,
        T dx,
        T *dy,
        int incy,
        T *params,
        T *result,
        int resultStride) {

    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Nd4jIndex i = tid;
    if(incy == 1 && resultStride == 1) {

        for (; i < n; i += totalThreads) {
            result[i] = OpTypeB::op(OpTypeA::op(dy[i], paramsA), paramsB);
        }
    }
    else {

        for (; i < n; i += totalThreads) {
            result[i * resultStride] = OpTypeB::op(OpTypeA::op(dy[i * incy], paramsA), paramsB);
        }
    }
}


template <typename T>
__device__ inline void metaStridedGeneric(int gridDepth, void **grid ) {

}


#endif //LIBND4J_GRID_H
