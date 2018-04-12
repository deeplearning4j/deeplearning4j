//
// Created by raver119 on 10.10.17.
//

#ifndef PROJECT_GRID_STRIDED_H
#define PROJECT_GRID_STRIDED_H


namespace functions {
    namespace grid {
        template <typename T>
        class GRIDStrided {
        public:
            static void execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jIndex N, T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB);

            template<typename OpType>
            static __device__ void transformCuda(Nd4jIndex n, T *dx, T *dy, int incx, int incy, T *params, T *result, int incz,int *allocationPointer, UnifiedSharedMemory *manager,int *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jIndex n, T *dx, T *dy, int incx, int incy, T *params, T *result, int incz,int *allocationPointer, UnifiedSharedMemory *manager,int *tadOnlyShapeInfo);
        };
    }
}

#endif //PROJECT_GRID_STRIDED_H
