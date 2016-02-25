#include <math.h>
#include <stdio.h>
#include <sharedmem.h>
#include <broadcasting.h>
#include <shape.h>


namespace functions {
    namespace broadcast {


        template<typename T>
        class BaseBroadcast : public Broadcast<T> {
        public:
            __device__ void transform(
                    T *x, int *xShapeInfo, T *y, int *yShapeInfo, T *result, int *resultShapeInfo,
                    int *dimension,
                    int dimensionLength,
                    int *gpuInformation) {


                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int xOffset = shape::offset(xShapeInfo);
                int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
                int yOffset = shape::offset(yShapeInfo);



                //length for the tad
                int yLength = shape::length(yShapeInfo);
                //length for the tad
                int xLength = shape::length(xShapeInfo);

                int resultLength = shape::length(resultShapeInfo);
                for (int i = blockIdx.x * blockDim.x + threadIdx.x;
                     i < resultLength;
                     i += blockDim.x * gridDim.x) {
                    int yOffset2 = yOffset + ((i / xElementWiseStride) % yLength) * yElementWiseStride;
                    if (i < resultLength)
                        result[i] = op(x[i], y[yOffset2]);

                }

            }

        };


    }
}

