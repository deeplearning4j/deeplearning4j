
#ifndef CUDA_LAMBDA_HELPER
#define CUDA_LAMBDA_HELPER

#include <pointercast.h>
#include <op_boilerplate.h>
#include <helpers/shape.h>

template <typename T, typename Lambda>
_CUDA_G void lambdaKernel(void* vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
    auto x = reinterpret_cast<T*>(vx);
    auto z = reinterpret_cast<T*>(vz);

    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto zEws = shape::elementWiseStride(zShapeInfo);

    auto xOrder = shape::order(xShapeInfo);
    auto zOrder = shape::order(zShapeInfo);

    auto zLength = shape::length(zShapeInfo);

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x)
            z[e * zEws] = lambda(x[e * xEws]);
    } else {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x) {
            auto xOffset = shape::getIndexOffset(e, xShapeInfo, zLength);
            auto zOffset = shape::getIndexOffset(e, zShapeInfo, zLength);

            z[zOffset] = lambda(x[xOffset]);
        }
    }
}


template <typename T>
class LambdaHelper {
public:
    template <typename Lambda>
    FORCEINLINE static void lambdaLauncher(cudaStream_t *stream, void* vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
        lambdaKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, lambda);
        auto err = cudaStreamSynchronize(*stream);
        if (err != 0)
            throw std::runtime_error("NDArray::applyLambda execution failed");
    }
};

#endif


template<typename Lambda>
void NDArray::applyLambda(Lambda func, NDArray* target) {
    LambdaHelper<double>::template lambdaLauncher<Lambda>(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(), target->specialBuffer(), target->specialShapeInfo(), func);
}

