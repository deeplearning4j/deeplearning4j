/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef CUDA_LAMBDA_HELPER
#define CUDA_LAMBDA_HELPER

#include <pointercast.h>
#include <op_boilerplate.h>
#include <helpers/shape.h>
#include <cuda.h>
#include <cuda_runtime.h>

static Nd4jLong __device__ __noinline__ __getIndexOffset(Nd4jLong index, Nd4jLong *shapeInfo, Nd4jLong length) {
    return shape::getIndexOffset(index, shapeInfo, length);
}

static Nd4jLong __device__ __noinline__ __length(Nd4jLong *shapeInfo) {
    return shape::length(shapeInfo);
}

template <typename T, typename Lambda> static _CUDA_G void lambdaKernel(void* vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda);
template <typename T, typename Lambda> static _CUDA_G void lambdaIndexedKernel(void* vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda);
template <typename T, typename Lambda> static _CUDA_G void lambdaIndexedPairwiseKernel(void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda);
template <typename T, typename Lambda> static _CUDA_G void lambdaPairwiseKernel(void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda);
template <typename T, typename Lambda> static _CUDA_G void lambdaTriplewiseKernel(void* vw, Nd4jLong *wShapeInfo, void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda);

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

    template <typename Lambda>
    FORCEINLINE static void lambdaIndexedLauncher(cudaStream_t *stream, void* vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
        lambdaIndexedKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, lambda);
        auto err = cudaStreamSynchronize(*stream);
        if (err != 0)
            throw std::runtime_error("NDArray::applyIndexedLambda execution failed");
    }

    template <typename Lambda>
    FORCEINLINE static void lambdaPairwiseLauncher(cudaStream_t *stream, void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
        lambdaPairwiseKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, lambda);
        auto err = cudaStreamSynchronize(*stream);
        if (err != 0)
            throw std::runtime_error("NDArray::applyPairwiseLambda execution failed");
    }

    template <typename Lambda>
    FORCEINLINE static void lambdaIndexedPairwiseLauncher(cudaStream_t *stream, void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
        lambdaIndexedPairwiseKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, lambda);
        auto err = cudaStreamSynchronize(*stream);
        if (err != 0)
            throw std::runtime_error("NDArray::applyIndexedPairwiseLambda execution failed");
    }

    template <typename Lambda>
    FORCEINLINE static void lambdaTriplewiseLauncher(cudaStream_t *stream, void* vw, Nd4jLong *wShapeInfo, void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
        lambdaTriplewiseKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vw, wShapeInfo, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, lambda);
        auto err = cudaStreamSynchronize(*stream);
        if (err != 0)
            throw std::runtime_error("NDArray::applyTriplewiseLambda execution failed");
    }
};

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static _CUDA_G void lambdaKernel(void* vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
    auto x = reinterpret_cast<T*>(vx);
    auto z = reinterpret_cast<T*>(vz);

    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto zEws = shape::elementWiseStride(zShapeInfo);

    auto xOrder = shape::order(xShapeInfo);
    auto zOrder = shape::order(zShapeInfo);

    auto zLength = __length(zShapeInfo);

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x)
            z[e * zEws] = lambda(x[e * xEws]);
    } else {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x) {
            auto xOffset = __getIndexOffset(e, xShapeInfo, zLength);
            auto zOffset = __getIndexOffset(e, zShapeInfo, zLength);

            z[zOffset] = lambda(x[xOffset]);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static _CUDA_G void lambdaIndexedKernel(void* vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
    auto x = reinterpret_cast<T*>(vx);
    auto z = reinterpret_cast<T*>(vz);

    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto zEws = shape::elementWiseStride(zShapeInfo);

    auto xOrder = shape::order(xShapeInfo);
    auto zOrder = shape::order(zShapeInfo);

    auto zLength = __length(zShapeInfo);

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x)
            z[e * zEws] = lambda(e, x[e * xEws]);
    } else {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x) {
            auto xOffset = __getIndexOffset(e, xShapeInfo, zLength);
            auto zOffset = __getIndexOffset(e, zShapeInfo, zLength);

            z[zOffset] = lambda(e, x[xOffset]);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static _CUDA_G void lambdaIndexedPairwiseKernel(void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
    auto x = reinterpret_cast<T*>(vx);
    auto y = reinterpret_cast<T*>(vy);
    auto z = reinterpret_cast<T*>(vz);

    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);
    auto zEws = shape::elementWiseStride(zShapeInfo);

    auto xOrder = shape::order(xShapeInfo);
    auto yOrder = shape::order(yShapeInfo);
    auto zOrder = shape::order(zShapeInfo);

    auto zLength = __length(zShapeInfo);

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == zOrder && yOrder == xOrder) {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x)
            z[e * zEws] = lambda(e, x[e * xEws], y[e * yEws]);
    } else {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x) {
            auto xOffset = __getIndexOffset(e, xShapeInfo, zLength);
            auto yOffset = __getIndexOffset(e, yShapeInfo, zLength);
            auto zOffset = __getIndexOffset(e, zShapeInfo, zLength);

            z[zOffset] = lambda(e, x[xOffset], y[yOffset]);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static _CUDA_G void lambdaPairwiseKernel(void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
    auto x = reinterpret_cast<T*>(vx);
    auto y = reinterpret_cast<T*>(vy);
    auto z = reinterpret_cast<T*>(vz);

    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);
    auto zEws = shape::elementWiseStride(zShapeInfo);

    auto xOrder = shape::order(xShapeInfo);
    auto yOrder = shape::order(yShapeInfo);
    auto zOrder = shape::order(zShapeInfo);

    auto zLength = __length(zShapeInfo);

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == zOrder && yOrder == xOrder) {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x)
            z[e * zEws] = lambda(x[e * xEws], y[e * yEws]);
    } else {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x) {
            auto xOffset = __getIndexOffset(e, xShapeInfo, zLength);
            auto yOffset = __getIndexOffset(e, yShapeInfo, zLength);
            auto zOffset = __getIndexOffset(e, zShapeInfo, zLength);

            z[zOffset] = lambda(x[xOffset], y[yOffset]);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static _CUDA_G void lambdaTriplewiseKernel(void* vw, Nd4jLong *wShapeInfo, void* vx, Nd4jLong *xShapeInfo, void* vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, Lambda lambda) {
    auto w = reinterpret_cast<T*>(vw);
    auto x = reinterpret_cast<T*>(vx);
    auto y = reinterpret_cast<T*>(vy);
    auto z = reinterpret_cast<T*>(vz);

    auto wEws = shape::elementWiseStride(wShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);
    auto zEws = shape::elementWiseStride(zShapeInfo);

    auto wOrder = shape::order(wShapeInfo);
    auto xOrder = shape::order(xShapeInfo);
    auto yOrder = shape::order(yShapeInfo);
    auto zOrder = shape::order(zShapeInfo);

    auto zLength = __length(zShapeInfo);

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (wEws > 1 && xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == zOrder && yOrder == xOrder && wOrder == xOrder) {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x)
            z[e * zEws] = lambda(w[e * wEws], x[e * xEws], y[e * yEws]);
    } else {
        for (uint e = tid; e < zLength; e += blockDim.x * gridDim.x) {
            auto wOffset = __getIndexOffset(e, wShapeInfo, zLength);
            auto xOffset = __getIndexOffset(e, xShapeInfo, zLength);
            auto yOffset = __getIndexOffset(e, yShapeInfo, zLength);
            auto zOffset = __getIndexOffset(e, zShapeInfo, zLength);

            z[zOffset] = lambda(w[wOffset], x[xOffset], y[yOffset]);
        }
    }
}

#endif

//////////////////////////////////////////////////////////////////////////
template<typename Lambda>
void NDArray::applyLambda(Lambda func, NDArray* target) {
    auto result = target == nullptr ? this : target;
    auto dtype = this->dataType();

    if (dtype != result->dataType())
        throw std::runtime_error("NDArray::applyLambda X/Z data types must be the same");
        //throw datatype_exception::build("NDArray::applyLambda X/Z data types must be the same", dtype, result->dataType());
    prepareSpecialUse({result}, {this});
    BUILD_SINGLE_SELECTOR(dtype, LambdaHelper ,::lambdaLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(), result->specialBuffer(), result->specialShapeInfo(), func), LIBND4J_TYPES);
    registerSpecialUse({result}, {this});

}

//////////////////////////////////////////////////////////////////////////
template<typename Lambda>
void NDArray::applyPairwiseLambda(const NDArray* other, Lambda func, NDArray* target) {
    auto result = target == nullptr ? this : target;
    auto dtype = this->dataType();

    if (dtype != result->dataType() || dtype != other->dataType())
        throw std::runtime_error("NDArray::applyPairwiseLambda X/Y/Z data types must be the same");
    //throw datatype_exception::build("NDArray::applyLambda X/Z data types must be the same", dtype, result->dataType());

    prepareSpecialUse({result}, {this, other});
    BUILD_SINGLE_SELECTOR(dtype, LambdaHelper ,::lambdaPairwiseLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(), other->getSpecialBuffer(), other->getSpecialShapeInfo(), result->specialBuffer(), result->specialShapeInfo(), func), LIBND4J_TYPES);
    registerSpecialUse({result}, {this, other});

}

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyIndexedLambda(Lambda func, NDArray* target) {
    auto result = target == nullptr ? this : target;
    auto dtype = this->dataType();
    if (dtype != result->dataType())
        throw std::runtime_error("NDArray::applyIndexedLambda X/Z data types must be the same");

    prepareSpecialUse({result}, {this});
    BUILD_SINGLE_SELECTOR(dtype, LambdaHelper ,::lambdaIndexedLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(), result->specialBuffer(), result->specialShapeInfo(), func), LIBND4J_TYPES);
    registerSpecialUse({result}, {this});
}

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyIndexedPairwiseLambda(NDArray* other, Lambda func, NDArray* target) {
    auto result = target == nullptr ? this : target;
    auto dtype = this->dataType();
    if (dtype != result->dataType() || dtype != other->dataType())
        throw std::runtime_error("NDArray::applyIndexedPairwiseLambda X/Y/Z data types must be the same");

    prepareSpecialUse({result}, {this, other});
    BUILD_SINGLE_SELECTOR(dtype, LambdaHelper ,::lambdaIndexedPairwiseLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(), other->getSpecialBuffer(), other->getSpecialShapeInfo(), result->specialBuffer(), result->specialShapeInfo(), func), LIBND4J_TYPES);
    registerSpecialUse({result}, {this, other});
}

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, Lambda func, NDArray* target) {
    auto result = target == nullptr ? this : target;
    auto dtype = this->dataType();

    if (dtype != result->dataType() || dtype != second->dataType() || dtype != third->dataType())
        throw std::runtime_error("NDArray::applyTriplewiseLambda X/Y/Z data types must be the same");

    prepareSpecialUse({result}, {this, second, third});
    BUILD_SINGLE_SELECTOR(dtype, LambdaHelper ,::lambdaTriplewiseLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(), second->specialBuffer(), second->specialShapeInfo(), third->specialBuffer(), third->specialShapeInfo(), result->specialBuffer(), result->specialShapeInfo(), func), LIBND4J_TYPES);
    registerSpecialUse({result}, {this, second, third});
}





