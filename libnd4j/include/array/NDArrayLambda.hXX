/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/shape.h>
#include <system/op_boilerplate.h>

static sd::LongType SD_DEVICE  getIndexOffset(sd::LongType index, const sd::LongType *shapeInfo) {
  return shape::getIndexOffset(index, shapeInfo);
}

static sd::LongType SD_DEVICE  length(const sd::LongType *shapeInfo) { return shape::length(shapeInfo); }

template <typename T, typename Lambda>
static SD_KERNEL void lambdaKernel(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                   const sd::LongType *zShapeInfo, Lambda lambda);
template <typename T, typename Lambda>
static SD_KERNEL void lambdaIndexedKernel(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                          const sd::LongType *zShapeInfo, Lambda lambda);
template <typename T, typename Lambda>
static SD_KERNEL void lambdaIndexedPairwiseKernel(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                                  const sd::LongType *yShapeInfo, void *vz,
                                                  const sd::LongType *zShapeInfo, Lambda lambda);
template <typename T, typename Lambda>
static SD_KERNEL void lambdaPairwiseKernel(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                           const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                           Lambda lambda);
template <typename T, typename Lambda>
static SD_KERNEL void lambdaPairwiseKernel(const void *scalarPtr, const void *vx, const sd::LongType *xShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                           Lambda lambda);
template <typename T, typename Lambda>
static SD_KERNEL void lambdaTriplewiseKernel(const void *vw, const sd::LongType *wShapeInfo, const void *vx,
                                             const sd::LongType *xShapeInfo, const void *vy,
                                             const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                             Lambda lambda);

template <typename T>
class LambdaHelper {
 public:
  template <typename Lambda>
  SD_INLINE static void lambdaLauncher(cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                       const sd::LongType *zShapeInfo, Lambda lambda) {
    lambdaKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, lambda);
    auto err = cudaStreamSynchronize(*stream);
    if (err != 0) THROW_EXCEPTION("NDArray::applyLambda execution failed");
  }

  template <typename Lambda>
  SD_INLINE static void lambdaIndexedLauncher(cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo,
                                              void *vz, const sd::LongType *zShapeInfo, Lambda lambda) {
    lambdaIndexedKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, lambda);
    auto err = cudaStreamSynchronize(*stream);
    if (err != 0) THROW_EXCEPTION("NDArray::applyIndexedLambda execution failed");
  }

  template <typename Lambda>
  SD_INLINE static void lambdaPairwiseLauncher(cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo, bool otherIsScalar,
                                               const void *vy, const sd::LongType *yShapeInfo, void *vz,
                                               const sd::LongType *zShapeInfo, Lambda lambda) {
    if (otherIsScalar) {
      lambdaPairwiseKernel<T, Lambda><<<256, 512, 1024, *stream>>>(vy, vx, xShapeInfo, vz, zShapeInfo, lambda);
    } else {
      lambdaPairwiseKernel<T, Lambda>
          <<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, lambda);
    }
    auto err = cudaStreamSynchronize(*stream);
    if (err != 0) THROW_EXCEPTION("NDArray::applyPairwiseLambda execution failed");
  }

  template <typename Lambda>
  SD_INLINE static void lambdaIndexedPairwiseLauncher(cudaStream_t *stream, const void *vx,
                                                      const sd::LongType *xShapeInfo, const void *vy,
                                                      const sd::LongType *yShapeInfo, void *vz,
                                                      const sd::LongType *zShapeInfo, Lambda lambda) {
    lambdaIndexedPairwiseKernel<T, Lambda>
        <<<256, 512, 1024, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, lambda);
    auto err = cudaStreamSynchronize(*stream);
    if (err != 0) THROW_EXCEPTION("NDArray::applyIndexedPairwiseLambda execution failed");
  }

  template <typename Lambda>
  SD_INLINE static void lambdaTriplewiseLauncher(cudaStream_t *stream, const void *vw, const sd::LongType *wShapeInfo,
                                                 const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                                 const sd::LongType *yShapeInfo, void *vz,
                                                 const sd::LongType *zShapeInfo, Lambda lambda) {
    lambdaTriplewiseKernel<T, Lambda>
        <<<256, 512, 1024, *stream>>>(vw, wShapeInfo, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, lambda);
    auto err = cudaStreamSynchronize(*stream);
    if (err != 0) THROW_EXCEPTION("NDArray::applyTriplewiseLambda execution failed");
  }
};

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static SD_KERNEL void lambdaKernel(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                   const sd::LongType *zShapeInfo, Lambda lambda) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  auto xOrder = shape::order(xShapeInfo);
  auto zOrder = shape::order(zShapeInfo);

  auto zLength = length(zShapeInfo);

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) z[e * zEws] = lambda(x[e * xEws]);
  } else {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) {
      auto xOffset = getIndexOffset(e, xShapeInfo);
      auto zOffset = getIndexOffset(e, zShapeInfo);

      z[zOffset] = lambda(x[xOffset]);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static SD_KERNEL void lambdaIndexedKernel(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                          const sd::LongType *zShapeInfo, Lambda lambda) {
  auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  auto xOrder = shape::order(xShapeInfo);
  auto zOrder = shape::order(zShapeInfo);

  auto zLength = length(zShapeInfo);

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) z[e * zEws] = lambda(e, x[e * xEws]);
  } else {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) {
      auto xOffset = getIndexOffset(e, xShapeInfo);
      auto zOffset = getIndexOffset(e, zShapeInfo);

      z[zOffset] = lambda(e, x[xOffset]);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static SD_KERNEL void lambdaIndexedPairwiseKernel(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                                  const sd::LongType *yShapeInfo, void *vz,
                                                  const sd::LongType *zShapeInfo, Lambda lambda) {
  auto x = reinterpret_cast<const T *>(vx);
  auto y = reinterpret_cast<const T *>(vy);
  auto z = reinterpret_cast<T *>(vz);

  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto yEws = shape::elementWiseStride(yShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  auto xOrder = shape::order(xShapeInfo);
  auto yOrder = shape::order(yShapeInfo);
  auto zOrder = shape::order(zShapeInfo);

  auto zLength = length(zShapeInfo);

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == zOrder && yOrder == xOrder) {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x)
      z[e * zEws] = lambda(e, x[e * xEws], y[e * yEws]);
  } else {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) {
      auto xOffset = getIndexOffset(e, xShapeInfo);
      auto yOffset = getIndexOffset(e, yShapeInfo);
      auto zOffset = getIndexOffset(e, zShapeInfo);

      z[zOffset] = lambda(e, x[xOffset], y[yOffset]);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static SD_KERNEL void lambdaPairwiseKernel(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                           const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                           Lambda lambda) {
  auto x = reinterpret_cast<const T *>(vx);
  auto y = reinterpret_cast<const T *>(vy);
  auto z = reinterpret_cast<T *>(vz);

  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto yEws = shape::elementWiseStride(yShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  auto xOrder = shape::order(xShapeInfo);
  auto yOrder = shape::order(yShapeInfo);
  auto zOrder = shape::order(zShapeInfo);

  auto zLength = length(zShapeInfo);

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == zOrder && yOrder == xOrder) {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) z[e * zEws] = lambda(x[e * xEws], y[e * yEws]);
  } else {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) {
      auto xOffset = getIndexOffset(e, xShapeInfo);
      auto yOffset = getIndexOffset(e, yShapeInfo);
      auto zOffset = getIndexOffset(e, zShapeInfo);

      z[zOffset] = lambda(x[xOffset], y[yOffset]);
    }
  }
}
///////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static SD_KERNEL void lambdaPairwiseKernel(const void *scalarPtr, const void *vx, const sd::LongType *xShapeInfo,
                                           void *vz, const sd::LongType *zShapeInfo, Lambda lambda) {
  auto x = reinterpret_cast<const T *>(vx);
  auto y = reinterpret_cast<const T *>(scalarPtr);
  auto z = reinterpret_cast<T *>(vz);

  auto yVal = *y;

  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  auto xOrder = shape::order(xShapeInfo);
  auto zOrder = shape::order(zShapeInfo);

  auto zLength = length(zShapeInfo);

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) z[e * zEws] = lambda(x[e * xEws], yVal);
  } else {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) {
      auto xOffset = getIndexOffset(e, xShapeInfo);
      auto zOffset = getIndexOffset(e, zShapeInfo);

      z[zOffset] = lambda(x[xOffset], yVal);
    }
  }
}
////////////////////////////////////////////////////////////////////////
template <typename T, typename Lambda>
static SD_KERNEL void lambdaTriplewiseKernel(const void *vw, const sd::LongType *wShapeInfo, const void *vx,
                                             const sd::LongType *xShapeInfo, const void *vy,
                                             const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                             Lambda lambda) {
  auto w = reinterpret_cast<const T *>(vw);
  auto x = reinterpret_cast<const T *>(vx);
  auto y = reinterpret_cast<const T *>(vy);
  auto z = reinterpret_cast<T *>(vz);

  auto wEws = shape::elementWiseStride(wShapeInfo);
  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto yEws = shape::elementWiseStride(yShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  auto wOrder = shape::order(wShapeInfo);
  auto xOrder = shape::order(xShapeInfo);
  auto yOrder = shape::order(yShapeInfo);
  auto zOrder = shape::order(zShapeInfo);

  auto zLength = length(zShapeInfo);

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (wEws > 1 && xEws >= 1 && yEws >= 1 && zEws >= 1 && xOrder == zOrder && yOrder == xOrder && wOrder == xOrder) {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x)
      z[e * zEws] = lambda(w[e * wEws], x[e * xEws], y[e * yEws]);
  } else {
    for (sd::LongType e = tid; e < zLength; e += blockDim.x * gridDim.x) {
      auto wOffset = getIndexOffset(e, wShapeInfo);
      auto xOffset = getIndexOffset(e, xShapeInfo);
      auto yOffset = getIndexOffset(e, yShapeInfo);
      auto zOffset = getIndexOffset(e, zShapeInfo);

      z[zOffset] = lambda(w[wOffset], x[xOffset], y[yOffset]);
    }
  }
}

#endif

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyLambda(Lambda func, NDArray &target) {
  auto dtype = this->dataType();

  if (dtype != target.dataType()) THROW_EXCEPTION("NDArray::applyLambda X/Z data types must be the same");
  prepareSpecialUse({&target}, {this});
  BUILD_SINGLE_SELECTOR(
      dtype, LambdaHelper,
      ::lambdaLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(),
                       target.specialBuffer(), target.specialShapeInfo(), func),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyPairwiseLambda(NDArray&other, Lambda func, NDArray &target) {
  auto dtype = this->dataType();

  if (dtype != target.dataType() || dtype != other.dataType())
    THROW_EXCEPTION("NDArray::applyPairwiseLambda X/Y/Z data types must be the same");
  bool otherIsScalar = other.isScalar();
  prepareSpecialUse({&target}, {this, &other});
  BUILD_SINGLE_SELECTOR(
      dtype, LambdaHelper,
      ::lambdaPairwiseLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(), otherIsScalar,
                               other.specialBuffer(), other.specialShapeInfo(), target.specialBuffer(),
                               target.specialShapeInfo(), func),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyIndexedLambda(Lambda func, NDArray &target) {
  auto dtype = this->dataType();
  if (dtype != target.dataType())
    THROW_EXCEPTION("NDArray::applyIndexedLambda X/Z data types must be the same");

  prepareSpecialUse({&target}, {this});
  BUILD_SINGLE_SELECTOR(
      dtype, LambdaHelper,
      ::lambdaIndexedLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(),
                              target.specialBuffer(), target.specialShapeInfo(), func),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyIndexedPairwiseLambda(NDArray &other, Lambda func, NDArray &target) {
  auto dtype = this->dataType();
  if (dtype != target.dataType() || dtype != other.dataType())
    THROW_EXCEPTION("NDArray::applyIndexedPairwiseLambda X/Y/Z data types must be the same");

  prepareSpecialUse({&target}, {this, &other});
  BUILD_SINGLE_SELECTOR(
      dtype, LambdaHelper,
      ::lambdaIndexedPairwiseLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(),
                                      other.specialBuffer(), other.specialShapeInfo(), target.specialBuffer(),
                                      target.specialShapeInfo(), func),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this, &other});
}

//////////////////////////////////////////////////////////////////////////
template <typename Lambda>
void NDArray::applyTriplewiseLambda(NDArray &second, NDArray &third, Lambda func, NDArray &target) {
  auto dtype = this->dataType();

  if (dtype != target.dataType() || dtype != second.dataType() || dtype != third.dataType())
    THROW_EXCEPTION("NDArray::applyTriplewiseLambda X/Y/Z data types must be the same");

  prepareSpecialUse({&target}, {this, &second, &third});
  BUILD_SINGLE_SELECTOR(
      dtype, LambdaHelper,
      ::lambdaTriplewiseLauncher(this->_context->getCudaStream(), this->specialBuffer(), this->specialShapeInfo(),
                                 second.specialBuffer(), second.specialShapeInfo(), third.specialBuffer(),
                                 third.specialShapeInfo(), target.specialBuffer(), target.specialShapeInfo(), func),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this, &second, &third});
}
