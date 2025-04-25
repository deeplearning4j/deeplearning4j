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

//
//  @author sgazeos@gmail.com (CUDA implementation)
//
#include <array/NDArray.h>
#include <helpers/ShapeUtils.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/minimax.h>
#include <system/op_boilerplate.h>
#include <exceptions/cuda_exception.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// CUDA kernels for minimum and maximum backprop operations

// Kernel for element-wise case
template <typename T>
static SD_KERNEL void minimumMaximumScalarBPKernel(void* vx, const LongType* xShapeInfo, 
                                                  T scalarVal, bool isMin,
                                                  void* veps, const LongType* epsShapeInfo,
                                                  void* vgradX, const LongType* gradXShapeInfo) {
    const auto x = reinterpret_cast<T*>(vx);
    const auto eps = reinterpret_cast<T*>(veps);
    auto gradX = reinterpret_cast<T*>(vgradX);
    
    __shared__ LongType xRank, xLength;
    __shared__ const LongType *xShape, *xStride;
    __shared__ LongType epsRank, epsLength;
    __shared__ const LongType *epsShape, *epsStride;
    
    if (threadIdx.x == 0) {
        xRank = shape::rank(xShapeInfo);
        xLength = shape::length(xShapeInfo);
        xShape = shape::shapeOf(xShapeInfo);
        xStride = shape::stride(xShapeInfo);
        epsRank = shape::rank(epsShapeInfo);
        epsLength = shape::length(epsShapeInfo);
        epsShape = shape::shapeOf(epsShapeInfo);
        epsStride = shape::stride(epsShapeInfo);
    }
    __syncthreads();
    
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto step = gridDim.x * blockDim.x;
    
    for (LongType i = tid; i < xLength; i += step) {
        LongType xOffset = shape::getIndexOffset(i, xShapeInfo);
        LongType epsOffset = shape::getIndexOffset(i, epsShapeInfo);
        
        if (isMin) {
            // Minimum backprop
            gradX[xOffset] = x[xOffset] <= scalarVal ? eps[epsOffset] : static_cast<T>(0.0);
        } else {
            // Maximum backprop
            gradX[xOffset] = x[xOffset] >= scalarVal ? eps[epsOffset] : static_cast<T>(0.0);
        }
    }
}

// Kernel for element-wise case
template <typename T>
static SD_KERNEL void minimumMaximumBPKernel(void* vx, const LongType* xShapeInfo, 
                                           void* vy, const LongType* yShapeInfo,
                                           void* veps, const LongType* epsShapeInfo,
                                           void* vgradX, const LongType* gradXShapeInfo,
                                           void* vgradY, const LongType* gradYShapeInfo, 
                                           bool isMin) {
    const auto x = reinterpret_cast<T*>(vx);
    const auto y = reinterpret_cast<T*>(vy);
    const auto eps = reinterpret_cast<T*>(veps);
    auto gradX = reinterpret_cast<T*>(vgradX);
    auto gradY = reinterpret_cast<T*>(vgradY);
    
    __shared__ LongType length;
    
    if (threadIdx.x == 0) {
        length = shape::length(xShapeInfo);
    }
    __syncthreads();
    
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto step = gridDim.x * blockDim.x;
    
    for (LongType i = tid; i < length; i += step) {
        // Calculate offsets for each array
        LongType xOffset = shape::getIndexOffset(i, xShapeInfo);
        LongType yOffset = shape::getIndexOffset(i, yShapeInfo);
        LongType epsOffset = shape::getIndexOffset(i, epsShapeInfo);
        LongType gradXOffset = shape::getIndexOffset(i, gradXShapeInfo);
        LongType gradYOffset = shape::getIndexOffset(i, gradYShapeInfo);
        
        if (isMin) {
            // Minimum backprop
            gradX[gradXOffset] = x[xOffset] <= y[yOffset] ? eps[epsOffset] : static_cast<T>(0.0);
            gradY[gradYOffset] = x[xOffset] >= y[yOffset] ? eps[epsOffset] : static_cast<T>(0.0);
        } else {
            // Maximum backprop
            gradX[gradXOffset] = x[xOffset] >= y[yOffset] ? eps[epsOffset] : static_cast<T>(0.0);
            gradY[gradYOffset] = x[xOffset] <= y[yOffset] ? eps[epsOffset] : static_cast<T>(0.0);
        }
    }
}

template <typename T>
static void minimumBPFunctor_(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
    NDArray::prepareSpecialUse({gradX, gradY}, {x, y, epsNext});
    
    dim3 launchDims = getLaunchDims("minimax");
    
    if (x->isSameShape(y)) {
        // Element-wise case (same shapes)
        minimumMaximumBPKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
            x->specialBuffer(), x->specialShapeInfo(),
            y->specialBuffer(), y->specialShapeInfo(),
            epsNext->specialBuffer(), epsNext->specialShapeInfo(),
            gradX->specialBuffer(), gradX->specialShapeInfo(),
            gradY->specialBuffer(), gradY->specialShapeInfo(),
            true);  // isMin = true
    } else if (y->isScalar()) {
        // Scalar case
        T scalar = y->e<T>(0);
        
        minimumMaximumScalarBPKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
            x->specialBuffer(), x->specialShapeInfo(),
            scalar, true,  // isMin = true
            epsNext->specialBuffer(), epsNext->specialShapeInfo(),
            gradX->specialBuffer(), gradX->specialShapeInfo());
            
        // Set gradY value based on comparison
        if (*x <= *y) {
            auto tmp = epsNext->reduceNumber(reduce::Sum);
            gradY->assign(&tmp);
        } else {
            gradY->assign(static_cast<T>(0.0));
        }
    } else {
        // Broadcast case - more complex, falls back to CPU for now
        // We'd need to tile arrays to same shape, compute, then reduce along broadcast dims
        
        PointersManager manager(context, "minimumBPFunctor");
        manager.synchronize();
        
        // Move to host, perform calculation, then copy back to device
        x->syncToHost();
        y->syncToHost();
        epsNext->syncToHost();
        
        auto lambdaX = LAMBDA_TTT(_e, _x, _y) { return _x <= _y ? _e : (T)0.; });
        auto lambdaY = LAMBDA_TTT(_e, _x, _y) { return _x >= _y ? _e : (T)0.; });
        
        auto preX = x->dup();
        auto preY = y->dup();
        auto targetShape = epsNext->getShapeAsVector();
        
        preX.tileToShape(targetShape, preX);
        preY.tileToShape(targetShape, preY);
        
        epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaX, &preX);
        epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaY, &preY);
        
        auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
        auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());
        
        if (axisX.size() > 0) {
            auto sum = preX.reduceAlongDimension(reduce::Sum, &axisX);
            gradX->assign(&sum);
        } else {
            gradX->assign(&preX);
        }
        
        if (axisY.size() > 0) {
            auto sum = preY.reduceAlongDimension(reduce::Sum, &axisY);
            gradY->assign(&sum);
        } else {
            gradY->assign(&preY);
        }
        
        gradX->syncToDevice();
        gradY->syncToDevice();
    }
    
    DebugHelper::checkErrorCode(context->getCudaStream(), "minimumBPFunctor CUDA kernel failed");
    NDArray::registerSpecialUse({gradX, gradY}, {x, y, epsNext});
}

template <typename T>
static void maximumBPFunctor_(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
    NDArray::prepareSpecialUse({gradX, gradY}, {x, y, epsNext});
    
    dim3 launchDims = getLaunchDims("minimax");
    
    if (x->isSameShape(y)) {
        // Element-wise case (same shapes)
        minimumMaximumBPKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
            x->specialBuffer(), x->specialShapeInfo(),
            y->specialBuffer(), y->specialShapeInfo(),
            epsNext->specialBuffer(), epsNext->specialShapeInfo(),
            gradX->specialBuffer(), gradX->specialShapeInfo(),
            gradY->specialBuffer(), gradY->specialShapeInfo(),
            false);  // isMin = false
    } else if (y->isScalar()) {
        // Scalar case
        T scalar = y->e<T>(0);
        
        minimumMaximumScalarBPKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
            x->specialBuffer(), x->specialShapeInfo(),
            scalar, false,  // isMin = false
            epsNext->specialBuffer(), epsNext->specialShapeInfo(),
            gradX->specialBuffer(), gradX->specialShapeInfo());
            
        // Set gradY value based on comparison
        if (*x <= *y) {
            auto tmp = epsNext->reduceNumber(reduce::Sum);
            gradY->assign(&tmp);
        } else {
            gradY->assign(static_cast<T>(0.0));
        }
    } else {
        // Broadcast case - more complex, falls back to CPU for now
        // We'd need to tile arrays to same shape, compute, then reduce along broadcast dims
        
        PointersManager manager(context, "maximumBPFunctor");
        manager.synchronize();
        
        // Move to host, perform calculation, then copy back to device
        x->syncToHost();
        y->syncToHost();
        epsNext->syncToHost();
        
        auto lambdaX = LAMBDA_TTT(_e, _x, _y) { return _x >= _y ? _e : (T)0.; });
        auto lambdaY = LAMBDA_TTT(_e, _x, _y) { return _x <= _y ? _e : (T)0.; });
        
        auto preX = x->dup();
        auto preY = y->dup();
        auto targetShape = epsNext->getShapeAsVector();
        
        preX.tileToShape(targetShape, preX);
        preY.tileToShape(targetShape, preY);
        
        epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaX, &preX);
        epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaY, &preY);
        
        auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
        auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());
        
        if (axisX.size() > 0) {
            auto sum = preX.reduceAlongDimension(reduce::Sum, &axisX);
            gradX->assign(&sum);
        } else {
            gradX->assign(&preX);
        }
        
        if (axisY.size() > 0) {
            auto sum = preY.reduceAlongDimension(reduce::Sum, &axisY);
            gradY->assign(&sum);
        } else {
            gradY->assign(&preY);
        }
        
        gradX->syncToDevice();
        gradY->syncToDevice();
    }
    
    DebugHelper::checkErrorCode(context->getCudaStream(), "maximumBPFunctor CUDA kernel failed");
    NDArray::registerSpecialUse({gradX, gradY}, {x, y, epsNext});
}

void minimumBPFunctor(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
    BUILD_SINGLE_SELECTOR(x->dataType(), minimumBPFunctor_, (context, x, y, epsNext, gradX, gradY), SD_NUMERIC_TYPES);
}

void maximumBPFunctor(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
    BUILD_SINGLE_SELECTOR(x->dataType(), maximumBPFunctor_, (context, x, y, epsNext, gradX, gradY), SD_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void minimumBPFunctor_,
                     (LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY), 
                     SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE(template void maximumBPFunctor_,
                     (LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY), 
                     SD_NUMERIC_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd