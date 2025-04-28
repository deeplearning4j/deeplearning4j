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
//  @author sgazeos@gmail.com
//

#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/choose.h>
#include <ops/ops.h>
#include <helpers/PointersManager.h>
#include <exceptions/cuda_exception.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void processChooseConditionCuda(int mode, 
                                                const void* vx, const LongType* xShapeInfo,
                                                const void* vy, const LongType* yShapeInfo, 
                                                void* vz, const LongType* zShapeInfo,
                                                void* vnum, const LongType* numShapeInfo,
                                                LongType length) {
    const T* x = reinterpret_cast<const T*>(vx);
    const T* y = reinterpret_cast<const T*>(vy);
    T* z = reinterpret_cast<T*>(vz);
    int* numResults = reinterpret_cast<int*>(vnum);
    
    __shared__ LongType xEWS, yEWS, zEWS;
    __shared__ LongType yLen;
    __shared__ bool yIsScalar;
    
    if (threadIdx.x == 0) {
        xEWS = shape::elementWiseStride(xShapeInfo);
        yEWS = shape::elementWiseStride(yShapeInfo);
        zEWS = shape::elementWiseStride(zShapeInfo);
        yLen = shape::length(yShapeInfo);
        yIsScalar = shape::isScalar(yShapeInfo);
    }
    __syncthreads();
    
    // Using atomics for numResults counter
    __shared__ int counter;
    if (threadIdx.x == 0) {
        counter = 0;
    }
    __syncthreads();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
        T d1 = x[i * xEWS];
        T d2 = yIsScalar ? y[0] : (i < yLen ? y[i * yEWS] : y[0]);
        
        T input[3] = {d2, (T)SD_EPSILON, (T)mode};
        T res = simdOps::MatchCondition<T, T>::op(d1, input);
        
        if (res > static_cast<T>(0)) {
            int idx = atomicAdd(&counter, 1);
            if (z != nullptr) {
                z[idx] = d1;
            }
        }
    }
    
    // Final step: Update the numResults output
    if (threadIdx.x == 0) {
        *numResults = counter;
    }
}

template <typename T>
static void processConditionCuda(LaunchContext* context, int mode, NDArray* arg, NDArray* comp, NDArray* output,
                                NDArray* numResult, NDArray& compScalar) {
    
    const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (arg->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    
    PointersManager manager(context, "chooseFunctor");
    
    NDArray* compTensor = comp;
    if (comp == nullptr) {
        compTensor = &compScalar;
    }
    
    processChooseConditionCuda<T><<<blocksPerGrid, threadsPerBlock, 1024, *context->getCudaStream()>>>(
        mode,
        arg->specialBuffer(), arg->specialShapeInfo(),
        compTensor->specialBuffer(), compTensor->specialShapeInfo(),
        output == nullptr ? nullptr : output->specialBuffer(), 
        output == nullptr ? nullptr : output->specialShapeInfo(),
        numResult == nullptr ? nullptr : numResult->specialBuffer(), 
        numResult == nullptr ? nullptr : numResult->specialShapeInfo(),
        arg->lengthOf());
    
    manager.synchronize();
    sd::DebugHelper::checkErrorCode(context->getCudaStream(), "chooseFunctor CUDA failed");
}

void chooseFunctorArray(LaunchContext* context, NDArray* arg, NDArray* comp, int mode, NDArray* result,
                        NDArray* numResults) {
    NDArray::prepareSpecialUse({result, numResults}, {arg, comp});
    
    if (arg->isScalar() || comp->isScalar()) {
        if (arg->isScalar()) {
            BUILD_SINGLE_SELECTOR(comp->dataType(), processConditionCuda, (context, mode, comp, nullptr, result, numResults, *arg), SD_FLOAT_TYPES);
        } else {
            BUILD_SINGLE_SELECTOR(arg->dataType(), processConditionCuda, (context, mode, arg, nullptr, result, numResults, *comp), SD_FLOAT_TYPES);
        }
    } else {
        auto zero = NDArrayFactory::create<float>(0);
        BUILD_SINGLE_SELECTOR(arg->dataType(), processConditionCuda, (context, mode, arg, comp, result, numResults, zero), SD_FLOAT_TYPES);
    }
    
    NDArray::registerSpecialUse({result, numResults}, {arg, comp});
}

void chooseFunctorScalar(LaunchContext* context, NDArray* arg, double scalar, int mode, NDArray* result,
                         NDArray* numResults) {
    auto scalarA = NDArrayFactory::create(scalar);
    NDArray::prepareSpecialUse({result, numResults}, {arg});
    
    BUILD_SINGLE_SELECTOR(arg->dataType(), processConditionCuda, (context, mode, arg, nullptr, result, numResults, scalarA), SD_FLOAT_TYPES);
    
    NDArray::registerSpecialUse({result, numResults}, {arg});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd