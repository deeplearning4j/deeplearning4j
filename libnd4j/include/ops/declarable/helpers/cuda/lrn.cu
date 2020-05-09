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

//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/lrn.h>
#include <graph/Status.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
namespace ops {
namespace helpers {

    template <typename T>
    static _CUDA_G void lrnKernel(void *vx, Nd4jLong  const*xTadShapeInfo, Nd4jLong  const*xTadOffsets, void *vz, Nd4jLong  const*zTadShapeInfo, Nd4jLong  const*zTadOffsets, Nd4jLong numTads, Nd4jLong tadLength, int depth, double bias, double alpha, double beta) {
        extern __shared__ char sharedChar[];
        T* shared = reinterpret_cast<T*>(sharedChar);

        auto xEws = shape::elementWiseStride(xTadShapeInfo);
        auto zEws = shape::elementWiseStride(zTadShapeInfo);

        auto xOrder = shape::order(xTadShapeInfo);
        auto zOrder = shape::order(zTadShapeInfo);

        const T tbias  = static_cast<T>(bias);
        const T tbeta  = static_cast<T>(beta);
        const T talpha = static_cast<T>(alpha);

        // one block of threads processes 1 example within batch
        for (uint i = blockIdx.x; i < numTads; i += gridDim.x) {
            auto x = reinterpret_cast<T*>(vx) + xTadOffsets[i];
            auto z = reinterpret_cast<T*>(vz) + zTadOffsets[i];

            // load everything into shared memory, so we'll operate on shared memory from now on
            shared[threadIdx.x] = x[threadIdx.x * xEws];
            __syncthreads();

            const uint begin = sd::math::nd4j_max<int>(0, threadIdx.x - depth);
            const uint last  = depth + threadIdx.x + 1;
            const uint end   = sd::math::nd4j_min<int>(last, tadLength);

            T prev = 0.;
            for (int s = begin; s < end; s++)
                prev = prev + shared[s] * shared[s];

            z[threadIdx.x * zEws] = shared[threadIdx.x] / sd::math::nd4j_pow<T, T, T>(tbias + alpha * prev, tbeta);
        }
    }

    template <typename X, typename Z>
    static _CUDA_G void lrnBPKernel(void const* vx, Nd4jLong const* xTadShapeInfo, Nd4jLong const* xTadOffsets, void *vz, Nd4jLong const* zTadShapeInfo, Nd4jLong const* zTadOffsets, Nd4jLong numTads, Nd4jLong tadLength, int depth, double bias, double alpha, double beta) {
        extern __shared__ char sharedChar[];
        X* sharedX = reinterpret_cast<X*>(sharedChar);
        Z* sharedY = reinterpret_cast<Z*>(sharedX + blockDim.x);

        auto xEws = shape::elementWiseStride(xTadShapeInfo);
        auto zEws = shape::elementWiseStride(zTadShapeInfo);

        auto xOrder = shape::order(xTadShapeInfo);
        auto zOrder = shape::order(zTadShapeInfo);

        const Z tbias  = static_cast<Z>(bias);
        const Z tbeta  = static_cast<Z>(beta);
        const Z talpha = static_cast<Z>(alpha);
        const Z coeff  = talpha * tbeta;



        for (uint i = blockIdx.x; i < numTads; i += gridDim.x) {
            auto x = reinterpret_cast<X const*>(vx) + xTadOffsets[i];
            auto z = reinterpret_cast<Z*>(vz) + zTadOffsets[i];

            const uint begin = sd::math::nd4j_max<int>(0, threadIdx.x - depth);
            const uint last  = depth + threadIdx.x + 1;
            const uint end   = sd::math::nd4j_min<int>(last, tadLength);

            // load everything into shared memory
            sharedX[threadIdx.x] = x[threadIdx.x * xEws];
            sharedY[threadIdx.x] = 0.f;
            __syncthreads();

            // we're operating in shared memory
            for (int s = begin; s < end; s++)
                sharedY[threadIdx.x] = sharedY[threadIdx.x] + sharedX[s] * sharedX[s];
            __syncthreads();

            Z factor[1024];
            Z init = tbias + talpha * sharedY[threadIdx.x];

            Z prev = 0.f;
            for (uint s = begin; s < end; ++s) {
                factor[s] = sd::math::nd4j_pow<Z, Z, Z>(tbias + talpha * sharedY[s], -tbeta - 1);
                prev = prev + sharedX[s] * factor[s];
            }

            z[threadIdx.x * zEws] = factor[threadIdx.x] * init - 2 * sharedX[threadIdx.x] * coeff * prev;
        }
    }


    template <typename X, typename Z>
    static void lrnBP_(sd::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta) {
        auto rank = input.rankOf();
        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input.shapeInfo(), {rank - 1});
        auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(gradI.shapeInfo(), {rank - 1});

        const auto tadLength = shape::length(packX.primaryShapeInfo());
        const int numBlocks = sd::math::nd4j_min<Nd4jLong>(1024, packX.numberOfTads());
        const int numThreads = tadLength;

        if (tadLength > 1024 || tadLength < 1)
            throw std::runtime_error("LRN: tadLength > 1024 isn't implemented yet");

        lrnBPKernel<X, Z><<<numBlocks, numThreads, numThreads * sizeof(X) + numThreads * sizeof(Z) + 1024, *block.launchContext()->getCudaStream()>>>(input.specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), gradI.specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), packX.numberOfTads(),  tadLength, depth, bias, alpha, beta);

        gradI.tickWriteDevice();
        gradI *= gradO;
    }

    void lrnBP(sd::graph::Context& block, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta) {
        input.syncToDevice();
        gradO.syncToDevice();

        BUILD_DOUBLE_SELECTOR(input.dataType(), gradO.dataType(), lrnBP_, (block, input, gradO, gradI, depth, bias, alpha, beta), FLOAT_TYPES, FLOAT_TYPES);

        gradI.tickWriteDevice();
    }

    template <typename T>
    static void lrnFunctor_(sd::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {
        auto rank = input->rankOf();
        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input->shapeInfo(), {rank - 1});
        auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), {rank - 1});

        const auto tadLength = shape::length(packX.primaryShapeInfo());
        const int numBlocks = sd::math::nd4j_min<Nd4jLong>(1024, packX.numberOfTads());
        const int numThreads = tadLength;

        if (tadLength > 1024 || tadLength < 1)
            throw std::runtime_error("LRN: tadLength > 1024 isn't implemented yet");

        lrnKernel<T><<<numBlocks, numThreads, numThreads * sizeof(T), *block.launchContext()->getCudaStream()>>>(input->specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), output->specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), packX.numberOfTads(), tadLength, depth, bias, alpha, beta);
    }

    int lrnFunctor(sd::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {
        input->syncToDevice();

        BUILD_SINGLE_SELECTOR(input->dataType(), lrnFunctor_, (block, input, output, depth, bias, alpha, beta), FLOAT_TYPES);

        output->tickWriteDevice();

        return Status::OK();
    }
}
}
}
