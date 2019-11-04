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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/random.h>
//#include <NativeOps.h>
#include <vector>
#include <memory>
#include <graph/Context.h>
#include <helpers/RandomLauncher.h>
#include <ShapeUtils.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    /*
     * fillGammaKernel - fill up output with gamma distributed values
     *
     *  uList - uniformly distributed values set
     *  uLength - length of uList
     *  alpha - alpha param
     *  beta - beta param
     *  output - distributed output.
     * */
    template <typename T>
    static __global__ void fillGammaKernel(T* uList, Nd4jLong uLength, T* alpha, Nd4jLong* alphaShape,
            T* beta, Nd4jLong* betaShape, T* output, Nd4jLong* outputShape) {
        // fill up
        __shared__ Nd4jLong aLength;
        if (threadIdx.x == 0) {
            aLength = shape::length(alphaShape);
        }
        __syncthreads();

        for (auto k = blockIdx.x; k < (int)uLength; k += gridDim.x) {
            auto pos = k * aLength;
            auto u = uList[k]; // this is a vector
            for (auto e = threadIdx.x; e < (int)aLength; e += blockDim.x) {
                auto aIndex = shape::getIndexOffset(e, alphaShape);
                auto bIndex = betaShape?shape::getIndexOffset(e, betaShape):-1LL;
                auto betaV = T(beta != nullptr ? beta[bIndex] * u : u);
                auto zIndex = shape::getIndexOffset(e + pos, outputShape);

                output[zIndex] = math::nd4j_igamma<T, T, T>(alpha[aIndex], betaV);
            }
        }
    }

    template <typename T>
    static void fillRandomGamma_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {
        // To fill up output need to broadcast alpha and beta to the same shape and in
        Nd4jLong* broadcasted = nullptr;
        if (beta != nullptr)
            ShapeUtils::evalBroadcastShapeInfo(*alpha, *beta, true, broadcasted, context->getWorkspace());
        else
            broadcasted = alpha->shapeInfo();
        auto step = shape::length(broadcasted);
        auto shift = output->lengthOf() / step;

        auto copyAlpha = alpha;
        auto copyBeta = beta;
        if (beta != nullptr) {
            NDArray alphaBroadcasted(broadcasted, alpha->dataType(), true, context);
            NDArray betaBroadcasted(broadcasted, beta->dataType(), true, context);

            copyAlpha = (alphaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), alpha));
            copyBeta = (betaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), beta));
            copyAlpha->tickWriteDevice(); copyBeta->tickWriteDevice();
        }

        auto stream = context->getCudaStream();
        NDArray uniform = NDArrayFactory::create<T>('c', {shift}, context);
        uniform.syncToDevice();
        // fill up uniform with given length
        RandomLauncher::fillUniform(context, rng, &uniform, 0., 1.);

        fillGammaKernel<T><<<128, 128, 256, *stream>>>(uniform.dataBuffer()->specialAsT<T>(), shift,
                copyAlpha->dataBuffer()->specialAsT<T>(), copyAlpha->specialShapeInfo(),
                beta?copyBeta->dataBuffer()->specialAsT<T>():(T*)nullptr,
                beta?copyBeta->specialShapeInfo():(Nd4jLong*)nullptr,
                output->dataBuffer()->specialAsT<T>(), output->specialShapeInfo());

        if (beta != nullptr) {
            delete copyAlpha;
            delete copyBeta;
            //delete broadcasted;
        }

    }

    void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {
        if (beta)
            NDArray::prepareSpecialUse({output}, {alpha, beta});
        else
            NDArray::prepareSpecialUse({output}, {alpha});
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomGamma_, (context, rng, alpha, beta, output), FLOAT_NATIVE);
        if (beta)
            NDArray::registerSpecialUse({output}, {alpha, beta});
        else
            NDArray::prepareSpecialUse({output}, {alpha});
    }
    BUILD_SINGLE_TEMPLATE(template void fillRandomGamma_, (LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output), FLOAT_NATIVE);


    /*
     * algorithm Poisson generator based upon the inversion by sequential search
     *
    init:
         Let x ← 0, p ← e−λ, s ← p.
         using uniformly random sequence U (u in U) distributed at [0, 1].
    while u > s do:
         x ← x + 1.
         p ← p * λ / x.
         s ← s + p.
    return x.
     * */
    template <typename T>
    static __global__ void fillPoissonKernel(T* uList, Nd4jLong uLength, T* lambda, Nd4jLong* lambdaShape, T* output,
            Nd4jLong* outputShape) {

        __shared__ Nd4jLong step;

        if (threadIdx.x == 0) {
            step = shape::length(lambdaShape);
        }
        __syncthreads();

        for (auto k = blockIdx.x; k < (int)uLength; k += gridDim.x) {
            auto pos = k * step;
            auto u = uList[k];
            for (auto e = threadIdx.x; e < step; e += blockDim.x) {
                auto p = math::nd4j_exp<T,T>(-lambda[e]);
                auto s = p;
                auto x = T(0.f);
                auto lIndex = shape::getIndexOffset(e, lambdaShape);
                auto zIndex = shape::getIndexOffset(e + pos, outputShape);
                while (u > s) {
                    x += T(1.);
                    p *= lambda[lIndex] / x;
                    s += p;
                }
                output[zIndex] = x;
            }
        }
    }

    template <typename T>
    static void fillRandomPoisson_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        auto shift = output->lengthOf() / lambda->lengthOf();
        NDArray uniform('c', {shift}, output->dataType());
        auto stream = context->getCudaStream();
        // fill up uniform with given length
        RandomLauncher::fillUniform(context, rng, &uniform, 0., 1.);
        fillPoissonKernel<T><<<128, 256, 128, *stream>>>(uniform.dataBuffer()->specialAsT<T>(), uniform.lengthOf(),
                lambda->dataBuffer()->specialAsT<T>(), lambda->specialShapeInfo(),
                output->dataBuffer()->specialAsT<T>(), output->specialShapeInfo());
    }

    void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {lambda});
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomPoisson_, (context, rng, lambda, output), FLOAT_NATIVE);
        NDArray::registerSpecialUse({output}, {lambda});
    }

    BUILD_SINGLE_TEMPLATE(template void fillRandomPoisson_, (LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output), FLOAT_NATIVE);
}
}
}