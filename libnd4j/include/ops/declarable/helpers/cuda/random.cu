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

namespace nd4j {
namespace ops {
namespace helpers {


    void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {

    }

    /*
 * algorithm Poisson generator based upon the inversion by sequential search
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
                while (u > s) {
                    x += T(1.);
                    p *= lambda[e] / x;
                    s += p;
                }
                output[pos + e] = x;
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