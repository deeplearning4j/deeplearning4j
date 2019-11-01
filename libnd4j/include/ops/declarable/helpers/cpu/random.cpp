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
#include <vector>
#include <memory>
#include <graph/Context.h>
namespace nd4j {
namespace ops {
namespace helpers {

    void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {
//        auto shift = output->lengthOf() / lambda->lengthOf();
//        auto step = lambda->lengthOf();
//        PRAGMA_OMP_PARALLEL_FOR
//        for (auto k = 0; k < shift; k++) {
//            auto pos = k * step;
//            auto u = rng.relativeT<float>(k, 0., 1.);
//            for (auto e = 0; e < lambda->lengthOf(); e++) {
//                auto p = math::nd4j_exp<float, float>(-lambda->e<float>(e));
//                auto s = p;
//                auto x = 0.f;
//                while (u > s) {
//                    x += 1.f;
//                    p *= lambda->e<float>(e) / x;
//                    s += p;
//                }
//                output->p(pos + e, x);
//            }
//        }
    }

    /*
     * algorithm Poisson generator based upon the inversion by sequential search:[48]:505
    init:
         Let x ← 0, p ← e−λ, s ← p.
         Generate uniform random number u in [0,1].
    while u > s do:
         x ← x + 1.
         p ← p * λ / x.
         s ← s + p.
    return x.
     * */
    void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        auto shift = output->lengthOf() / lambda->lengthOf();
        auto step = lambda->lengthOf();
        PRAGMA_OMP_PARALLEL_FOR
        for (auto k = 0; k < shift; k++) {
            auto pos = k * step;
            auto u = rng.relativeT<float>(k, 0., 1.);
            for (auto e = 0; e < step; e++) {
                auto p = math::nd4j_exp<float, float>(-lambda->e<float>(e));
                auto s = p;
                auto x = 0.f;
                while (u > s) {
                    x += 1.f;
                    p *= lambda->e<float>(e) / x;
                    s += p;
                }
                output->p(pos + e, x);
            }
        }
    }

}
}
}