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

#include <NDArray.h>
#include <helpers/helper_random.h>
#include <graph/RandomGenerator.h>
#include <execution/LaunchContext.h>

namespace nd4j {
    class ND4J_EXPORT RandomLauncher {
    public:
        static void applyDropOut(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z = nullptr);
        static void applyInvertedDropOut(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z = nullptr);
        static void applyAlphaDropOut(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray *array, double retainProb, double alpha, double beta, double alphaPrime, NDArray* z = nullptr);

        static void fillUniform(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray* array, double from, double to);

        static void fillGaussian(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev);

        static void fillExponential(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray* array, double lambda);

        static void fillLogNormal(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev);

        static void fillTruncatedNormal(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev);

        static void fillBinomial(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray* array, int trials, double prob);

        static void fillBernoulli(nd4j::LaunchContext *context, nd4j::graph::RandomGenerator& rng, NDArray* array, double prob);
    };
}