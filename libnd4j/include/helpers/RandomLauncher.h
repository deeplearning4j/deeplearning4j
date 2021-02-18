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
//  @author raver119@gmail.com
//

#include <array/NDArray.h>
#include <helpers/helper_random.h>
#include <graph/RandomGenerator.h>
#include <execution/LaunchContext.h>

namespace sd {
    class ND4J_EXPORT RandomLauncher {
    public:
        static void applyDropOut(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z = nullptr);
        static void applyInvertedDropOut(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z = nullptr);
        static void applyAlphaDropOut(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray *array, double retainProb, double alpha, double beta, double alphaPrime, NDArray* z = nullptr);

        static void fillUniform(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double from, double to);

        static void fillGaussian(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev);

        static void fillExponential(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double lambda);

        static void fillLogNormal(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev);

        static void fillTruncatedNormal(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev);

        static void fillBinomial(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, int trials, double prob);

        static void fillBernoulli(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double prob);
    };
}