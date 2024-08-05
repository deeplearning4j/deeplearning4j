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
#include <execution/LaunchContext.h>
#include <graph/RandomGenerator.h>
#include <helpers/helper_random.h>

namespace sd {
class SD_LIB_EXPORT RandomLauncher {
 public:
  static void applyDropOut(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                           double retainProb, NDArray* z = nullptr);
  static void applyInvertedDropOut(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                   double retainProb, NDArray* z = nullptr);
  static void applyAlphaDropOut(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                double retainProb, double alpha, double beta, double alphaPrime, NDArray* z = nullptr);

  static void fillUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array, double from,
                          double to);

  static void fillGaussian(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array, double mean,
                           double stdev);

  static void fillExponential(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                              double lambda);

  static void fillLogNormal(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array, double mean,
                            double stdev);

  static void fillTruncatedNormal(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                  double mean, double stdev);

  static void fillBinomial(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array, int trials,
                           double prob);

  static void fillBernoulli(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array, double prob);
};
}  // namespace sd
