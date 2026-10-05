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
#include <graph/RandomGenerator.h>
#include <helpers/RandomLauncher.h>
#include <types/float16.h>
#include <helpers/PointersManager.h>
#include <legacy/NativeOpExecutioner.h>
namespace sd {
void RandomLauncher::applyDropOut(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                  double retainProb, NDArray* z) {
  if (z == nullptr) z = array;

  ExtraArguments arguments({retainProb});
  PointersManager pm(context, "applyDropOut");

  NDArray::prepareSpecialUse({z}, {array});

  NativeOpExecutioner::execRandom(context, random::DropOut, &rng, sd::LegacyTensorArg::fromArray(array),
                                  sd::LegacyTensorArg::fromArray(z), arguments.argumentsAsT(z->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({z}, {array});
}

void RandomLauncher::applyInvertedDropOut(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                          double retainProb, NDArray* z) {
  if (z == nullptr) z = array;

  ExtraArguments arguments({retainProb});
  PointersManager pm(context, "applyInvertedDropOut");

  NDArray::prepareSpecialUse({z}, {array});

  NativeOpExecutioner::execRandom(context, random::DropOutInverted, &rng, sd::LegacyTensorArg::fromArray(array),
                                  sd::LegacyTensorArg::fromArray(z), arguments.argumentsAsT(z->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({z}, {array});
}

void RandomLauncher::applyAlphaDropOut(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                       double retainProb, double alpha, double beta, double alphaPrime, NDArray* z) {
  if (z == nullptr) z = array;

  ExtraArguments arguments({retainProb, alpha, beta, alphaPrime});
  PointersManager pm(context, "applyAlphaDropOut");

  NDArray::prepareSpecialUse({z}, {array});

  NativeOpExecutioner::execRandom(context, random::AlphaDropOut, &rng, sd::LegacyTensorArg::fromArray(array),
                                  sd::LegacyTensorArg::fromArray(z), arguments.argumentsAsT(z->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({z}, {array});
}

void RandomLauncher::fillBernoulli(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                   double prob) {
  ExtraArguments arguments({prob});
  PointersManager pm(context, "fillBernoulli");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::BernoulliDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                 double from, double to) {
  ExtraArguments arguments({from, to});
  PointersManager pm(context, "fillUniform");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::UniformDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillGaussian(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                  double mean, double stdev) {
  ExtraArguments arguments({mean, stdev});
  PointersManager pm(context, "fillGaussian");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::GaussianDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
                                  sd::LegacyTensorArg::fromArray(array), sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillExponential(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                     double lambda) {
  ExtraArguments arguments({lambda});
  PointersManager pm(context, "fillExponential");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::ExponentialDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                 double lambda) {
  ExtraArguments arguments({lambda});
  PointersManager pm(context, "fillPoisson");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::PoissonDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array, double alpha,
                               double beta) {
  ExtraArguments arguments({alpha, beta});
  PointersManager pm(context, "fillGamma");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::GammaDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillLogNormal(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                   double mean, double stdev) {
  ExtraArguments arguments({mean, stdev});
  PointersManager pm(context, "fillLogNormal");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::GaussianDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
                                  sd::LegacyTensorArg::fromArray(array), sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillTruncatedNormal(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                         double mean, double stdev) {
  ExtraArguments arguments({mean, stdev});
  PointersManager pm(context, "fillTruncatedNormal");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(
      context, random::TruncatedNormalDistribution, &rng, sd::LegacyTensorArg::fromArray(array),
      sd::LegacyTensorArg::fromArray(array), sd::LegacyTensorArg::fromArray(array),
      arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}

void RandomLauncher::fillBinomial(LaunchContext* context, graph::RandomGenerator& rng, NDArray* array,
                                  int trials, double prob) {
  ExtraArguments arguments({(double)trials, prob});
  PointersManager pm(context, "fillBinomial");

  NDArray::prepareSpecialUse({array}, {});

  NativeOpExecutioner::execRandom(context, random::BinomialDistributionEx, &rng, sd::LegacyTensorArg::fromArray(array),
                                  sd::LegacyTensorArg::fromArray(array), sd::LegacyTensorArg::fromArray(array),
                                  arguments.argumentsAsT(array->dataType()));
  pm.synchronize();

  NDArray::registerSpecialUse({array}, {});
}
}  // namespace sd
