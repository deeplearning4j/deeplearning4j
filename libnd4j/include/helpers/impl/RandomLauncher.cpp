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

#include <types/float16.h>
#include <system/dll.h>
#include <helpers/RandomLauncher.h>
#include <graph/RandomGenerator.h>
//#include <ops/declarable/CustomOperations.h>
#include <helpers/PointersManager.h>

namespace sd {
    // FIXME: implement this

    void RandomLauncher::applyDropOut(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z) {
        if (z == nullptr)
            z = array;

        ExtraArguments arguments({retainProb});
        PointersManager pm(context, "applyDropOut");

        NativeOpExecutioner::execRandom(context, random::DropOut, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), arguments.argumentsAsT(z->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::applyInvertedDropOut(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z) {
        if (z == nullptr)
            z = array;

        ExtraArguments arguments({retainProb});
        PointersManager pm(context, "applyInvertedDropOut");

        NativeOpExecutioner::execRandom(context, random::DropOutInverted, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), arguments.argumentsAsT(z->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::applyAlphaDropOut(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray *array, double retainProb, double alpha, double beta, double alphaPrime, NDArray* z) {
        if (z == nullptr)
            z = array;

        ExtraArguments arguments({retainProb, alpha, beta, alphaPrime});
        PointersManager pm(context, "applyAlphaDropOut");

        NativeOpExecutioner::execRandom(context, random::AlphaDropOut, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), arguments.argumentsAsT(z->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::fillBernoulli(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double prob) {
        ExtraArguments arguments({prob});
        PointersManager pm(context, "fillBernoulli");

        NativeOpExecutioner::execRandom(context, random::BernoulliDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::fillUniform(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double from, double to) {
        ExtraArguments arguments({from, to});
        PointersManager pm(context, "fillUniform");

        NativeOpExecutioner::execRandom(context, random::UniformDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::fillGaussian(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev) {
        ExtraArguments arguments({mean, stdev});
        PointersManager pm(context, "fillGaussian");

        NativeOpExecutioner::execRandom(context, random::GaussianDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::fillExponential(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double lambda) {
        ExtraArguments arguments({lambda});
        PointersManager pm(context, "fillExponential");

        NativeOpExecutioner::execRandom(context, random::ExponentialDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::fillLogNormal(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev) {
        ExtraArguments arguments({mean, stdev});
        PointersManager pm(context, "fillLogNormal");

        NativeOpExecutioner::execRandom(context, random::GaussianDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::fillTruncatedNormal(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev) {
        ExtraArguments arguments({mean, stdev});
        PointersManager pm(context, "fillTruncatedNormal");

        NativeOpExecutioner::execRandom(context, random::TruncatedNormalDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
        pm.synchronize();
    }

    void RandomLauncher::fillBinomial(sd::LaunchContext *context, sd::graph::RandomGenerator& rng, NDArray* array, int trials, double prob) {
        ExtraArguments arguments({(double) trials, prob});
        PointersManager pm(context, "fillBinomial");

        NativeOpExecutioner::execRandom(context, random::BinomialDistributionEx, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
        pm.synchronize();
    }
}
