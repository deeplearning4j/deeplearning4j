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
#include <dll.h>
#include <helpers/RandomLauncher.h>
#include <graph/RandomGenerator.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    // FIXME: implement this

    void RandomLauncher::applyDropOut(nd4j::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z) {
        if (z == nullptr)
            z = array;

        ExtraArguments arguments({retainProb});

        NativeOpExecutioner::execRandom(nullptr, random::DropOut, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), arguments.argumentsAsT(z->dataType()));
    }

    void RandomLauncher::applyInvertedDropOut(nd4j::graph::RandomGenerator& rng, NDArray *array, double retainProb, NDArray* z) {
        if (z == nullptr)
            z = array;

        ExtraArguments arguments({retainProb});

        NativeOpExecutioner::execRandom(nullptr, random::DropOutInverted, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), arguments.argumentsAsT(z->dataType()));
    }

    void RandomLauncher::applyAlphaDropOut(nd4j::graph::RandomGenerator& rng, NDArray *array, double retainProb, double alpha, double beta, double alphaPrime, NDArray* z) {
        if (z == nullptr)
            z = array;

        ExtraArguments arguments({retainProb, alpha, beta, alphaPrime});

        NativeOpExecutioner::execRandom(nullptr, random::AlphaDropOut, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), arguments.argumentsAsT(z->dataType()));
    }

    void RandomLauncher::fillBernoulli(nd4j::graph::RandomGenerator& rng, NDArray* array, double prob) {
        ExtraArguments arguments({prob});

        NativeOpExecutioner::execRandom(nullptr, random::BernoulliDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    }

    void RandomLauncher::fillUniform(nd4j::graph::RandomGenerator& rng, NDArray* array, double from, double to) {
        ExtraArguments arguments({from, to});

        NativeOpExecutioner::execRandom(nullptr, random::UniformDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    }

    void RandomLauncher::fillGaussian(nd4j::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev) {
        ExtraArguments arguments({mean, stdev});

        NativeOpExecutioner::execRandom(nullptr, random::GaussianDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    }

    void RandomLauncher::fillExponential(nd4j::graph::RandomGenerator& rng, NDArray* array, double lambda) {
        ExtraArguments arguments({lambda});

        NativeOpExecutioner::execRandom(nullptr, random::ExponentialDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    }

    void RandomLauncher::fillLogNormal(nd4j::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev) {
        ExtraArguments arguments({mean, stdev});

        NativeOpExecutioner::execRandom(nullptr, random::GaussianDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    }

    void RandomLauncher::fillTruncatedNormal(nd4j::graph::RandomGenerator& rng, NDArray* array, double mean, double stdev) {
        ExtraArguments arguments({mean, stdev});

        NativeOpExecutioner::execRandom(nullptr, random::TruncatedNormalDistribution, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    }

    void RandomLauncher::fillBinomial(nd4j::graph::RandomGenerator& rng, NDArray* array, int trials, double prob) {
        ExtraArguments arguments({(double) trials, prob});

        NativeOpExecutioner::execRandom(nullptr, random::BinomialDistributionEx, &rng, array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), array->buffer(), array->shapeInfo(), array->specialBuffer(), array->specialShapeInfo(), arguments.argumentsAsT(array->dataType()));
    }
}
