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

namespace nd4j {
    // FIXME: implement this

    void RandomLauncher::applyDropOut(nd4j::random::RandomBuffer* buffer, NDArray *array, double retainProb, NDArray* z) {
        if (z == nullptr)
            z = array;

        //array->template applyRandom<randomOps::DropOut<T>>(buffer, nullptr, z, &retainProb);
        NativeOpExcutioner::execRandom(random::DropOut, buffer, z->buffer(), z->shapeInfo(), &retainProb);
    }

    void RandomLauncher::applyInvertedDropOut(nd4j::random::RandomBuffer* buffer, NDArray *array, double retainProb, NDArray* z) {
        if (z == nullptr)
            z = array;

        //array->template applyRandom<randomOps::DropOutInverted<T>>(buffer, nullptr, z, &retainProb);
        NativeOpExcutioner::execRandom(random::DropOutInverted, buffer, z->buffer(), z->shapeInfo(), &retainProb);
    }

    void RandomLauncher::applyAlphaDropOut(nd4j::random::RandomBuffer* buffer, NDArray *array, double retainProb, double alpha, double beta, double alphaPrime, NDArray* z) {
        if (z == nullptr)
            z = array;

        //  FIXME: this isn't portable code :(
        //T args[] = {retainProb, alpha, beta, alphaPrime};

        //array->template applyRandom<randomOps::AlphaDropOut<T>>(buffer, nullptr, z, args);
        NativeOpExcutioner::execRandom(random::AlphaDropOut, buffer, z->buffer(), z->shapeInfo(), &retainProb);
    }

    void RandomLauncher::fillBernoulli(nd4j::random::RandomBuffer* buffer, NDArray* array, double prob) {
        //array->template applyRandom<randomOps::BernoulliDistribution<T>>(buffer, nullptr, array, &prob);
        NativeOpExcutioner::execRandom(random::BernoulliDistribution, buffer, array->buffer(), array->shapeInfo(), &prob);
    }

    void RandomLauncher::fillUniform(nd4j::random::RandomBuffer* buffer, NDArray* array, double from, double to) {
        //T args[] = {from, to};

        //array->template applyRandom<randomOps::UniformDistribution<T>>(buffer, nullptr, nullptr, args);
        NativeOpExcutioner::execRandom(random::UniformDistribution, buffer, array->buffer(), array->shapeInfo(), nullptr);
    }

    void RandomLauncher::fillGaussian(nd4j::random::RandomBuffer* buffer, NDArray* array, double mean, double stdev) {
        //T args[] = {mean, stdev};

        //array->template applyRandom<randomOps::GaussianDistribution<T>>(buffer, array, array, args);
        NativeOpExcutioner::execRandom(random::GaussianDistribution, buffer, array->buffer(), array->shapeInfo(), array->buffer(), array->shapeInfo(), nullptr);
    }

    void RandomLauncher::fillLogNormal(nd4j::random::RandomBuffer* buffer, NDArray* array, double mean, double stdev) {
        //T args[] = {mean, stdev};

        //array->template applyRandom<randomOps::LogNormalDistribution<T>>(buffer, array, array, args);
        NativeOpExcutioner::execRandom(random::GaussianDistribution, buffer, array->buffer(), array->shapeInfo(), array->buffer(), array->shapeInfo(), nullptr);
    }

    void RandomLauncher::fillTruncatedNormal(nd4j::random::RandomBuffer* buffer, NDArray* array, double mean, double stdev) {
        //T args[] = {mean, stdev};

        //array->template applyRandom<randomOps::TruncatedNormalDistribution<T>>(buffer, array, array, args);
        NativeOpExcutioner::execRandom(random::GaussianDistribution, buffer, array->buffer(), array->shapeInfo(), array->buffer(), array->shapeInfo(), nullptr);
    }

    void RandomLauncher::fillBinomial(nd4j::random::RandomBuffer* buffer, NDArray* array, int trials, double prob) {
        //T args[] = {(T) trials, prob};

        //array->template applyRandom<randomOps::BinomialDistributionEx<T>>(buffer, array, array, args);
        NativeOpExcutioner::execRandom(random::BinomialDistributionEx, buffer, array->buffer(), array->shapeInfo(), array->buffer(), array->shapeInfo(), nullptr);
    }

}