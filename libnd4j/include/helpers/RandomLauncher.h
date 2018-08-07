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

namespace nd4j {
    template <typename T>
    class RandomLauncher {
    public:
        static void applyDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, NDArray<T>* z = nullptr);
        static void applyInvertedDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, NDArray<T>* z = nullptr);
        static void applyAlphaDropOut(nd4j::random::RandomBuffer* buffer, NDArray<T> *array, T retainProb, T alpha, T beta, T alphaPrime, NDArray<T>* z = nullptr);

        static void fillUniform(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T from, T to);
        static void fillGaussian(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev);
        static void fillLogNormal(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev);
        static void fillTruncatedNormal(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T mean, T stdev);
        static void fillBinomial(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, int trials, T prob);
        static void fillBernoulli(nd4j::random::RandomBuffer* buffer, NDArray<T>* array, T prob);
    };
}