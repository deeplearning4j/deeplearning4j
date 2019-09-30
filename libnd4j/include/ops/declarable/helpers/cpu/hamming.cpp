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
// @author raver119@gmail.com
//

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hamming.h>
#include <execution/Threads.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            static Nd4jLong hamming_distance(unsigned long long x, unsigned long long y) {
                Nd4jLong dist = 0;

                for (unsigned long long val = x ^ y; val > 0; val /= 2) {
                    if (val & 1)
                        dist++;
                }
                return dist;
            }


            template <typename X, typename Z>
            static void _hamming(NDArray &x, NDArray &y, NDArray &z) {
                auto xEws = x.ews();
                auto yEws = y.ews();

                auto xBuffer = x.bufferAsT<X>();
                auto yBuffer = y.bufferAsT<X>();

                Nd4jLong distance = 0;
                auto lengthOf = x.lengthOf();
                const int maxThreads = nd4j::math::nd4j_min<int>(256, omp_get_max_threads());
                Nd4jLong intermediate[256];

                // nullify temp values
                for (int e = 0; e < maxThreads; e++)
                    intermediate[e] = 0;

                if (xEws == 1 && yEws == 1 && x.ordering() == y.ordering()) {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e += increment) {
                            auto _x = static_cast<unsigned long long>(xBuffer[e]);
                            auto _y = static_cast<unsigned long long>(yBuffer[e]);

                            intermediate[thread_id] += hamming_distance(_x, _y);
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, lengthOf);
                } else if (xEws > 1 && yEws > 1 && x.ordering() == y.ordering()) {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e += increment) {
                            auto _x = static_cast<unsigned long long>(xBuffer[e * xEws]);
                            auto _y = static_cast<unsigned long long>(yBuffer[e * yEws]);

                            intermediate[thread_id] += hamming_distance(_x, _y);
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, lengthOf);
                } else {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e += increment) {
                            auto _x = static_cast<unsigned long long>(x.e<Nd4jLong>(e));
                            auto _y = static_cast<unsigned long long>(y.e<Nd4jLong>(e));

                            intermediate[thread_id] += hamming_distance(_x, _y);
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, lengthOf);
                }

                // accumulate intermediate variables into output array
                for (int e = 0; e < maxThreads; e++)
                    distance += intermediate[e];

                z.p(0, distance);
            }

            void hamming(LaunchContext *context, NDArray &x, NDArray &y, NDArray &output) {
                BUILD_DOUBLE_SELECTOR(x.dataType(), output.dataType(), _hamming, (x, y, output), INTEGER_TYPES, INDEXING_TYPES);
            }
        }
    }
}