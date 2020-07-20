/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

#include <ops/declarable/helpers/threshold.h>
#include <execution/Threads.h>
#include <helpers/threshold.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename T>
            static int32_t thresholdEstimate_(const NDArray &updates, const float threshold) {
                auto N = updates.lengthOf();
                const auto buffer = updates.bufferAsT<T>();

                auto func = PRAGMA_REDUCE_LONG {
                    int64_t cnt = 0;
                    for (auto e = start; e < stop; e++) {
                        auto v = sd::math::nd4j_abs<T>(buffer[e]);
                        if (v >= threshold)
                            cnt++;
                    }

                    return cnt;
                };

                return samediff::Threads::parallel_long(func, LAMBDA_AL { return _old + _new; }, 0, N);
            }

            int32_t thresholdEstimate(const NDArray &updates, const float threshold) {
                BUILD_SINGLE_SELECTOR(updates.dataType(), return thresholdEstimate_, (updates, threshold), FLOAT_TYPES);

                return 0;
            }

            void thresholdEncode(NDArray &updates, NDArray &encoded, float threshold) {
                BUILD_SINGLE_SELECTOR(updates.dataType(), sd::TypeCast::convertToThreshold, (nullptr, updates.buffer(), updates.lengthOf(), encoded.buffer()), FLOAT_TYPES);
            }

            void thresholdDecode(const NDArray &encoded, NDArray &updates) {
                BUILD_SINGLE_SELECTOR(updates.dataType(), sd::TypeCast::convertFromThreshold, (nullptr, encoded.buffer(), updates.lengthOf(), updates.buffer()), FLOAT_TYPES);
            }
        }
    }
}
