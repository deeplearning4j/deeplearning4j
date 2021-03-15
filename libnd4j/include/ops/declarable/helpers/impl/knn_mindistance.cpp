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

#include <ops/declarable/helpers/knn.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename T>
            void mindistance_(const void* vinput, const void *vlow, const void *vhigh, int32_t length, void *vout) {
                auto input = reinterpret_cast<const T*>(vinput);
                auto low = reinterpret_cast<const T*>(vlow);
                auto high = reinterpret_cast<const T*>(vhigh);
                auto output = reinterpret_cast<T*>(vout);

                T res = 0.0f;
                T po = 2.f;
                T o = 1.f;

#pragma omp simd reduction(sumT:res)
                for (auto e = 0; e < length; e++) {
                    T p = input[e];
                    T l = low[e];
                    T h = high[e];
                    if (!(l <= p || h <= p)) {
                        if (p < l)
                            res += sd::math::nd4j_pow<T, T, T>((p - o), po);
                        else
                            res += sd::math::nd4j_pow<T, T, T>((p - h), po);
                    }
                }

                output[0] = sd::math::nd4j_pow<T, T, T>(res, (T) 0.5f);
            }

            void knn_mindistance(const NDArray &input, const NDArray &lowest, const NDArray &highest, NDArray &output) {
                NDArray::preparePrimaryUse({&output}, {&input, &lowest, &highest});

                BUILD_SINGLE_SELECTOR(input.dataType(), mindistance_, (input.buffer(), lowest.buffer(), highest.buffer(), input.lengthOf(), output.buffer()), FLOAT_TYPES);

                NDArray::registerPrimaryUse({&output}, {&input, &lowest, &highest});
            }
        }
    }
}