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

#include <ops/declarable/helpers/compare_elem.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template<typename T>
            void compare_elem(NDArray<T> *input, bool isStrictlyIncreasing, bool& output) {
                auto length = shape::length(input->getShapeInfo());

                int elementsPerThread = length / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                T sum = (T) 0.f;

                if(isStrictlyIncreasing) {
#pragma omp parallel reduction(sumT:sum) num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < length - 1; i++) {
                        auto val0 = (*input)(i);
                        auto val1 = (*input)(i + 1);
                        sum += val0 >= val1 ? (T) -1.f : (T) 0.0f;
                    }
                } else {
#pragma omp parallel reduction(sumT:sum) num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < length - 1; i++) {
                        auto val0 = (*input)(i);
                        auto val1 = (*input)(i + 1);
                        sum += val0 > val1 ? (T) -1.f : (T) 0.0f;
                    }
                }

                output = (sum > -1);

            }
            template void compare_elem<float>(NDArray<float> *A, bool isStrictlyIncreasing, bool& output);
            template void compare_elem<float16>(NDArray<float16> *A, bool isStrictlyIncreasing, bool& output);
            template void compare_elem<double>(NDArray<double> *A, bool isStrictlyIncreasing, bool& output);
            template void compare_elem<int>(NDArray<int> *A, bool isStrictlyIncreasing, bool& output);
            template void compare_elem<Nd4jLong>(NDArray<Nd4jLong> *A, bool isStrictlyIncreasing, bool& output);
        }
    }
}
