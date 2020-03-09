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
#include <execution/Threads.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template<typename T>
            static void _compare_elem(NDArray *input, bool isStrictlyIncreasing, bool& output) {
                auto length = shape::length(input->getShapeInfo());

                int elementsPerThread = length / ELEMENT_THRESHOLD;
                int num_threads = sd::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = sd::math::nd4j_min<int>(num_threads, omp_get_max_threads());
                Nd4jLong sumt = 0;

                if(isStrictlyIncreasing) {
                    //PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(+:sum)
                    auto func = PRAGMA_REDUCE_LONG {
                        Nd4jLong sum = 0;
                        for (auto i = start; i < stop; i++) {
                            auto val0 = input->t<T>(i);
                            auto val1 = input->t<T>(i + 1);
                            sum += val0 >= val1 ? -1 : 0;
                        }
                        return sum;
                    };
                    sumt = sd::Threads::parallel_long(func, LAMBDA_SUML, 0, length - 1);
                } else {
                    //PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(+:sum)
                    auto func = PRAGMA_REDUCE_LONG {
                        Nd4jLong sum = 0;
                        for (auto i = start; i < stop; i++) {
                            auto val0 = input->t<T>(i);
                            auto val1 = input->t<T>(i + 1);
                            sum += val0 > val1 ? -1 : 0;
                        }

                        return sum;
                    };
                    sumt = sd::Threads::parallel_long(func, LAMBDA_SUML, 0, length - 1);
                }

                //nd4j_printf("Sum: %lld\n", sumt)

                output = (sumt > -1);

            }

            void compare_elem(sd::LaunchContext * context, NDArray *input, bool isStrictlyIncreasing, bool& output) {
                auto xType = input->dataType();

                BUILD_SINGLE_SELECTOR(xType, _compare_elem, (input, isStrictlyIncreasing, output), LIBND4J_TYPES);
            }


            BUILD_SINGLE_TEMPLATE(template void _compare_elem, (NDArray *A, bool isStrictlyIncreasing, bool& output);, LIBND4J_TYPES);
        }
    }
}
