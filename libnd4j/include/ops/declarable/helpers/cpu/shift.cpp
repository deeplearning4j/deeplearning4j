/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#include <ops/declarable/helpers/shift.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void rshift_bits_(LaunchContext* launchContext, NDArray &input, NDArray &output, uint32_t shift) {
                auto lambda = LAMBDA_T(x, shift) {
                    return x >> shift;
                };

                input.applyLambda<T>(lambda, &output);
            }

            void rshift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift) {
                BUILD_SINGLE_SELECTOR(x.dataType(), rshift_bits_, (launchContext, x, z, shift), INTEGER_TYPES);
            }

            template <typename T>
            void shift_bits_(LaunchContext* launchContext, NDArray &input, NDArray &output, uint32_t shift) {
                auto lambda = LAMBDA_T(x, shift) {
                    return x << shift;
                };

                input.applyLambda<T>(lambda, &output);
            }

            void shift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift) {
                BUILD_SINGLE_SELECTOR(x.dataType(), shift_bits_, (launchContext, x, z, shift), INTEGER_TYPES);
            }

            template <typename T>
            void cyclic_rshift_bits_(LaunchContext* launchContext, NDArray &input, NDArray &output, uint32_t shift) {
                auto step = (sizeof(T) * 8) - shift;
                auto lambda = LAMBDA_T(x, shift, step) {
                    return x >> shift | x << step;
                };

                input.applyLambda<T>(lambda, &output);
            }

            void cyclic_rshift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift) {
                BUILD_SINGLE_SELECTOR(x.dataType(), cyclic_rshift_bits_, (launchContext, x, z, shift), INTEGER_TYPES);
            }

            template <typename T>
            void cyclic_shift_bits_(LaunchContext* launchContext, NDArray &input, NDArray &output, uint32_t shift) {
                auto step = (sizeof(T) * 8) - shift;
                auto lambda = LAMBDA_T(x, shift, step) {
                    return x << shift | x >> step;
                };

                input.applyLambda<T>(lambda, &output);
            }

            void cyclic_shift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift) {
                BUILD_SINGLE_SELECTOR(x.dataType(), cyclic_shift_bits_, (launchContext, x, z, shift), INTEGER_TYPES);
            }
        }
    }
}