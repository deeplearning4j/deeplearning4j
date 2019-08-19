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

#ifndef DEV_TESTS_SHIFT_H
#define DEV_TESTS_SHIFT_H

#include <op_boilerplate.h>
#include <types/types.h>
#include <NDArray.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            void rshift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift);

            void shift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift);

            void cyclic_rshift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift);

            void cyclic_shift_bits(LaunchContext* launchContext, NDArray &x, NDArray &z, uint32_t shift);
        }
    }
}

#endif //DEV_TESTS_SHIFT_H
