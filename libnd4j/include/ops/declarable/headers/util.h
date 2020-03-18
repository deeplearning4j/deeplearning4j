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

#ifndef LIBND4J_UTILS_H
#define LIBND4J_UTILS_H

#include <ops/declarable/headers/common.h>

namespace sd {
    namespace ops {
        /**
         * This operation prints out NDArray content, either on host or device.
         */
    #if NOT_EXCLUDED(OP_print_variable)
        DECLARE_CUSTOM_OP(print_variable, 1, 1, true, 0, 0);
    #endif

        /**
         * This operation prints out affinity & locality status of given NDArray
         */
    #if NOT_EXCLUDED(OP_print_affinity)
        DECLARE_CUSTOM_OP(print_affinity, 1, 1, true, 0, 0);
    #endif
    }
}

#endif //LIBND4J_UTILS_H
