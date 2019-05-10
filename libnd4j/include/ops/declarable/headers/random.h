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

#ifndef LIBND4J_HEADERS_RANDOM_H
#define LIBND4J_HEADERS_RANDOM_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        #if NOT_EXCLUDED(OP_set_seed)
        DECLARE_CUSTOM_OP(set_seed, -2, 1, false, 0, -2);
        #endif

        #if NOT_EXCLUDED(OP_get_seed)
        DECLARE_CUSTOM_OP(get_seed, -2, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_randomuniform)
        DECLARE_CUSTOM_OP(randomuniform, 1, 1, true, 2, 0);
        #endif

        #if NOT_EXCLUDED(OP_random_normal)
        DECLARE_CUSTOM_OP(random_normal, 1, 1, true, 2, 0);
        #endif

        #if NOT_EXCLUDED(OP_random_bernoulli)
        DECLARE_CUSTOM_OP(random_bernoulli, 1, 1, true, 0, 1);
        #endif

        #if NOT_EXCLUDED(OP_random_exponential)
        DECLARE_CUSTOM_OP(random_exponential, 1, 1, true, 1, 0);
        #endif

        DECLARE_CUSTOM_OP(random_crop, 2, 1, false, 0, 0);
    }
}

#endif