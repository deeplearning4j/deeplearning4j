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
#ifndef LIBND4J_HEADERS_DTYPE_H
#define LIBND4J_HEADERS_DTYPE_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation casts elements of input array to double data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_double)
        DECLARE_OP(to_double, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to float16 data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_float16)
        DECLARE_OP(to_float16, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to float data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_float32)
        DECLARE_OP(to_float32, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to int32 data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_int32)
        DECLARE_OP(to_int32, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to int64 (aka long long) data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_int64)
        DECLARE_OP(to_int64, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to unsinged int32 data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_uint32)
        DECLARE_OP(to_uint32, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to unsigned int64 (aka unsigned long long) data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
        #if NOT_EXCLUDED(OP_to_uint64)
        DECLARE_OP(to_uint64, 1, 1, true);
        #endif

        /**
         * This operation casts elements of input array to specified data type
         * 
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         * 
         * 
         * Int args:
         * 0: target DataType
         */
        #if NOT_EXCLUDED(OP_cast)
        DECLARE_CUSTOM_OP(cast, 1, 1, false, 0, 1);
        #endif
    }
}

#endif