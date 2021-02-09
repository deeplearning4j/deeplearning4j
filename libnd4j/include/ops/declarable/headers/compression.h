/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author sgazeos@gmail.com
//
#ifndef SD_HEADERS_COMPRESSION_H
#define SD_HEADERS_COMPRESSION_H

#include <ops/declarable/headers/common.h>

namespace sd {
    namespace ops {
        
        /**
         * encode_bitmap - reinterpret 3D float tensor into uint8_t vector with length N.
         *
         * Input:
         *      0 - 3D float tensor with shape {height, width, channels}
         *
         * Output:
         *      0 - 1D uint8_t tensor with shape {N}
         */
        #if NOT_EXCLUDED(OP_encode_bitmap)
        DECLARE_CUSTOM_OP(encode_bitmap, 1, 3, true, 1, 0);
        #endif

        /**
         *  decode_bitmap - reinterpret uint8_t linear tensor as data to float tensor with shape
         *
         *  Input:
         *      0 - uint8_t vector with length N ( shape {N})
         *
         *  Output:
         *      0 - 3D tensor with shape {height, width, channels}
         *
         */
        #if NOT_EXCLUDED(OP_decode_bitmap)
        DECLARE_CUSTOM_OP(decode_bitmap, 2, 1, true, 0, 0);
        #endif


        DECLARE_CUSTOM_OP(encode_threshold, 2, 1, true, 1, 0);
        DECLARE_CUSTOM_OP(decode_threshold, 2, 1, true, 0, 0);
    }
}

#endif // SD_HEADERS_COMPRESSION_H