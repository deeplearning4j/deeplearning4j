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

#ifndef SAMEDIFF_STRINGS_H
#define SAMEDIFF_STRINGS_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation splits input string into pieces separated by delimiter
         *
         * Input[0] - string to split
         * Input[1] - delimiter
         */
    #if NOT_EXCLUDED(OP_split_string)
        DECLARE_CUSTOM_OP(split_string, 2, 1, true, 0, 0);
    #endif

    }
}


#endif //SAMEDIFF_STRINGS_H
