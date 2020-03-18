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

#ifndef SAMEDIFF_COMPAT_H
#define SAMEDIFF_COMPAT_H

#include <ops/declarable/headers/common.h>

namespace sd {
    namespace ops {
        /**
         * This operation splits input string into pieces separated by delimiter
         * PLEASE NOTE: This implementation is compatible with TF 1.x
         *
         * Input[0] - string to split
         * Input[1] - delimiter
         *
         * Returns:
         * Output[0] - indices tensor
         * Output[1] - values tensor
         */
    #if NOT_EXCLUDED(OP_compat_string_split)
        DECLARE_CUSTOM_OP(compat_string_split, 2, 2, false, 0, 0);
    #endif

        /**
         * This operation converts TF sparse array representation to dense NDArray
         */
    #if NOT_EXCLUDED(OP_compat_sparse_to_dense)
        DECLARE_CUSTOM_OP(compat_sparse_to_dense, 4, 1, false, 0, 0);
    #endif

    }
}


#endif //SAMEDIFF_COMPAT_H
