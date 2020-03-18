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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_split_string)

#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(split_string, 2, 1, true, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto delim = INPUT_VARIABLE(1);

            return Status::OK();
        };

        DECLARE_SHAPE_FN(split_string) {
            auto input = INPUT_VARIABLE(0);
            auto delim = INPUT_VARIABLE(1);

            return SHAPELIST();
        }

        DECLARE_TYPES(split_string) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_STRINGS})
                    ->setAllowedOutputTypes({ALL_STRINGS});
        }
    }
}

#endif