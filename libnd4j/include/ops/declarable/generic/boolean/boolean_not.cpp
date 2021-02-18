/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver on 6/6/2018.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_boolean_not)

#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        OP_IMPL(boolean_not, 1, 1,true) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            x->applyTransform(transform::Not, *z);

            return Status::OK();
        }

        DECLARE_TYPES(boolean_not) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, DataType::BOOL)
                    ->setAllowedOutputTypes(0, DataType::BOOL);
        }
    }
}

#endif