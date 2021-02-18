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
//  @author @cpuheater
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_is_numeric_tensor)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/compare_elem.h>

namespace sd {
    namespace ops {
        BOOLEAN_OP_IMPL(is_numeric_tensor, 1, true) {

            auto input = INPUT_VARIABLE(0);

            return input->isR() || input->isZ() ? ND4J_STATUS_TRUE : ND4J_STATUS_FALSE;
        }

        DECLARE_TYPES(is_numeric_tensor) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, DataType::ANY)
                    ->setAllowedOutputTypes(0, DataType::BOOL);
        }
    }
}

#endif