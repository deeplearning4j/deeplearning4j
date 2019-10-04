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
//  @author George A. Shulinok <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_draw_bounding_boxes)

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>
namespace nd4j {
    namespace ops {
        OP_IMPL(draw_bounding_boxes, 3, 1, true) {

            auto image = INPUT_VARIABLE(0);
            auto boxes = INPUT_VARIABLE(1);
            auto colors = INPUT_VARIABLE(2);
            
            return ND4J_STATUS_OK;
        }

        DECLARE_TYPES(draw_bounding_boxes) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {HALF, FLOAT32})// TF allows HALF and FLOAT32 only
                    ->setAllowedInputTypes(1, {FLOAT32}) // as TF
                    ->setAllowedInputTypes(2, {FLOAT32}) // as TF
                    ->setAllowedOutputTypes({HALF, FLOAT32}); // TF allows HALF and FLOAT32 only
        }
    }
}

#endif
