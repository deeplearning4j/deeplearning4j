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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_cast)

#include <array/DataTypeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(cast, 1, 1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            // TODO: once we add support for multiple dtypes - uncommend this
            /*
            int it = INT_ARG(0);
            DataType newType = DataTypeUtils::fromInt(it);

            input->cast(output, newType);
            */

            if (!block.isInplace())
                output->assign(input);
            
            STORE_RESULT(output);
            return Status::OK();
        }
        DECLARE_SYN(Cast, cast);

        DECLARE_SHAPE_FN(cast) {
            auto inShape = inputShape->at(0);

            auto it = INT_ARG(0);
            DataType newType = DataTypeUtils::fromInt(it);

            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);
            ArrayOptions::setDataType(newShape, newType);

            return SHAPELIST(newShape);
        }
    }
}

#endif