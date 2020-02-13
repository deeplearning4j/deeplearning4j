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
// @author sgazeos@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lin_space)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

    CUSTOM_OP_IMPL(lin_space, 3, 1, false, 0, 0) {
        auto output = OUTPUT_VARIABLE(0);
        auto start = INPUT_VARIABLE(0);
        auto finish = INPUT_VARIABLE(1);
        auto numOfElements = INPUT_VARIABLE(2);

        if (numOfElements->e<Nd4jLong>(0) == 1) {
            output->assign(start);
            return Status::OK();
        }
    
        output->linspace(start->e<double>(0), (finish->e<double>(0) - start->e<double>(0)) / (numOfElements->e<Nd4jLong>(0) - 1.));
        return Status::OK();
    }
    
    DECLARE_SHAPE_FN(lin_space) {
        auto dataType = ArrayOptions::dataType(inputShape->at(0));
        Nd4jLong steps = INPUT_VARIABLE(2)->e<Nd4jLong>(0);
        return SHAPELIST(ConstantShapeHelper::getInstance()->vectorShapeInfo(steps, dataType));
    }


    DECLARE_TYPES(lin_space) {
        getOpDescriptor()
                ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
                ->setAllowedInputTypes(1, {ALL_FLOATS, ALL_INTS})
                ->setAllowedInputTypes(2, {ALL_INTS})
                ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
    }
}
}

#endif