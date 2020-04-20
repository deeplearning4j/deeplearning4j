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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lin_space)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

    CUSTOM_OP_IMPL(lin_space, 0, 1, false, 0, 0) {

        auto output = OUTPUT_VARIABLE(0);

        const int nInputs = block.width();
        bool bInputs = (3 == nInputs || 3 == block.numI() || (2 == block.numT() && block.numI() > 0));

        REQUIRE_TRUE(bInputs, 0, "lin_space OP: Have to be supplied correct inputs, input size or T_ARG size have to be equal 3, but got inputs - %i, T_ARGS - %i!", nInputs, block.numT());
        
        auto start = (nInputs > 0) ?  INPUT_VARIABLE(0)->e<double>(0) : static_cast<double>(T_ARG(0));
        auto finish = (nInputs > 0) ? INPUT_VARIABLE(1)->e<double>(0) : static_cast<double>(T_ARG(1));
        auto numOfElements = (nInputs > 0) ? INPUT_VARIABLE(2)->e<Nd4jLong>(0) : static_cast<Nd4jLong>(I_ARG(0));

        if (numOfElements == 1) {
            output->assign(start);
            return Status::OK();
        }
    
        output->linspace(start, (finish - start) / ( numOfElements - 1.0 ));
        return Status::OK();
    }
    
    DECLARE_SHAPE_FN(lin_space) {

        const int nInputs = block.width();
        bool bInputs = (3 == nInputs || 3 == block.numI() || (2 == block.numT() && block.numI() > 0));
        REQUIRE_TRUE(bInputs, 0, "lin_space OP: Have to be supplied correct inputs, input size or T_ARG size have to be equal 3, but got inputs - %i, T_ARGS - %i!", nInputs, block.numT() );


        auto dataType = (nInputs > 0) ? ArrayOptions::dataType(inputShape->at(0)) : ( block.numD() > 0 ? static_cast<DataType>(D_ARG(0)) : DataType::FLOAT32) ;
        Nd4jLong steps = (nInputs > 0) ? INPUT_VARIABLE(2)->e<Nd4jLong>(0) : static_cast<Nd4jLong>(I_ARG(0));

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