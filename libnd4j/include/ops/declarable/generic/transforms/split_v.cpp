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
#if NOT_EXCLUDED(OP_split_v)

#include <ops/declarable/headers/parity_ops.h>

namespace sd {
namespace ops {
    CUSTOM_OP_IMPL(split_v, 2, -1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);
        auto sizes = INPUT_VARIABLE(1);

        int axis = 0;

        if (block.getIArguments()->size() > 0) {
            axis = INT_ARG(0);
        } else if (block.width() > 2){
            auto _a = INPUT_VARIABLE(2);
            axis = _a->e<int>(0);
        } 

        if (axis < 0)
            axis += input->rankOf();

        std::vector<int> dims = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});

        int pos = 0;
        std::vector<Nd4jLong> indices(2 * input->rankOf());
        
        for (Nd4jLong e = 0; e < sizes->lengthOf(); e++) {
            int c_size = sizes->e<int>(e);
            
            for (int d = 0; d < input->rankOf(); d++) {
                if (d == axis)                          
                    indices[2*d + 1] = (indices[2*d] = pos) + c_size;                
                else 
                    indices[2*d] = indices[2*d + 1] = 0;
            }

            auto output = OUTPUT_VARIABLE(e);
            REQUIRE_TRUE(output->dataType() == input->dataType(), 0, "SplitV: all outputs must have same data type as input");

            auto sub = (*input)(indices);

            output->assign(sub);

            pos += c_size;            
        }

        //delete tads;
        return Status::OK();
    }

    DECLARE_TYPES(split_v) {
        getOpDescriptor()
                ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
                ->setAllowedInputTypes(1, {ALL_INTS})
                ->setAllowedInputTypes(2, {ALL_INTS})
                ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(split_v) {
        auto input = inputShape->at(0);
        //auto sizes = inputShape->at(1);

        auto shapeList = SHAPELIST();
        int rank = shape::rank(input);

        // 0 is just default axis
        int axis = 0;

        if (block.getIArguments()->size() > 0)
            axis = INT_ARG(0);
        else if (block.width() > 2) {
            auto _a = INPUT_VARIABLE(2);
            axis = _a->e<int>(0);
        }

        if (axis < 0)
            axis += shape::rank(input);
        
        // this op assumes we have sizes defined
        auto sizes = INPUT_VARIABLE(1);
        
        auto length = sizes->lengthOf();
        int pos = 0;
        for (Nd4jLong e = 0; e < length; e++) {
            int c_size = sizes->e<int>(e);
            

            std::vector<Nd4jLong> shape(rank);

            for (int d = 0; d < rank; d++) {
                if (d != axis)
                    shape[d] = shape::sizeAt(input, d);
                else 
                    shape[d] = c_size;
            }

            auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(input), shape::order(input), shape);
            shapeList->push_back(newShape);
        }

        return shapeList;
    }
}
}

#endif