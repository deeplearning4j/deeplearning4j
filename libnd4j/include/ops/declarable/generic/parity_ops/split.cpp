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

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_split)

#include <ops/declarable/headers/parity_ops.h>
#include <array>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(split, 1, -1, false, 0, 1) {
        NDArray *input = nullptr;
        int num_splits = INT_ARG(0);

        // axis is 0 by default
        int axis = 0;

        if (block.width() == 1) {
            input = INPUT_VARIABLE(0);
        } else {
            auto a = INPUT_VARIABLE(0);
            auto b = INPUT_VARIABLE(1);

            if (a->isScalar()) {
                // axis goes first
                axis = a->e<int>(0);
                input = b;
            } else if (b->isScalar()) {
                axis = b->e<int>(0);
                input = a;
            }
        }
		
		//Edge case: splitting empty array (mainly for TF import compatibility) -> return N empty arrays
		if(input->isEmpty()){
			for( int i=0; i< num_splits; i++ ){
				REQUIRE_TRUE(OUTPUT_VARIABLE(i)->isEmpty(), 0, "Split: When input array is empty, all output arrays must be empty");
			}
			//No op
			return Status::OK();
		}

        if (block.numI() == 2)
            axis = INT_ARG(1);

        if(axis < 0) axis += input->rankOf();

        REQUIRE_TRUE(input->sizeAt(axis) % num_splits == 0, 0, "Split: num_splits has wrong value, remainder of division should be 0, but it's %i", input->sizeAt(axis) % num_splits);

        int pos = 0;
        int split = input->sizeAt(axis) / num_splits;
        std::vector<Nd4jLong> indices(2 * input->rankOf());
        
        for (int e = 0; e < num_splits; e++) {
            
            auto out = OUTPUT_VARIABLE(e);
            
            for (int d = 0; d < input->rankOf(); d++) {
                if (d == axis) {
                    indices[2*d]     = pos;
                    indices[2*d + 1] = pos + split; 
                }
                else 
                    indices[2*d] = indices[2*d + 1] = 0;
            }

            auto sub = (*input)(indices, true);
            
            out->assign(sub);

            pos += split;
        }

        return Status::OK();
    }

    DECLARE_TYPES(split) {
        getOpDescriptor()
                ->setAllowedInputTypes({ALL_INTS, ALL_FLOATS})
                ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(split) {
        int num_splits = INT_ARG(0);
        Nd4jLong *input = nullptr;
        nd4j::DataType dataType;

        // axis is 0 by default
        int axis = 0;

		int inputVar = 0;
        if (inputShape->size() == 1) {
            input = inputShape->at(0);
            dataType = ArrayOptions::dataType(input);
        } else {
            auto shape0 = inputShape->at(0);
            auto shape1 = inputShape->at(1);

            if (shape::isScalar(shape0)) {
                input = shape1;
                auto _a = INPUT_VARIABLE(0);
                axis = _a->e<int>(0);
                dataType = ArrayOptions::dataType(shape1);
				inputVar = 1;
            } else if (shape::isScalar(shape1)) {
                input = shape0;
                auto _a = INPUT_VARIABLE(1);
                axis = _a->e<int>(0);
                dataType = ArrayOptions::dataType(shape0);
				inputVar = 0;
            }
        }
		
		auto shapes = SHAPELIST();
		
		//Edge case: splitting empty array (mainly for TF import compatibility) -> return N empty arrays
		if(INPUT_VARIABLE(inputVar)->isEmpty()){
			for (int e = 0; e < num_splits; e++) {
                auto empty = ConstantShapeHelper::getInstance()->scalarShapeInfo(dataType);
				shapes->push_back(empty);
			}
			return shapes;
		}

        if (block.numI() == 2)
            axis = INT_ARG(1);

        if (axis < 0)
            axis += shape::rank(input);

        std::vector<Nd4jLong> shape(shape::rank(input));

        for (int e = 0; e < shape::rank(input); e++)
            if (e == axis)
                shape[e] = shape::sizeAt(input, e) / num_splits;
            else 
                shape[e] = shape::sizeAt(input, e);
        
        for (int e = 0; e < num_splits; e++) {
            auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(dataType, shape::order(input), shape);
            shapes->push_back(newShape);
        }

        return shapes;
    }
}
}

#endif