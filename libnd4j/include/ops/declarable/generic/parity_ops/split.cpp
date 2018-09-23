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
                axis = a->getScalar<int>(0);
                input = b;
            } else if (b->isScalar()) {
                axis = b->getScalar<int>(0);
                input = a;
            }
        }

        if (block.numI() == 2)
            axis = INT_ARG(1);

        if(axis < 0) axis += input->rankOf();

        REQUIRE_TRUE(input->sizeAt(axis) % num_splits == 0, 0, "Split: num_splits has wrong value, remainder of division should be 0, but it's %i", input->sizeAt(axis) % num_splits);

        int pos = 0;
        int split = input->sizeAt(axis) / num_splits;
        for (int e = 0; e < num_splits; e++) {
            auto out = OUTPUT_VARIABLE(e);

            IndicesList indices;
            for (int d = 0; d < input->rankOf(); d++) {
                if (d == axis)
                    indices.push_back(NDIndex::interval(pos, pos + split));
                else 
                    indices.push_back(NDIndex::all());
            }

            auto sub = input->subarray(indices);
            
            out->assign(sub);

            delete sub;

            pos += split;
        }



        return Status::OK();
    }

    DECLARE_SHAPE_FN(split) {
        int num_splits = INT_ARG(0);
        Nd4jLong *input = nullptr;

        // axis is 0 by default
        int axis = 0;

        if (inputShape->size() == 1)
            input = inputShape->at(0);
        else {
            auto shape0 = inputShape->at(0);
            auto shape1 = inputShape->at(1);

            if (shape::isScalar(shape0)) {
                input = shape1;
                auto _a = INPUT_VARIABLE(0);
                axis = _a->getScalar<int>(0);
            } else if (shape::isScalar(shape1)) {
                input = shape0;
                auto _a = INPUT_VARIABLE(1);
                axis = _a->getScalar<int>(0);
            }
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

        auto shapes = SHAPELIST();

        for (int e = 0; e < num_splits; e++) {
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(input), Nd4jLong);

            if (shape::order(input) == 'c')
                shape::shapeBuffer(shape.size(), block.dataType(), shape.data(), newShape);
            else
                shape::shapeBufferFortran(shape.size(), block.dataType(), shape.data(), newShape);

            shapes->push_back(newShape);
        }

        return shapes;
    }
}
}

#endif