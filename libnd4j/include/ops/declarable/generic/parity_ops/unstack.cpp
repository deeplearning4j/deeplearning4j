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
#if NOT_EXCLUDED(OP_unstack)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(unstack, 1, -1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);

            auto dim = INT_ARG(0);
            if (dim < 0)
                dim += input->rankOf();


            REQUIRE_TRUE(dim < input->rankOf(), 0, "Unstack dimension should be lower then rank of input %i, but got dimension=%i !", input->rankOf(), dim);
            REQUIRE_TRUE(dim >= 0, 0, "Unstack dimension should be non-negative value, but got %i !", dim);

            std::vector<int> dims;
            for (int e = 0; e < input->rankOf(); e++)
                if (e != dim)
                    dims.emplace_back(e);

            auto tads = input->allTensorsAlongDimension(dims);
            //nd4j_printf("Tad size: %d\n",tads->size());
            for (int e = 0; e < tads->size(); e++) {
                //nd4j_printf("Calling assign at index %d\n",e);
                auto outE = OUTPUT_VARIABLE(e);
                auto tadAtE = tads->at(e);

                outE->assign(tadAtE);

                this->storeResult(block, e, *outE);
            }

            delete tads;

            return Status::OK();
        }
        DECLARE_SYN(unpack, unstack);
        
        DECLARE_SHAPE_FN(unstack) {
            auto inShape = inputShape->at(0);

            auto dim = INT_ARG(0);
            if (dim < 0)
                dim += shape::rank(inShape);

            REQUIRE_TRUE(dim < inShape[0], 0, "UNSTACK op: dimension should be lower then rank of input %i, but got dimension=%i !", inShape[0], dim);
            REQUIRE_TRUE(dim >= 0, 0, "UNSTACK op: dimension should be non-negative value, but got %i !", dim);

            std::vector<int> dims;
            for (int e = 0; e < shape::rank(inShape); e++)
                if (e != dim)
                    dims.emplace_back(e);

            shape::TAD tad(inShape, dims.data(), (int) dims.size());
            tad.createTadOnlyShapeInfo();
            Nd4jLong numTads = shape::length(inShape) / shape::tadLength(inShape, dims.data(), (int) dims.size());
            
            std::vector<Nd4jLong> shape(shape::rank(tad.tadOnlyShapeInfo));
            for (int e = 0; e < shape::rank(tad.tadOnlyShapeInfo); e++)
                shape[e] = shape::shapeOf(tad.tadOnlyShapeInfo)[e];

            // remove leading and trailing 1
            if (inShape[0] == 2 && shape.size() == 2) {
                if (shape[0] == 1) {
                    shape.erase(shape.begin());
                } else if (shape[1] == 1) {
                    shape.erase(shape.end());
                }
            }
            
            auto result = SHAPELIST();
            for (int e = 0; e < numTads; e++) {
                Nd4jLong *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), Nd4jLong);
                if (shape::order(inShape) == 'c')
                    shape::shapeBuffer(shape.size(), shape.data(), newShape);
                else
                    shape::shapeBufferFortran(shape.size(), shape.data(), newShape);
                result->push_back(newShape);
            }

            return result;
        }
    }
}

#endif