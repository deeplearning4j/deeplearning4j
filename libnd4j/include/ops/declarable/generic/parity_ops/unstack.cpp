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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_unstack)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ConstantTadHelper.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(unstack, 1, -1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);

            auto dim = INT_ARG(0);
            if (dim < 0)
                dim += input->rankOf();


            REQUIRE_TRUE(dim < input->rankOf(), 0, "Unstack dimension should be lower then rank of input %i, but got dimension=%i !", input->rankOf(), dim);
            REQUIRE_TRUE(dim >= 0, 0, "Unstack dimension should be non-negative value, but got %i !", dim);

            if(input->isEmpty())
                return Status::OK();

            std::vector<int> dims;
            for (int e = 0; e < input->rankOf(); e++)
                if (e != dim)
                    dims.emplace_back(e);
            if (dims.size() == 0 && input->rankOf() == 1) { // split vector into lenthOf scalars
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    auto outE = OUTPUT_VARIABLE(e);
                    outE->assign(input->e(e));
                }
            }

            auto tads = input->allTensorsAlongDimension(dims);
            //nd4j_printf("Tad size: %d\n",tads.size());
            for (int e = 0; e < tads.size(); e++) {
                //nd4j_printf("Calling assign at index %d\n",e);
                auto outE = OUTPUT_VARIABLE(e);
                auto tadAtE = tads.at(e);

                outE->assign(tadAtE);

                this->storeResult(block, e, *outE);
            }

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

            if(ArrayOptions::arrayType(inShape) == ArrayType::EMPTY) {
                if(shape::shapeOf(inShape)[dim] == 0)
                    return SHAPELIST();
                const Nd4jLong numTads = shape::shapeOf(inShape)[dim];
                std::vector<Nd4jLong> outShape;
                for(uint i = 0; i < shape::rank(inShape); ++i)
                    if(i != dim)
                        outShape.push_back(shape::shapeOf(inShape)[i]);

                auto result = SHAPELIST();
                for(uint i = 0; i < numTads; ++i)
                    result->push_back(ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), outShape));
                return result;
            }

            std::vector<int> dims;
            for (int e = 0; e < shape::rank(inShape); e++)
                if (e != dim)
                    dims.emplace_back(e);
            if (dims.size() == 0 && shape::rank(inShape) == 1) { // split vector into lenthOf scalars
                //
                auto result = SHAPELIST();
                for (Nd4jLong e = 0; e < shape::length(inShape); e++)
                    result->push_back(ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(inShape)));
                return result;
            }

            auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(inShape, dims);
            auto numTads = tadPack.numberOfTads();

            std::vector<Nd4jLong> shape(shape::rank(tadPack.primaryShapeInfo()));
            for (int e = 0; e < shape::rank(tadPack.primaryShapeInfo()); e++)
                shape[e] = shape::shapeOf(tadPack.primaryShapeInfo())[e];

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
                auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), shape);
                result->push_back(newShape);
            }
            return result;
        }

        DECLARE_TYPES(unstack) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
                    ->setSameMode(true);
        }
    }
}

#endif