/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by raver119 on 02.11.2017.
//

#include <op_boilerplate.h>
//#if NOT_EXCLUDED(OP_slice)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(slice, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            int x_rank = input->rankOf();

            std::vector<int> begin;
            std::vector<int> end;

            if (block.width() == 3) {
                auto b = INPUT_VARIABLE(1);
                auto e = INPUT_VARIABLE(2);

                begin = b->template asVectorT<int>();
                end = e->template asVectorT<int>();
            } else {
                REQUIRE_TRUE(block.numI() >= x_rank * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead", x_rank * 2, block.numI());

                ShapeUtils::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
                ShapeUtils::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
            }

            REQUIRE_TRUE(begin.size() == x_rank, 0, "begin array should have length of [%i] but got [%i] instead", x_rank, begin.size());
            REQUIRE_TRUE(end.size() == x_rank, 0, "end array should have length of [%i] but got [%i] instead", x_rank, end.size());

            std::vector<Nd4jLong> indices(2 * x_rank);
            for (int e = 0; e < x_rank; e++) {
                int size = end[e];
                int start = begin[e];

                REQUIRE_TRUE(start >= 0, 0, "Slice: start index should not be negative");

                REQUIRE_TRUE(start < input->sizeAt(e), 0, "Index %i is invalid for dimension %i with size %i.", start, e, input->shapeInfo()[e + 1]);
                if (size == -1){
                    size = input->sizeAt(e) - start;
                }
                REQUIRE_TRUE(size > 0, 0, "Slice: interval for dimension %i is less then 1");
                REQUIRE_TRUE(start + size <= input->sizeAt(e), 0, "Slice: interval [%i, %i] is out of bounds for dimension %i with size %i", start, start + size, e, input->sizeAt(e));
                
                indices[2*e]   = start;
                indices[2*e+1] = start + size;
            }
            auto sub = (*input)(indices, true);
            output->assign(sub);

            STORE_RESULT(output);

            return Status::OK();
        }

        DECLARE_TYPES(slice) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setSameMode(true);
        }

        DECLARE_SHAPE_FN(slice) {
            auto inShape = inputShape->at(0);
            auto x_rank = shape::rank(inShape);

            std::vector<int> begin;
            std::vector<int> end;

            if (block.width() == 3) {
                auto b = INPUT_VARIABLE(1);
                auto e = INPUT_VARIABLE(2);

                begin = b->template asVectorT<int>();
                end = e->template asVectorT<int>();
            } else {
                REQUIRE_TRUE(block.numI() >= x_rank * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead", x_rank * 2, block.numI());

                ShapeUtils::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
                ShapeUtils::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
            }

            REQUIRE_TRUE(begin.size() == x_rank, 0, "begin array should have length of [%i] but got [%i] instead", x_rank, begin.size());
            REQUIRE_TRUE(end.size() == x_rank, 0, "end array should have length of [%i] but got [%i] instead", x_rank, end.size());
            
            std::vector<Nd4jLong> shape;
            for (int e = 0; e < x_rank; e++) {
                auto stop = end[e];
                auto start = begin[e];

                if(stop == -1){
                    stop = inShape[e+1] - start;
                }

                shape.emplace_back(stop);
            }

            Nd4jLong *newShape = nd4j::ShapeBuilders::createShapeInfo(ArrayOptions::dataType(inShape), 'c', shape, block.getWorkspace());
            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(slice_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }


        CUSTOM_OP_IMPL(slice_bp, 2, 1, false, 0, -1) {
            auto input = INPUT_VARIABLE(0);
            auto epsNext = block.width() == 4 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(1);

            auto output = OUTPUT_VARIABLE(0);
            output->assign(0.);
            int x_rank = input->rankOf();

            std::vector<int> begin;
            std::vector<int> end;

            if (block.width() == 4) {
                auto b = INPUT_VARIABLE(1);
                auto e = INPUT_VARIABLE(2);

                begin = b->template asVectorT<int>();
                end = e->template asVectorT<int>();
            } else {
                REQUIRE_TRUE(block.numI() >= x_rank * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead", x_rank * 2, block.numI());

                ShapeUtils::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
                ShapeUtils::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
            }

            REQUIRE_TRUE(begin.size() == x_rank, 0, "begin array should have length of [%i] but got [%i] instead", x_rank, begin.size());
            REQUIRE_TRUE(end.size() == x_rank, 0, "end array should have length of [%i] but got [%i] instead", x_rank, end.size());

            std::vector<Nd4jLong> indices(2 * x_rank);
            for (int e = 0; e < x_rank; e++) {
                int stop = end[e];
                int start = begin[e];

                REQUIRE_TRUE(stop > 0, 0, "Slice: interval for dimension %i is less then 1", e);
                
                indices[2*e]     = start;
                indices[2*e + 1] = start + stop;
            }
            auto sub = (*output)(indices, true);
            sub.assign(epsNext);            

            return Status::OK();
        }

        DECLARE_SHAPE_FN(slice_bp) {
            auto inShape = inputShape->at(0);
            Nd4jLong *newShape;
            COPY_SHAPE(inShape, newShape);

            return SHAPELIST(newShape);
        }
    }
}

//#endif