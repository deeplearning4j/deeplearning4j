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
#if NOT_EXCLUDED(OP_dilation2d)

#include <ops/declarable/headers/convo.h>
#include <ops/declarable/helpers/dilation2d.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dilation2d, 2, 1, false, 0, 1) {
        auto input = INPUT_VARIABLE(0);
        auto weights = INPUT_VARIABLE(1);

        auto output = OUTPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 4, 0, "Dilation2D: input should be 4D");
        REQUIRE_TRUE(weights->rankOf() == 3, 0, "Dilation2D: weights should be 3D");

        const int batch_size = input->sizeAt(0);
        const int depth = input->sizeAt(3);
        const bool isSameShape = INT_ARG(0) == 1;

        REQUIRE_TRUE(input->sizeAt(3) == weights->sizeAt(2), 0, "Dilation2D: number of input channels doesn't match number of channels in weights: %i vs %i", input->sizeAt(3), weights->sizeAt(2));

        std::vector<int> strides(4);
        std::vector<int> rates(4);

        if (block.width() > 2) {
            REQUIRE_TRUE(block.width() >= 4, 0, "Dilation2D: number of input arrays should be 4 at least");


            auto r = INPUT_VARIABLE(2);
            auto s = INPUT_VARIABLE(3);

            strides = s->template asVectorT<int>();
            rates = r->template asVectorT<int>();
        } else {
            REQUIRE_TRUE(block.numI() >= 9, 0, "Dilation2D: number of Int arguments should be 9 at least");

            int e = 1;
            for (int cnt = 0;cnt < 4; cnt++)
                rates[cnt] = INT_ARG(e++);


            for (int cnt = 0; cnt < 4; cnt++)
                strides[cnt] = INT_ARG(e++);
        }


        int stride_rows = 0, stride_cols = 0;
        int rate_rows = 0, rate_cols = 0;
        int pad_top = 0, pad_left = 0;
        int out_rows = 0, out_cols = 0;

        helpers::_dilation_hw(input->shapeInfo(), weights->shapeInfo(), strides, rates, isSameShape, &stride_rows, &stride_cols, &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows, &out_cols);


        REQUIRE_TRUE(out_rows > 0 && out_cols > 0, 0, "Dilation2D: outY and outX should have positive values, but got [%i, %i] instead", out_rows, out_cols);

        helpers::dilation2d(input, weights, output, stride_rows, stride_cols, rate_rows, rate_cols, pad_top, pad_left);

        return Status::OK();
    }

    DECLARE_SHAPE_FN(dilation2d) {
        auto input = inputShape->at(0);
        auto weights = inputShape->at(1);

        const int batch_size = shape::sizeAt(input, 0);
        const int depth = shape::sizeAt(input, 3);
        const bool isSameShape = INT_ARG(0) == 1;

        std::vector<int> strides(4);
        std::vector<int> rates(4);

        Nd4jLong *newShape;

        if (block.width() > 2) {
            auto r = INPUT_VARIABLE(2);
            auto s = INPUT_VARIABLE(3);


            strides = s->template asVectorT<int>();
            rates = r->template asVectorT<int>();
        } else {
            if (block.numI() < 9) {
                newShape = ShapeBuilders::createScalarShapeInfo(block.dataType(), block.workspace());
                return SHAPELIST(newShape);
            }
                
            int e = 1;
            for (int cnt = 0;cnt < 4; cnt++)
                rates[cnt] = INT_ARG(e++);

            for (int cnt = 0; cnt < 4; cnt++)
                strides[cnt] = INT_ARG(e++);
        }

        int stride_rows = 0, stride_cols = 0;
        int rate_rows = 0, rate_cols = 0;
        int pad_top = 0, pad_left = 0;
        int out_rows = 0, out_cols = 0;

        helpers::_dilation_hw(input, weights, strides, rates, isSameShape, &stride_rows, &stride_cols, &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows, &out_cols);

        std::array<Nd4jLong, 4> shape = {{batch_size, out_rows, out_cols, depth}};
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), Nd4jLong);
        shape::shapeBuffer(4, block.dataType(), shape.data(), newShape);
        ArrayOptions::setDataType(newShape, ArrayOptions::dataType(input));
        return SHAPELIST(newShape);
    }
}
}

#endif