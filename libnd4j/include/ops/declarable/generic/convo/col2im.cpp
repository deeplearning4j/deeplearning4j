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
// Created by raver119 on 17.10.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_col2im)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/col2im.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(col2im, 1, 1, false, 0, 9) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->rankOf() == 6, 0, "col2im input should be 6D, but got %i instead", x->rankOf());
            REQUIRE_TRUE(z->rankOf() == 4, 0, "col2im output should be 4D, but got %i instead", z->rankOf());

            int strideY = INT_ARG(0);
            int strideX = INT_ARG(1);
            int padHeight = INT_ARG(2);
            int padWidth = INT_ARG(3);
            int imgHeight = INT_ARG(4);
            int imgWidth = INT_ARG(5);
            int dY = INT_ARG(6);			//Dilation in height/y dimension
            int dX = INT_ARG(7);			//Dilation in width/x dimension

            LaunchContext ctx;
            nd4j::ops::helpers::col2im(ctx, *z, *x, strideY, strideX, padHeight, padWidth, imgHeight, imgWidth, dY, dX);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(col2im) {
            auto inShape = inputShape->at(0);

            int bS = shape::shapeOf(inShape)[0];
            int iD = shape::shapeOf(inShape)[1];

            int sY = INT_ARG(0);
            int sX = INT_ARG(1);
            int pY = INT_ARG(2);
            int pX = INT_ARG(3);
            int inY = INT_ARG(4);
            int inX = INT_ARG(5);
            int dY = INT_ARG(6);			//Dilation, height/y dimension
            int dX = INT_ARG(7);			//Dilation, width/x dimension
            bool isSameMode = INT_ARG(8) > 0;

            Nd4jLong* zShape;
            ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(4), Nd4jLong);

            zShape[0] = 4;
            zShape[1] = bS;
            zShape[2] = iD;
            zShape[3] = inY;
            zShape[4] = inX;

            zShape[shape::shapeInfoLength(zShape) - 3] = 0;
            zShape[shape::shapeInfoLength(zShape) - 2] = 1;
            zShape[shape::shapeInfoLength(zShape) - 1] = 99;

            shape::updateStrides(zShape, 'c');

            return SHAPELIST(zShape);
        }
    }
}

#endif