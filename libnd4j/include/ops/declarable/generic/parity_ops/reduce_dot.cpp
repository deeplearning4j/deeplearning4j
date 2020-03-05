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
// Created by george@skymind.io on 6/1/2018.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_dot_bp)

////////////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_dot_bp, 3, 2, false, 0, 0) {

    auto x     = INPUT_VARIABLE(0);
    auto y     = INPUT_VARIABLE(1);
    auto gradO = INPUT_VARIABLE(2);

    auto gradX = OUTPUT_VARIABLE(0);
    auto gradY = OUTPUT_VARIABLE(1);

    // L(x,y) = SUM(x_i * y_i)
    // dL/dx_i = y_i

    REQUIRE_TRUE(x->isSameShape(y), 0, "REDUCE_DOT_BP OP: both input arrays x and y should have same shapes, but got %s and %s correspondingly", ShapeUtils::shapeAsString(x).c_str(), ShapeUtils::shapeAsString(y).c_str());

    if (gradO->lengthOf() == 1) { // scalar of reduced to scalar with keep dimensions
        gradX->assign((*y) * (*gradO));
        gradY->assign((*x) * (*gradO));
    }
    else {

        bool keepDims = false;
        auto dimensions = *block.getIArguments();

        if (block.width() > 3) {
            auto axesVector = INPUT_VARIABLE(3);
            helpers::adjustAxis(x->rankOf(), axesVector, dimensions);
        }

        if (block.getBArguments()->size())
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size())
            keepDims = (bool)T_ARG(0);

        REQUIRE_TRUE(dimensions.size() <= x->rankOf(), 0, "REDUCE_DOT_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

        for(const auto& item : dimensions)
            REQUIRE_TRUE(item >= -x->rankOf() && item < x->rankOf(), 0, "REDUCE_DOT_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , x->rankOf(), x->rankOf(), item);

        if(!keepDims) {
            auto gradOShapeKeepDims = ShapeUtils::evalReduceShapeInfo(gradO->ordering(), dimensions, *x, true, false, block.getWorkspace());
            auto r = gradO->reshape(gradO->ordering(), ShapeUtils::pullShapeFromShapeInfo(gradOShapeKeepDims));  // for example could be something like [a,b] -> [1,a,1,b]

            gradX->assign((*y) * r);
            gradY->assign((*x) * r);
        }
        else {
            gradX->assign((*y) * (*gradO));
            gradY->assign((*x) * (*gradO));
        }

    }
    return Status::OK();
}


DECLARE_SHAPE_FN(reduce_dot_bp) {

    if(shape::length(inputShape->at(2)) > 1) {

        bool keepDims = false;
        auto dimensions = *block.getIArguments();

        if (block.width() > 3) {
            auto axesVector = INPUT_VARIABLE(3);
            helpers::adjustAxis(INPUT_VARIABLE(0)->rankOf(), axesVector, dimensions);
        }

        if (block.getBArguments()->size())
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size())
            keepDims = (bool)T_ARG(0);

        REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_DOT_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

        for(const auto& item : dimensions)
            REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_DOT_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);
    }

    Nd4jLong *outShapeInfo1, *outShapeInfo2;
    COPY_SHAPE(inputShape->at(0), outShapeInfo1);
    COPY_SHAPE(inputShape->at(1), outShapeInfo2);

    return SHAPELIST(CONSTANT(outShapeInfo1), CONSTANT(outShapeInfo2));
}

DECLARE_TYPES(reduce_dot_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(sd::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}

#endif

}
}
