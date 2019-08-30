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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_broadcast_dynamic_shape)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(broadcast_dynamic_shape, 2, 1, false, 0, 0) {

    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    auto z = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(x->rankOf() == 1, 0, "BROADCAST_DYNAMIC_SHAPE OP: the first input array must have rank = 1, but got %i instead!", x->rankOf());
    REQUIRE_TRUE(y->rankOf() == 1, 0, "BROADCAST_DYNAMIC_SHAPE OP: the second input array must have rank = 1, but got %i instead!", y->rankOf());
    REQUIRE_TRUE(x->dataType() == y->dataType(), 0, "BROADCAST_DYNAMIC_SHAPE OP: both input arrays must have the same integer type !");

    // contract shapeInfos, neglect and don't fill strides, ews, order
    // shapes are of interest only
    std::vector<Nd4jLong> xShapeInfo(shape::shapeInfoLength(x->lengthOf()));
    std::vector<Nd4jLong> yShapeInfo(shape::shapeInfoLength(y->lengthOf()));

    // fill rank and data type
    xShapeInfo[0] = x->lengthOf();
    yShapeInfo[0] = y->lengthOf();
    ArrayOptions::setDataType(xShapeInfo.data(), nd4j::DataType::INT64); // fill with some data type, it doesn't matter what type exactly to choose
    ArrayOptions::setDataType(yShapeInfo.data(), nd4j::DataType::INT64);

    for (Nd4jLong i = 0; i < x->lengthOf(); ++i)
        xShapeInfo[i + 1] = x->e<Nd4jLong>(i);

    for (Nd4jLong i = 0; i < y->lengthOf(); ++i)
        yShapeInfo[i + 1] = y->e<Nd4jLong>(i);

    Nd4jLong* poinerOnOutShapeInfo = nullptr;

    const bool isBroadcastPossible = ShapeUtils::evalBroadcastShapeInfo(xShapeInfo.data(), yShapeInfo.data(), true, poinerOnOutShapeInfo, block.launchContext()->getWorkspace());

    REQUIRE_TRUE(isBroadcastPossible, 0, "BROADCAST_DYNAMIC_SHAPE OP: the shapes of two input arrays %s and %s are not suitable for broadcast operation !", ShapeUtils::shapeAsString(xShapeInfo.data()).c_str(), ShapeUtils::shapeAsString(yShapeInfo.data()).c_str());

    for (Nd4jLong i = 0; i < z->lengthOf(); ++i)
        z->p<Nd4jLong>(i, poinerOnOutShapeInfo[i + 1]);

    return Status::OK();
}

DECLARE_TYPES(broadcast_dynamic_shape) {
    getOpDescriptor()
        ->setAllowedOutputTypes({ALL_INTS})
        ->setAllowedInputTypes({ALL_INTS});
}


//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(broadcast_dynamic_shape) {

    const int xRank = INPUT_VARIABLE(0)->lengthOf();
    const int yRank = INPUT_VARIABLE(1)->lengthOf();

    const int maxRank = xRank > yRank ? xRank : yRank;

    auto outputShapeInfo = ConstantShapeHelper::getInstance()->vectorShapeInfo(maxRank, ArrayOptions::dataType(inputShape->at(0)));

    return SHAPELIST(outputShapeInfo);
}

}
}

#endif