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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 03.09.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_broadcast_to)

#include <ops/declarable/headers/shape.h>

namespace sd {
namespace ops  {

CUSTOM_OP_IMPL(broadcast_to, 2, 1, false, 0, 0) {

    auto input  = INPUT_VARIABLE(0);
    auto shape  = INPUT_VARIABLE(1);
    
    auto output = OUTPUT_VARIABLE(0);

    const int      inputRank = input->rankOf();
    const int      shapeRank = shape->rankOf();
    const Nd4jLong shapeLen  = shape->lengthOf();

    REQUIRE_TRUE(shapeRank <= 1, 0, "BROADCAST_TO op: rank of shape array should be <= 1, bot got %i instead !", shapeRank);
    REQUIRE_TRUE(inputRank <= shapeLen, 0, "BROADCAST_TO op: rank of input shape array should be <= length of shape array, bot got %i and %i correspondingly !", inputRank, shapeLen);

    std::vector<Nd4jLong > shapeBuff = shape->getBufferAsVector<Nd4jLong>();
    std::vector<Nd4jLong> outShape(shapeBuff.begin(), shapeBuff.end());

    for(int i = 1; i <= inputRank; ++i)
        REQUIRE_TRUE(input->sizeAt(inputRank-i) == outShape[shapeLen-i] || input->sizeAt(inputRank-i) == 1, 0, "BROADCAST_TO op: shape of input array %s can't be broadcasted to the shape %s !", ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(outShape).c_str());

    input->tile(*output);

    return Status::OK();
}

DECLARE_TYPES(broadcast_to) {
    getOpDescriptor()
        ->setAllowedInputTypes(DataType::ANY)
        ->setSameMode(true);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(broadcast_to) {
    
    auto inputShapeInfo = inputShape->at(0);
    auto shape  = INPUT_VARIABLE(1);

    const int      inputRank = inputShapeInfo[0];
    const int      shapeRank = shape->rankOf();
    const Nd4jLong shapeLen  = shape->lengthOf();

    REQUIRE_TRUE(shapeRank <= 1, 0, "BROADCAST_TO op: rank of input shape array should be <= 1, bit got %i instead !", shapeRank);
    REQUIRE_TRUE(inputRank <= shapeLen, 0, "BROADCAST_TO op: rank of input shape array should be <= length of shape array, bot got %i and %i correspondingly !", inputRank, shapeLen);

    std::vector<Nd4jLong> shapeBuff = shape->getBufferAsVector<Nd4jLong>();
    std::vector<Nd4jLong> outShape(shapeBuff.begin(), shapeBuff.end());

    for(int i = 1; i <= inputRank; ++i)
        REQUIRE_TRUE(inputShapeInfo[inputRank+1-i] == outShape[shapeLen-i] || inputShapeInfo[inputRank+1-i] == 1, 0, "BROADCAST_TO op: shape of input array %s can't be broadcasted to the shape %s !", ShapeUtils::shapeAsString(inputShapeInfo).c_str(), ShapeUtils::shapeAsString(outShape).c_str());
        
    auto outShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inputShapeInfo), shape::order(inputShapeInfo), outShape);
    return SHAPELIST(outShapeInfo);
}

}
}

#endif