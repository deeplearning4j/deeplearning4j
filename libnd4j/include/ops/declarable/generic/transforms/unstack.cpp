/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_unstack)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/stack.h>

namespace sd {
namespace ops  {

CUSTOM_OP_IMPL(unstack, 1, -1, false, 0, 1) {

    auto input = INPUT_VARIABLE(0);

    auto dim = INT_ARG(0);
    if (dim < 0)
        dim += input->rankOf();


    REQUIRE_TRUE(dim < input->rankOf(), 0, "Unstack dimension should be lower then rank of input %i, but got dimension=%i !", input->rankOf(), dim);
    REQUIRE_TRUE(dim >= 0, 0, "Unstack dimension should be non-negative value, but got %i !", dim);

    if(input->isEmpty())
        return Status::OK();

    std::vector<NDArray*> outArrs(input->sizeAt(dim));
    for(uint i = 0; i < outArrs.size(); ++i)
        outArrs[i] = OUTPUT_VARIABLE(i);

    helpers::unstack(block.launchContext(), *input, outArrs, dim);

    return Status::OK();
}

DECLARE_SYN(unpack, unstack);

DECLARE_SHAPE_FN(unstack) {
    auto inShapeInfo = inputShape->at(0);

    auto dim = INT_ARG(0);
    if (dim < 0)
        dim += shape::rank(inShapeInfo);

    REQUIRE_TRUE(dim < inShapeInfo[0], 0, "UNSTACK op: dimension should be lower then rank of input %i, but got dimension=%i !", inShapeInfo[0], dim);
    REQUIRE_TRUE(dim >= 0, 0, "UNSTACK op: dimension should be non-negative value, but got %i !", dim);

    if(ArrayOptions::arrayType(inShapeInfo) == ArrayType::EMPTY) {

        if(shape::shapeOf(inShapeInfo)[dim] == 0)
            return SHAPELIST();

        const Nd4jLong numTads = shape::shapeOf(inShapeInfo)[dim];
        std::vector<Nd4jLong> outShape;
        for(uint i = 0; i < shape::rank(inShapeInfo); ++i)
            if(i != dim)
                outShape.push_back(shape::shapeOf(inShapeInfo)[i]);

        auto result = SHAPELIST();
        for(uint i = 0; i < numTads; ++i)
            result->push_back(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShapeInfo), shape::order(inShapeInfo), outShape));

        return result;
    }

    std::vector<int> dims = ShapeUtils::evalDimsToExclude(inShapeInfo[0], {dim});

    if (dims.size() == 0 && shape::rank(inShapeInfo) == 1) { // split vector into lenthOf scalars

        auto result = SHAPELIST();
        for (Nd4jLong e = 0; e < shape::length(inShapeInfo); e++)
            result->push_back(ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(inShapeInfo)));

        return result;
    }

    std::vector<Nd4jLong> subArrShape(shape::rank(inShapeInfo) - 1);

    for(uint j = 0, i = 0; i < shape::rank(inShapeInfo); i++)
        if(i != dim)
            subArrShape[j++] = shape::shapeOf(inShapeInfo)[i];

    // remove leading and trailing 1
    if (inShapeInfo[0] == 2 && subArrShape.size() == 2) {

        if (subArrShape[0] == 1)
            subArrShape.erase(subArrShape.begin());
        else if (subArrShape[1] == 1)
            subArrShape.erase(subArrShape.end());
    }

    auto result = SHAPELIST();
    for (int e = 0; e < shape::shapeOf(inShapeInfo)[dim]; e++) {
        auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShapeInfo), shape::order(inShapeInfo), subArrShape);
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