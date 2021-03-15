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
// Created by GS <sgazeos@gmail.com>
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_embedding_lookup)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>
#include <numeric>


namespace sd {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(embedding_lookup, 2, 1, false, 0, 1) {
    auto input   = INPUT_VARIABLE(0); // lookup param
    auto indices = INPUT_VARIABLE(1); // indices, as is
    auto output  = OUTPUT_VARIABLE(0); //

    if (block.width() > 2) { // multiple input
        indices = INPUT_VARIABLE(block.width() - 1);
        std::vector<int> dims(input->rankOf());
        int i = output->rankOf() - input->rankOf();
        for (auto& v: dims){
            v = i++;
        }

        ResultSet outputView = output->allTensorsAlongDimension(dims);
        REQUIRE_TRUE(block.width() > output->sizeAt(0), 0, "embedding_lookup: input list should be greater then %i, but %i given.",
                    output->sizeAt(0), block.width()
                );
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
            Nd4jLong thisIndex = (*indices).e<Nd4jLong>(e);
            input   = INPUT_VARIABLE(thisIndex); // lookup param

            outputView.at(e)->assign(input);
        }
    }
    else {
        int indexRank = indices->rankOf();
        REQUIRE_TRUE(indexRank > 0, 0, "embeded_lookup: input array of indexes can't be single scalar, the requirement is: rank > 0 !");

        int inputRank = input->rankOf();
        int lastIndDim = indices->lengthOf();
        int partition_mode = INT_ARG(0); // partition_mode == 0 - i.e. 'mod' , 1 - 'div'

        sd::ops::gather op;

        auto result(op.evaluate({input, indices}, {0}));
        REQUIRE_TRUE(result.status() == Status::OK(), 0, "embedding_lookup: cannot retrieve results from gather op.");
        REQUIRE_TRUE(result.at(0)->isSameShape(output), 0, "embedding_lookup: wrong shape of return from gather op.");
        output->assign(result.at(0));
    }
    return Status::OK();
}

DECLARE_TYPES(embedding_lookup) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setAllowedOutputTypes(sd::DataType::ANY);
}

DECLARE_SHAPE_FN(embedding_lookup) {

    auto inShapeInfo = inputShape->at(0);
    auto indicesShapeInfo = inputShape->at(1);
    int inRank = shape::rank(inShapeInfo);
    if (inputShape->size() == 2u) {
        int outRank = inRank;

        std::vector<Nd4jLong> shapeInfo(outRank);

        shapeInfo[0] = indicesShapeInfo[1]; // vector - how many elements
        for (int e = 1; e < outRank; e++)
            shapeInfo[e] = shape::sizeAt(inShapeInfo, e);

        auto outShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShapeInfo), shape::order(inShapeInfo), shapeInfo);
        return SHAPELIST(outShapeInfo);
    }


    int outRank = inRank + 1;
    std::vector<Nd4jLong> shapeInfo(outRank);
    auto indices = INPUT_VARIABLE(block.width() - 1);
    shapeInfo[0] = indices->lengthOf(); // vector - how many elements
    for (int e = 1; e < outRank; e++)
        shapeInfo[e] = shape::sizeAt(inShapeInfo, e);

    auto outShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShapeInfo), shape::order(inShapeInfo), shapeInfo);
    return SHAPELIST(outShapeInfo);
}




}
}

#endif