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
// @author GS <sgazeos@gmail.com> 3/21/2018
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matrixSetDiag.h>

namespace sd {
namespace ops  {

CUSTOM_OP_IMPL(matrix_diag, 1, 1, false, 0, 0) {

    auto diagonal = INPUT_VARIABLE(0);
    auto output   = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(!diagonal->isScalar(), 0, "CUSTOM_OP matrix_diag: input diagonal array must be at list a vector, but scalar was given!");

    helpers::matrixSetDiag(block.launchContext(), *output, *diagonal, *output, true);

    return Status::OK();
}

DECLARE_SHAPE_FN(matrix_diag) {

    Nd4jLong* outShapeInfo = nullptr;
    auto in = inputShape->at(0);
    int inRank = shape::rank(in);

    //  if for example diagonal array has shape [A,B,C] then output array has shape [A,B,C,C]

    int outRank = inRank + 1;

    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);
    outShapeInfo[0] = outRank;
    for(int i = 0; i < inRank; ++i)
        outShapeInfo[i + 1] = shape::sizeAt(in, i);
    outShapeInfo[outRank] = shape::sizeAt(in, -1);

    ShapeUtils::updateStridesAndType(outShapeInfo, in, shape::order(in));

    return SHAPELIST(CONSTANT(outShapeInfo));
}

DECLARE_TYPES(matrix_diag) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setSameMode(true);
}
}
}

