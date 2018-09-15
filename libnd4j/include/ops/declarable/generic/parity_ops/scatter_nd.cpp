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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 21.08.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_scatter_nd)

#include <ops/declarable/CustomOperations.h>
//#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(scatter_nd, 3, 1, false, 0, 0) {
    auto indices = INPUT_VARIABLE(0);
    auto updates = INPUT_VARIABLE(1);
    auto shape   = INPUT_VARIABLE(2);

    auto output = OUTPUT_VARIABLE(0);

        // FIXME: scatter helper should be updated
    /*

    const int indRank   = indices->rankOf();
    const int updRank   = updates->rankOf();
    const int shapeRank = shape->rankOf();
    const Nd4jLong shapeLen  = shape->lengthOf();
    
    REQUIRE_TRUE(shapeRank == 1, 0, "SCATTER_ND OP: the rank of shape array must be 1, but got %i instead !", shapeRank);
    REQUIRE_TRUE(indices->sizeAt(-1) <= shapeRank, 0, "SCATTER_ND OP: last dimension of indices array must be <= rank of shape array, but got %i and %i correspondingly !", indices->sizeAt(-1), shapeRank);    
    REQUIRE_TRUE(updRank == (indRank + shapeLen - 2), 0, "SCATTER_ND OP: the equality updates_rank = (indices_rank + shape_length - 2) must be true for input arrays, but got instead: updates_rank = %i, indices_rank = %i, shape_length = %i !", updRank, indRank, shapeLen);

    std::vector<Nd4jLong> outShape = shape->getBufferAsVector<Nd4jLong>();
    std::vector<Nd4jLong> updShape = updates->getShapeAsVector();
    std::vector<Nd4jLong> indShape = indices->getShapeAsVector();    
    std::vector<Nd4jLong> expectedUpdShape(std::begin(indShape), std::end(indShape) - 1);     
    std::move(std::begin(outShape) + indices->sizeAt(-1), std::end(outShape), std::back_inserter(expectedUpdShape));        
    REQUIRE_TRUE(expectedUpdShape == updShape, 0, "SCATTER_ND OP: wrong shape of updates array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedUpdShape).c_str(), ShapeUtils::shapeAsString(updShape).c_str());
    
    // initial zeroing of output
    if(output->ews() == 1)
        memset(output->getBuffer(), 0, output->lengthOf() * sizeof(T));
    else 
        *output = 0;

    ScatterHelper<T>::template scatterND<simdOps::Copy<T>>(*indices, *updates, *output);
*/
    return Status::OK();
}

////////////////////////////////////////////////////////////////////////7
DECLARE_SHAPE_FN(scatter_nd) {

    auto shape = INPUT_VARIABLE(2);
    auto updShapeInfo = inputShape->at(1);

    Nd4jLong* outShapeInfo;
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(shape->lengthOf()), Nd4jLong);

    outShapeInfo[0] = shape->lengthOf();
    for(int i = 0; i < outShapeInfo[0]; ++i)
        outShapeInfo[i + 1] = shape->getScalar<Nd4jLong>(i);

    shape::updateStrides(outShapeInfo, shape::order(updShapeInfo));
        
    return SHAPELIST(outShapeInfo);
}

}
}

#endif
