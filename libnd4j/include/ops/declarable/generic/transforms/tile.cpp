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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_tile)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(tile, 1, 1, false, 0, -2) {
    
    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);
    
    const int inRank = input->rankOf();
    std::vector<Nd4jLong> reps;

    if (block.getIArguments()->size() == inRank) {

        reps = ArrayUtils::toLongVector(*(block.getIArguments()));        
    } 
    else if (block.width() > 1)  {
        
        auto reps_vector = INPUT_VARIABLE(1);
        REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0, "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", reps_vector->lengthOf(), inRank);

        reps = reps_vector->template asVectorT<Nd4jLong>();
    }
    else {
        REQUIRE_TRUE(false, 0, "TILE op: this op requires repeats vector, either as IArgs or second array with length equal to rank of input array to be tiled !");
    }
            
    input->tile(reps, *output);

    return Status::OK();
}


DECLARE_SHAPE_FN(tile) {
    
    Nd4jLong* inShape = inputShape->at(0);
    const int inRank = inShape[0];
    std::vector<Nd4jLong> reps;

    if (block.getIArguments()->size() == inRank) {

        reps = ArrayUtils::toLongVector(*(block.getIArguments()));        
    } 
    else if (block.width() > 1)  {
        
        auto reps_vector = INPUT_VARIABLE(1);
        REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0, "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", reps_vector->lengthOf(), inRank);
        reps = reps_vector->template asVectorT<Nd4jLong>();
    }
    else {
        REQUIRE_TRUE(false, 0, "TILE op: this op requires repeats vector, either as IArgs or second array with length equal to rank of input array to be tiled !");
    }

    Nd4jLong* newShape;
    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    
    std::vector<Nd4jLong> shape(inRank);
    for (int e = 0; e < shape::rank(inShape); e++)
        shape[e] = shape::sizeAt(inShape, e) * reps[e];

    if (shape::order(inShape) == 'c')
        shape::shapeBuffer(shape.size(), block.dataType(), shape.data(), newShape);
    else
        shape::shapeBufferFortran(shape.size(), block.dataType(), shape.data(), newShape);
       
    return SHAPELIST(newShape);
}


////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tile_bp, 1, 1, false, 0, -2) {
    
    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);
    auto gradI = OUTPUT_VARIABLE(0);
    
    const int inRank = input->rankOf();

    REQUIRE_TRUE(inRank == gradO->rankOf(), 0, "TILE_BP op: the ranks of input array and output's gradients array (next epsilon) must be equal, but got %i and %i correspondingly !", inRank, gradO->rankOf());

    std::vector<Nd4jLong> reps;

    if (block.getIArguments()->size() == inRank) {

        reps = ArrayUtils::toLongVector(*(block.getIArguments()));        
    } 
    else if (block.width() > 1)  {
        
        auto reps_vector = INPUT_VARIABLE(1);
        REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0, "TILE_BP op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", reps_vector->lengthOf(), inRank);

        reps = reps_vector->template asVectorT<Nd4jLong>();
    }
    else {
        REQUIRE_TRUE(false, 0, "TILE_BP op: this op requires repeats vector, either as IArgs or second array with length equal to rank of input array to be tiled !");
    }

    for (int i = 0; i < inRank; ++i)
        REQUIRE_TRUE(gradO->sizeAt(i) == gradI->sizeAt(i) * reps[i], 0, "TILE_BP op: shapes of input array and output's gradients array (next epsilon) are inconsistent !");
            
    helpers::tileBP(*gradO, *gradI, reps);

    return Status::OK();
}

DECLARE_SHAPE_FN(tile_bp) {
    
    Nd4jLong* inShape    = inputShape->at(0);
    Nd4jLong* gradOShape = inputShape->at(1);
    const int inRank = inShape[0];

    REQUIRE_TRUE(inRank == gradOShape[0], 0, "TILE_BP op: the ranks of input array and output's gradients array (next epsilon) must be equal, but got %i and %i correspondingly !", inRank, gradOShape[0]);

    std::vector<Nd4jLong> reps;

    if (block.getIArguments()->size() == inRank) {

        reps = ArrayUtils::toLongVector(*(block.getIArguments()));        
    } 
    else if (block.width() > 1)  {
        
        auto reps_vector = INPUT_VARIABLE(1);
        REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0, "TILE_BP op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", reps_vector->lengthOf(), inRank);
        reps = reps_vector->template asVectorT<Nd4jLong>();
    }
    else {
        REQUIRE_TRUE(false, 0, "TILE_BP op: this op requires repeats vector, either as IArgs or second array with length equal to rank of input array to be tiled !");
    }
    
    for (int i = 0; i < inRank; ++i)
        REQUIRE_TRUE(shape::sizeAt(gradOShape, i) == shape::sizeAt(inShape, i) * reps[i], 0, "TILE_BP op: shapes of input array and output's gradients array (next epsilon) are inconsistent !");

    Nd4jLong *gradIShape;
    COPY_SHAPE(inShape, gradIShape);

    return SHAPELIST(gradIShape);

}


}
}

#endif