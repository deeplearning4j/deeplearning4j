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
// Created by GS <sgazeos@gmail.com>
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/random_crop.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(random_crop, 2, 1, false, 0, 0) {
    auto input   = INPUT_VARIABLE(0); // values for crop
    auto shape   = INPUT_VARIABLE(1); // shape for result

    NDArray* reduceShape = nullptr; // this param is optional
    auto output  = OUTPUT_VARIABLE(0); //
    
    int seed = 0;

    if (block.getIArguments()->size() > 0)
        seed = INT_ARG(0);

    REQUIRE_TRUE(shape->isVector(), 0, "random_crop: Shape tensor should be a vector.");
    
    REQUIRE_TRUE(input->rankOf() == shape->lengthOf(), 0, "random_crop: The length of the shape vector is not match input rank. %i and %i were given.", 
        input->rankOf(), shape->lengthOf());

    for (int e = 0; e < shape->lengthOf(); ++e) {
        REQUIRE_TRUE((*shape).getScalar<Nd4jLong>(e) <= input->sizeAt(e), 0, "random_crop: Shape tensor should be less than proper input dimension (dim %i, %i > %i).", e, (*shape).getScalar<Nd4jLong>(e), input->sizeAt(e));
    }

    nd4j::random::RandomBuffer* rng = block.getRNG();
        
    if (rng == nullptr)
        return ND4J_STATUS_BAD_RNG;

    return helpers::randomCropFunctor(rng, input, shape, output, seed);
}

DECLARE_SHAPE_FN(random_crop) {
    auto in = INPUT_VARIABLE(1);
    std::vector<Nd4jLong> shape(in->lengthOf());

    Nd4jLong *newShape;
    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), Nd4jLong);
    for (int e = 0; e < shape.size(); e++)
        shape[e] = (*in).getScalar<Nd4jLong>(e);

    shape::shapeBuffer(shape.size(), shape.data(), newShape);

    return SHAPELIST(newShape);
}


}
}