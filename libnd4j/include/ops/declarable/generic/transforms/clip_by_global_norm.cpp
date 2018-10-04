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
//  @author sgazeos@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_clip_by_global_norm)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(clip_by_global_norm, 1, 2, true, 1, 0) {

    std::vector<NDArray*> inputs(block.width());
    std::vector<NDArray*> outputs(block.width() + 1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = INPUT_VARIABLE(i);
        outputs[i] = OUTPUT_VARIABLE(i);
    }
    outputs[inputs.size()] = OUTPUT_VARIABLE(inputs.size());
    double clipNorm = T_ARG(0);
    bool isInplace = block.isInplace();
    helpers::clipByGlobalNorm(inputs, clipNorm, block.workspace(), outputs, isInplace);

    return Status::OK();
}

DECLARE_SHAPE_FN(clip_by_global_norm) {

    auto shapeList = SHAPELIST();
            
    for (int e = 0; e < block.width(); e++) {
        auto in = inputShape->at(e);
                
        Nd4jLong* newShape;
        COPY_SHAPE(in, newShape);
        shapeList->push_back(newShape);
    }

    shapeList->push_back(ShapeBuilders::createScalarShapeInfo(block.workspace()));
    return shapeList;
}


}
}

#endif