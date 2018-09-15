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

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(clip_by_global_norm, 1, 2, true, 1, 0) {

    // FIXME: lambdas
    /*
    const T clipNorm = T_ARG(0);
    T globalNorm = 0; //sqrt(sum([l2norm(t)**2 for t in t_list]))

    for (int e = 0; e < block.width(); e++) {
        auto input = INPUT_VARIABLE(e);
        auto l2norm = input->reduceNumber(reduce::Norm2);
        globalNorm += l2norm * l2norm;
    }

    globalNorm = nd4j::math::nd4j_sqrt(globalNorm);
    OUTPUT_VARIABLE(block.width())->putScalar(0, globalNorm);
    const T factor = clipNorm / globalNorm;

    for (int e = 0; e < block.width(); e++) {
        // all-reduce
        auto input = INPUT_VARIABLE(e);
        auto output = OUTPUT_VARIABLE(e);

        if (globalNorm <= clipNorm) {
            output->assign(input);
        } 
        else {
            
            auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
            input->applyLambda(lambda, output);
        }
    }
    */

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