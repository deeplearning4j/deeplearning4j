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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_clipbynorm)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

    CONFIGURABLE_OP_IMPL(clipbynorm, 1, 1, true, 1, 0) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);

        const T clipNorm = T_ARG(0);
        const bool isInplace = block.isInplace();
        
        helpers::clipByNorm(*input, *output, *block.getIArguments(), clipNorm, isInplace);

        return Status::OK();
    }


    CUSTOM_OP_IMPL(clipbynorm_bp, 2, 1, false, 1, 0) {
        auto input = INPUT_VARIABLE(0);
        auto gradO = INPUT_VARIABLE(1);

        auto gradI = OUTPUT_VARIABLE(0);
        const T clipNorm = T_ARG(0);

        helpers::clipByNormBP(*input, *gradO, *gradI, *block.getIArguments(), clipNorm); 

        return Status::OK();
    }

    DECLARE_SHAPE_FN(clipbynorm_bp) {
        auto inShapeInfo = inputShape->at(0);

        Nd4jLong *newShape = nullptr;
        COPY_SHAPE(inShapeInfo, newShape);

        return SHAPELIST(newShape);
    }


}
}

#endif