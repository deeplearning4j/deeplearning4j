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
// @author George A. Shulinok <sgazeos@gmail.com), created on 4/18/2019.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_barnes_gains)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace nd4j {
namespace ops  {
		
    OP_IMPL(barnes_gains, 3, 1, true) {
        auto input  = INPUT_VARIABLE(0);
        auto gradX = INPUT_VARIABLE(1);
        auto epsilon = INPUT_VARIABLE(2);

        auto output = OUTPUT_VARIABLE(0);
        //        gains = gains.add(.2).muli(sign(yGrads)).neq(sign(yIncs)).castTo(Nd4j.defaultFloatingPointType())
        //                .addi(gains.mul(0.8).muli(sign(yGrads)).neq(sign(yIncs)));
        auto signGradX = *gradX;
        auto signEpsilon = *epsilon;
        gradX->applyTransform(transform::Sign, &signGradX, nullptr);
        epsilon->applyTransform(transform::Sign, &signEpsilon, nullptr);
        auto leftPart = (*input + 2.) * signGradX;
        auto leftPartBool = NDArrayFactory::create<bool>(leftPart.ordering(), leftPart.getShapeAsVector());

        leftPart.applyPairwiseTransform(pairwise::NotEqualTo, &signEpsilon, &leftPartBool, nullptr);
        auto rightPart = *input * 0.8 * signGradX;
        auto rightPartBool = NDArrayFactory::create<bool>(rightPart.ordering(), rightPart.getShapeAsVector());
        rightPart.applyPairwiseTransform(pairwise::NotEqualTo, &signEpsilon, &rightPartBool, nullptr);
        leftPart.assign(leftPartBool);
        rightPart.assign(rightPartBool);
        leftPart.applyPairwiseTransform(pairwise::Add, &rightPart, output, nullptr);

        return Status::OK();
    }

    DECLARE_TYPES(barnes_gains) {
        getOpDescriptor()
            ->setAllowedInputTypes(nd4j::DataType::ANY)
            ->setSameMode(true);
    }
}
}

#endif