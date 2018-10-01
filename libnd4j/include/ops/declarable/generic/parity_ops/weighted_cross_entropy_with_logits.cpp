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
//  @author @shugeo
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_weighted_cross_entropy_with_logits)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/cross.h>

namespace nd4j {
namespace ops {

    OP_IMPL(weighted_cross_entropy_with_logits, 3, 1, true) {
        auto targets = INPUT_VARIABLE(0);
        auto input = INPUT_VARIABLE(1);
        auto weights = INPUT_VARIABLE(2);
        auto output = OUTPUT_VARIABLE(0);

        REQUIRE_TRUE(targets->isSameShape(input), 0, "WEIGHTED_CROSS_ENTROPY_WITH_LOGITS op: The shape of both input params should be equal, but got input_shape=%s and targets_shape=%s !", ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(targets).c_str());
        REQUIRE_TRUE(weights->isScalar() || targets->sizeAt(-1) == weights->lengthOf(), 0, "WEIGHTED_CROSS_ENTROPY_WITH_LOGITS op: The weights should be scalar or vector with length equal to size of last targets dimension, but got weights_shape=%s instead!", ShapeUtils::shapeAsString(weights).c_str());

        // helpers::weightedCrossEntropyWithLogitsFunctor(targets, input, weights, output);

        return Status::OK();
    }
}
}

#endif