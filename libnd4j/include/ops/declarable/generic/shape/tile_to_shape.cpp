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
#if NOT_EXCLUDED(OP_tile_to_shape)

#include <ops/declarable/headers/shape.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(tile_to_shape, 1, 1, true, 0, -1) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);

        if (block.isInplace()) {
            //input->tileToShape(*block.getIArguments());
        } else {
            //input->tileToShape(*block.getIArguments(), output);
        }

        return Status::OK();
    }

    DECLARE_SHAPE_FN(tile_to_shape) {
        auto in = inputShape->at(0);

        // output shape always equals to arguments
        Nd4jLong *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(block.numI()), Nd4jLong);

        auto conv = ArrayUtils::toLongVector(*block.getIArguments());

        if (shape::order(in) == 'c')
            shape::shapeBuffer(block.numI(), block.dataType(), conv.data(), newShape);
        else
            shape::shapeBufferFortran(block.numI(),block.dataType(),  conv.data(), newShape);

        return SHAPELIST(newShape);
    }


    CUSTOM_OP_IMPL(tile_to_shape_bp, 2, 1, true, 0, -1) {
        auto input = INPUT_VARIABLE(0);
        auto epsNext = INPUT_VARIABLE(1);

        auto gradX = OUTPUT_VARIABLE(0);

        auto axisX = ShapeUtils::evalBroadcastBackwardAxis(input->shapeInfo(), epsNext->shapeInfo());
        // FIX ME: reduceAlongDims should have a signature with result pass to to avoid assigning twice
        if (!axisX.empty()) {
            auto tempRes = epsNext->reduceAlongDims(reduce::Sum, axisX);
            gradX->assign(tempRes);
        } else
            gradX->assign(epsNext);

        STORE_RESULT(gradX);

        return Status::OK();
    }

    DECLARE_SHAPE_FN(tile_to_shape_bp) {
        auto in = inputShape->at(0);

        Nd4jLong *newShape;
        COPY_SHAPE(in, newShape);

        return SHAPELIST(newShape);
    }
}
}

#endif