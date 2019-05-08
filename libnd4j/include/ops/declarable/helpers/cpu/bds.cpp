/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/bds.h>
#include <Status.h>


namespace nd4j {
namespace ops {
namespace helpers {

    Nd4jStatus bdsFunctor(NDArray* x_shape, NDArray* y_shape, NDArray* output) {
        int e = 0, x = 0, y = 0;

        if (x_shape->lengthOf() == 1 || y_shape->lengthOf() == 1) {// except case
            auto lesser = (x_shape->lengthOf() == 1 ? x_shape: y_shape);
            auto greater = (x_shape->lengthOf() == 1 ? y_shape: x_shape);
            output->assign(greater);

            output->p(greater->lengthOf() - 1, lesser->e(0L));
        }
        else {
            for (; e < output->lengthOf(); e++) {
                Nd4jLong val;
                if (x < x_shape->lengthOf() && y < y_shape->lengthOf()) {
                    val = nd4j::math::nd4j_max(x_shape->e<Nd4jLong>(x++), y_shape->e<Nd4jLong>(y++));
                } else if (x < x_shape->lengthOf()) {
                    val = nd4j::math::nd4j_max(x_shape->e<Nd4jLong>(x++), y_shape->e<Nd4jLong>(y - 1));
                } else if (y < y_shape->lengthOf()) {
                    val = nd4j::math::nd4j_max(x_shape->e<Nd4jLong>(x - 1), y_shape->e<Nd4jLong>(y++));
                } else {
                    //REQUIRE_TRUE(e < 0, 0, "broadcast_dynamic_shape: Wrong value in a shape vector");
                    return ND4J_STATUS_OK;
                }
//                if (e)
//                    if (val != output->e<Nd4jLong>(e - 1)) {
//                        nd4j_printf(
//                                "broadcast_dynamic_shape: Input shapes should be compatible, but %lld and %lld were given.\n",
//                                val, output->e<Nd4jLong>(e - 1));
//                        return Status::CODE(ND4J_STATUS_VALIDATION, "broadcast_dynamic_shape: BDS validation failed!");
//                    }
                output->p(e, val);
            }
        }
        return Status::OK();
    }

}
}
}