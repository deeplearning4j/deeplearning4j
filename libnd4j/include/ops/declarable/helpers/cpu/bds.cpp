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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/bds.h>
#include <Status.h>


namespace nd4j {
namespace ops {
namespace helpers {

    Nd4jStatus bdsFunctor(nd4j::LaunchContext * context, NDArray* x_shape, NDArray* y_shape, NDArray* output) {


        if (x_shape->lengthOf() == 1 || y_shape->lengthOf() == 1) {// except case
            // lenght are equals
            if (x_shape->lengthOf() == y_shape->lengthOf()) {
                auto greater = (x_shape->e<Nd4jLong>(0) < y_shape->e<Nd4jLong>(0) ? y_shape : x_shape);
                output->assign(greater);
            }
            else {
                auto lesser = (x_shape->lengthOf() == 1 ? x_shape : y_shape);
                auto greater = (x_shape->lengthOf() == 1 ? y_shape : x_shape);
                output->assign(greater);
                auto lastG = greater->lengthOf() - 1;
                auto lastL = lesser->lengthOf() - 1;
                if (greater->e<Nd4jLong>(lastG) < lesser->e<Nd4jLong>(lastL))
                    output->p(lastG, lesser->e(lastL));
            }
        }
        else {
            //int e = 0, x = 0, y = 0;
            Nd4jLong xLen = x_shape->lengthOf();
            Nd4jLong yLen = y_shape->lengthOf();
            Nd4jLong zLen = output->lengthOf();
            Nd4jLong borderLen = nd4j::math::nd4j_min(xLen, yLen);
            for (Nd4jLong e = 0; e < zLen; e++) {
                Nd4jLong val;
                if (e < borderLen) {
                    val = nd4j::math::nd4j_max(x_shape->e<Nd4jLong>(e), y_shape->e<Nd4jLong>(e));
                } else if (e < xLen) {
                    val = nd4j::math::nd4j_max(x_shape->e<Nd4jLong>(e), y_shape->e<Nd4jLong>(yLen - 1));
                } else {
                    val = nd4j::math::nd4j_max(x_shape->e<Nd4jLong>(xLen - 1), y_shape->e<Nd4jLong>(e));
                }

                output->p(e, val);
            }
        }
        return Status::OK();
    }

}
}
}