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


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int bdsFunctor(NDArray<T>* x_shape, NDArray<T>* y_shape, NDArray<T>* output) {
        int e = 0, x = 0, y = 0;
//#pragma omp parallel for
        for ( ; e < output->lengthOf(); e++) {
            T val;
            if (x < x_shape->lengthOf() && y < y_shape->lengthOf()) {
                val = nd4j::math::nd4j_max((*x_shape)(x++), (*y_shape)(y++));
            }
            else if (x < x_shape->lengthOf()) {
                val = nd4j::math::nd4j_max((*x_shape)(x++), (*y_shape)(y - 1));
            }
            else if (y < y_shape->lengthOf()) {
                val = nd4j::math::nd4j_max((*x_shape)(x - 1), (*y_shape)(y++));
            }
            else {
                //REQUIRE_TRUE(e < 0, 0, "broadcast_dynamic_shape: Wrong value in a shape vector");
                return ND4J_STATUS_OK;
            }
            if (e)
                if (val != (*output)(e - 1)) {
                    nd4j_printf("broadcast_dynamic_shape: Input shapes should be compatible", "");
                    return ND4J_STATUS_VALIDATION;
                }
            (*output)(e) = val;
        }
        return ND4J_STATUS_OK;
    }

    template int bdsFunctor(NDArray<float>* x_shape, NDArray<float>* y_shape, NDArray<float>* output);
    template int bdsFunctor(NDArray<float16>* x_shape, NDArray<float16>* y_shape, NDArray<float16>* output);
    template int bdsFunctor(NDArray<double>* x_shape, NDArray<double>* y_shape, NDArray<double>* output);
}
}
}