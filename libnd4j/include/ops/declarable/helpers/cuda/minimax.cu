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
#ifndef __MIN_I_MAX_H_HELPERS__
#define __MIN_I_MAX_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T> 
    static void minimumBPFunctor_(NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {

    }

    template <typename T>
    void maximumBPFunctor_(NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {

    }

    void minimumBPFunctor(nd4j::LaunchContext * context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
        BUILD_SINGLE_SELECTOR(x->dataType(), minimumBPFunctor_, (x, y, epsNext, gradX, gradY), NUMERIC_TYPES);
    }

    void maximumBPFunctor(nd4j::LaunchContext * context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
        BUILD_SINGLE_SELECTOR(x->dataType(), maximumBPFunctor_, (x, y, epsNext, gradX, gradY), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void minimumBPFunctor_, (NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY), NUMERIC_TYPES);
    BUILD_SINGLE_TEMPLATE(template void maximumBPFunctor_, (NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY), NUMERIC_TYPES);

}
}
}
#endif
