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
// Created by GS <sgazeos@gmail.com> at 2/20/18
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_max_pool_with_argmax)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <ops/declarable/helpers/max_pooling.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(max_pool_with_argmax, 1, 2, false, 0, 9) {

            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);
            auto indeces = OUTPUT_VARIABLE(1);

            REQUIRE_TRUE(x->rankOf() == 4, 0, "max_pool_with_argmax: Input should have rank of 4, but got %i instead", x->rankOf());

            auto argI = *(block.getIArguments());

            helpers::maxPoolingFunctor(x, z, argI, indeces);

            return Status::OK();
        }
        
        DECLARE_SHAPE_FN(max_pool_with_argmax) {
            auto in = inputShape->at(0);
            Nd4jLong* valuesShape = nullptr;
            Nd4jLong* indicesShape = nullptr;
            COPY_SHAPE(in, valuesShape);
            COPY_SHAPE(in, indicesShape);
            auto shapes = SHAPELIST(valuesShape, indicesShape);

            return shapes;
        }
    }
}

#endif

