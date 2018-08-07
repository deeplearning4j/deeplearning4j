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
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reshapeas)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {


    //////////////////////////////////////////////////////////////////////////
    CUSTOM_OP_IMPL(reshapeas, 2, 1, true, 0, 0) {
    
        NDArray<T> *x = INPUT_VARIABLE(0);
        NDArray<T> *y = INPUT_VARIABLE(1);

        NDArray<T>* z = OUTPUT_VARIABLE(0);
        std::vector<Nd4jLong> shapeNew(y->shapeOf(), y->shapeOf() + y->rankOf());
        char order = y->ordering();

        if (x->reshapei(order, shapeNew)) {
            *z = *x;
            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }

        return ND4J_STATUS_BAD_INPUT;
    }
    DECLARE_SYN(reshape_as, reshapeas);
    
    DECLARE_SHAPE_FN(reshapeas) {
    
    auto inputShapeInfo = inputShape->at(1);    
    int shapeInfoLength = inputShapeInfo[0]*2 + 4;

    // FIXME: remove memcpy
    Nd4jLong* outputShapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, outputShapeInfo);
    
    return SHAPELIST(outputShapeInfo);
}

}
}

#endif