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
//  mmultimul op. Created by GS <sgazeos@google.com> 8/9/2018
//  multiple matricies product
//

#include <op_boilerplate.h>

#if NOT_EXCLUDED(OP_multimmul2)

#include <ops/declarable/CustomOperations.h>
//#include <ops/declarable/helpers/matmul.h>
//#include <MmulHelper.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(multimmul2, 2, 1, false, 0, 0) {
            NDArray<T>* startMatrix = INPUT_VARIABLE(0);
//            NDArray<T>* y = INPUT_VARIABLE(1);
//            NDArray<T>* b = INPUT_VARIABLE(2);
            NDArray<T>* z = OUTPUT_VARIABLE(0);
            for (Nd4jLong nextI = 1; nextI < block.width(); ++nextI) {
                NDArray<T>* nextM = INPUT_VARIABLE(nextI);
                // TO DO: the rest of op here.
                for (Nd4jLong e = 0; e < z->lengthOf(); ++e) {

                }
            }
            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(multimmul2) {
            //auto outputShape = ShapeUtils<T>::matrixProductShape(inputShape->at(0), inputShape->at(1), false, false, block.getWorkspace());
            auto lastShape = inputShape->at(block.width() - 1);
            auto firstShape = inputShape->at(0);

            Nd4jLong* outputShape;
            Nd4jLong outRank = 2; // 2D tensor as output
            ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(shape::rank(firstShape)), Nd4jLong);
            outputShape[1] = shape::sizeAt(firstShape, 0);
            outputShape[2] = shape::sizeAt(lastShape, 1);
            shape::updateStrides(outputShape, shape::order(firstShape));

            return SHAPELIST(outputShape);
        }

    }
}

#endif