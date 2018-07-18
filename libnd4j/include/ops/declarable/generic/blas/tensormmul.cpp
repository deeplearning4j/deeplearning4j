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
#if NOT_EXCLUDED(OP_tensormmul)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <MmulHelper.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(tensormmul, 2, 1, false, 0, -1) {
            NDArray<T>* a = INPUT_VARIABLE(0);
            NDArray<T>* b = INPUT_VARIABLE(1);

            NDArray<T>* c = OUTPUT_VARIABLE(0);                // 

            // building axes
            int axe0_size = INT_ARG(0);
            int axe1_size = INT_ARG(axe0_size+1);
            std::vector<int> axes_0(axe0_size), axes_1(axe1_size);
            for (int e = 0; e < axe0_size; e++)
                axes_0[e] = (int) INT_ARG(e+1);

            for (int e = 0; e < axe1_size; e++)
                axes_1[e] = (int) INT_ARG(e + axe0_size + 2);

            nd4j_verbose("axe0: %i; axe1: %i;\n", axes_0.size(), axes_1.size());

            // nd4j::MmulHelper<T>::tensorDot(a, b, c, axes_0, axes_1);
            NDArray<T>* result = nd4j::MmulHelper<T>::tensorDot(a, b, axes_0, axes_1);
            *c = *result;

            STORE_RESULT(*c);
            delete result;  
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(tensordot, tensormmul);


        DECLARE_SHAPE_FN(tensormmul) {               
        
            auto aShapeInfo = inputShape->at(0);
            auto bShapeInfo = inputShape->at(1);  
            // building axes
            int axe0_size = INT_ARG(0);
            int axe1_size = INT_ARG(axe0_size+1);
            std::vector<int> axes_0(axe0_size), axes_1(axe1_size);
            for (int e = 0; e < axe0_size; e++)
                axes_0[e] = (int) INT_ARG(e+1);

            for (int e = 0; e < axe1_size; e++)
                axes_1[e] = (int) INT_ARG(e + axe0_size + 2);

            // evaluate shapes 
            std::vector<int> permutAt, permutBt;
            std::vector<Nd4jLong> shapeAt, shapeBt;
            auto outShape = nd4j::ShapeUtils<T>::evalShapeForTensorDot(aShapeInfo, bShapeInfo, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);
            
            int rank = outShape.size();

            Nd4jLong* newShapeInfo = nullptr; 
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong); 
            newShapeInfo[0] = rank;
            std::copy(outShape.begin(), outShape.end(), newShapeInfo+1);
            
            shape::updateStrides(newShapeInfo, 'c');

            return SHAPELIST(newShapeInfo);
        }
    }
}

#endif