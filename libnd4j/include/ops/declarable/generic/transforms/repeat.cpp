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
#if NOT_EXCLUDED(OP_repeat)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
		// here iArgs is int vector of repeats at the beginning and last element in iArgs is dimension
		CUSTOM_OP_IMPL(repeat, 1, 1, true, 0, -1) {			

			NDArray<T>* x   = INPUT_VARIABLE(0);
            NDArray<T>* ret = OUTPUT_VARIABLE(0);

			x->repeat(block.getIArguments()->back(), *ret);

			return Status::OK();
        }
		
        DECLARE_SHAPE_FN(repeat) {                               
            
            NDArray<T>* x   = INPUT_VARIABLE(0);
            auto argumets = block.getIArguments();
            int argsSize = argumets->size();
            int dimension = (*argumets)[argsSize-1];
            auto repeats = *argumets;
            repeats.pop_back();
            
            auto outShape = ShapeUtils<T>::evalRepeatShape(dimension, ArrayUtils::toLongVector(repeats), *x);
            int rank = outShape.size();

            Nd4jLong* newShapeInfo = nullptr; 
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong); 
            newShapeInfo[0] = rank;
            std::copy(outShape.begin(), outShape.end(), newShapeInfo+1);
            shape::updateStrides(newShapeInfo, x->ordering());

            return SHAPELIST(newShapeInfo);
        }
    }
}

#endif