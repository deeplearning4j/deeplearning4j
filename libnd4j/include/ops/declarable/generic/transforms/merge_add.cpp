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
// Created by raver119 on 24.11.17.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_mergeadd)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops  {

OP_IMPL(mergeadd, -1, 1, false) {
    
    REQUIRE_OK(this->validateInputDimensionsMatch(block));
        
    auto output = OUTPUT_VARIABLE(0);

    std::vector<const NDArray*> inArrs(block.width());
    
    for(int i = 0; i < block.width(); ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    helpers::mergeAdd(block.launchContext(), inArrs, *output);

    return Status::OK();
}
DECLARE_SYN(mergesum, mergeadd);
DECLARE_SYN(add_n, mergeadd);
DECLARE_SYN(addn, mergeadd);
DECLARE_SYN(accumulaten, mergeadd);
DECLARE_SYN(accumulate_n, mergeadd);

    DECLARE_TYPES(mergeadd) {
        getOpDescriptor()
                ->setAllowedInputTypes(sd::DataType::ANY)
                ->setAllowedOutputTypes(sd::DataType::ANY);
    }


    CUSTOM_OP_IMPL(mergeadd_bp, 2, 1, false, 0, 0) {

        auto inSize = block.width() - 1;

        REQUIRE_OK(this->validateInputDimensionsMatch(block));

        std::vector<NDArray*> outArrs(inSize);
        
        const auto gradient = INPUT_VARIABLE(inSize);

        for (int i = 0; i < inSize; ++i) {
            outArrs[i] = OUTPUT_VARIABLE(i);
        }
        helpers::mergeAddBp(block.launchContext(), *gradient, outArrs);

        return Status::OK();
    }

    DECLARE_TYPES(mergeadd_bp) {
        getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setAllowedOutputTypes(sd::DataType::ANY);
    }
    DECLARE_SHAPE_FN(mergeadd_bp) {

        const int numOfInArrs = block.width() - 1;

        auto shapeList = SHAPELIST();

        for (int e = 0; e < numOfInArrs; e++) {
            auto inShape = inputShape->at(e);
            shapeList->push_back(ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(ArrayOptions::dataType(inShape), shape::order(inShape), shape::shapeOf(inShape), shape::rank(inShape))));
        }

        return shapeList;
    }

}
}

#endif