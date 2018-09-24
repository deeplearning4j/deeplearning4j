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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 22.01.2018
//
#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_eye)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops {


    CUSTOM_OP_IMPL(eye, -2, 1, false, 0, -2) {

        helpers::eye(*OUTPUT_VARIABLE(0));

        return Status::OK();
    }


    DECLARE_SHAPE_FN(eye) {

        std::vector<int> params;
        if(block.width() == 0) {
            params = *block.getIArguments();
        }
        else {
            for (int i = 0; i < block.width(); i++) {
                auto input = INPUT_VARIABLE(i);
                REQUIRE_TRUE(input->rankOf() == 1, 0, "Inputs to eye should be 1D");
                for (int e = 0; e < input->lengthOf(); e++) {
                    params.emplace_back(input->e<int>(e));
                }
                delete input;
            }
        }


        REQUIRE_TRUE(params.size() > 0, 0, "Size not provided for eye op.");

        const bool ordered = (params[0] == -99 || params[0] == -102); // -99 :'c', -102 : 'f'
        if (!ordered)
            params.insert(params.begin(), -99);

        REQUIRE_TRUE(params.size() > 1, 0, "Size not provided for eye op.");

        Nd4jLong* outShapeInfo(nullptr);

        const int size = params.size();

        switch(size) {
        
            case 2:
                ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
                outShapeInfo[0] = 2;
                outShapeInfo[1] = params[1];
                outShapeInfo[2] = params[1];
                break;

            case 3:
                ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
                outShapeInfo[0] = 2;
                outShapeInfo[1] = params[1];
                outShapeInfo[2] = params[2];
                break;

            default:
                int rank = size-1;
                ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
                outShapeInfo[0] = rank;
                outShapeInfo[rank-1] = params[1];
                outShapeInfo[rank] = params[2];
                for(int i = 1; i < rank-1; ++i)
                    outShapeInfo[i] = params[i+2];
                break;
        }
        
        shape::updateStrides(outShapeInfo, static_cast<char>(-params[0]));
        
        return SHAPELIST(outShapeInfo);
    }


}
}

#endif