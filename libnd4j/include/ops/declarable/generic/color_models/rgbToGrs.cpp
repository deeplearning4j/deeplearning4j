/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
// 

#include <ops/declarable/headers/common.h>
#include <ops/declarable/CustomOperations.h>  
#include <ops/declarable/helpers/rgbToGrs_conf_op.h>
#include <ops/declarable/headers/rgbToGrs.h>

#if NOT_EXCLUDED(OP_rgb_to_grs)

namespace nd4j {
namespace ops {
namespace helpers{
    CUSTOM_OP_IMPL(rgb_to_grs, 1, 1, false, 0, 0) {
        const auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);
        if (input->isEmpty()){
            return Status::OK();
        }
        const int rank = input->rankOf();
        REQUIRE_TRUE(rank >= 1, 0, "RGBtoGrayScale: Fails to meet the rank requirement: %i >= 1 ", rank);
        const int dimLast = input->sizeAt(rank - 1);
        REQUIRE_TRUE(dimLast == 3, 0, "RGBtoGrayScale: operation expects 3 channels R, B, G), but received %i", dimLast);
        rgb_to_grs(block.launchContext(), input,  output, dimLast);
        return Status::OK();
    }
    DECLARE_TYPES(rgb_to_grs) {
        getOpDescriptor()->setAllowedInputTypes({ALL_INTS})
                         ->setAllowedInputTypes({ALL_FLOATS})
                         ->setSameMode(true);
    }
// TODO make shape for output
    // DECLARE_SHAPE_FN(rgb_to_grs) {
    
    // we always give out C order here
    // return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', {dim0 / (blockSize * blockSize), oH, oW, inputShapeInfo[4]}));
    // }
}
}
}
#endif
