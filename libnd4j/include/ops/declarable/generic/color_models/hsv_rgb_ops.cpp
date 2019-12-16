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

 
#include <ops/declarable/headers/color_models.h>
#include <ops/declarable/CustomOperations.h>  
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>

namespace nd4j {
    namespace ops {



        CONFIGURABLE_OP_IMPL(hsv_to_rgb, 1, 1, false, 0, 0) {

            auto input  = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            if (input->isEmpty())
                return Status::OK();

            const int rank = input->rankOf();
            const int dimC =  block.getIArguments()->size() > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;

            REQUIRE_TRUE(rank >= 1, 0, "HSVtoRGB: Fails to meet the requirement: %i >= 1 ", rank);
            REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "HSVtoRGB: operation expects 3 channels (H, S, V), but got %i instead", input->sizeAt(dimC));

            helpers::transform_hsv_rgb(block.launchContext(), input, output, dimC);

            return Status::OK();
        }

        CONFIGURABLE_OP_IMPL(rgb_to_hsv, 1, 1, false, 0, 0) {

            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            if (input->isEmpty())
                return Status::OK();

            const int rank = input->rankOf();
            const int dimC = block.getIArguments()->size() > 0 ? (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + rank) : rank - 1;

            REQUIRE_TRUE(rank >= 1, 0, "RGBtoHSV: Fails to meet the requirement: %i >= 1 ", rank);
            REQUIRE_TRUE(input->sizeAt(dimC) == 3, 0, "RGBtoHSV: operation expects 3 channels (H, S, V), but got %i instead", input->sizeAt(dimC));

            helpers::transform_rgb_hsv(block.launchContext(), input,  output, dimC);

            return Status::OK();
        }


        DECLARE_TYPES(hsv_to_rgb) {
            getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
                ->setSameMode(true);
        }

        DECLARE_TYPES(rgb_to_hsv) {
            getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
                ->setSameMode(true);
        }
    }
}
