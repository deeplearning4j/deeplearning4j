/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_grid_sample)

#include <ops/declarable/headers/images.h>
#include <ops/declarable/helpers/affine_grid.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(grid_sample, 2, 1, false, 0, -2) {
    auto input = INPUT_VARIABLE(0);  // [N, C, H, W] or [N, C, D, H, W]
    auto grid = INPUT_VARIABLE(1);   // [N, H_out, W_out, 2] or [N, D_out, H_out, W_out, 3]
    auto output = OUTPUT_VARIABLE(0);

    int mode = block.numI() > 0 ? INT_ARG(0) : 0;  // 0=bilinear, 1=nearest, 2=bicubic
    int paddingMode = block.numI() > 1 ? INT_ARG(1) : 0;  // 0=zeros, 1=border, 2=reflection
    bool alignCorners = block.numI() > 2 ? INT_ARG(2) != 0 : false;

    REQUIRE_TRUE(input->rankOf() == 4 || input->rankOf() == 5, 0,
                 "grid_sample: input must be rank 4 or 5, got rank %i", input->rankOf());

    REQUIRE_TRUE(grid->rankOf() == input->rankOf(), 0,
                 "grid_sample: grid rank must match input rank, got input rank %i and grid rank %i",
                 input->rankOf(), grid->rankOf());

    helpers::gridSample(block.launchContext(), input, grid, mode, paddingMode, alignCorners, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(grid_sample) {
    auto inShape = inputShape->at(0);
    auto gridShapeInfo = inputShape->at(1);

    auto inputRank = shape::rank(inShape);
    auto batchSize = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
    auto channels = shape::sizeAt(inShape, static_cast<sd::LongType>(1));

    std::vector<sd::LongType> outputShapeVec;

    if (inputRank == 4) {
        // 2D: output is [N, C, H_out, W_out]
        auto H_out = shape::sizeAt(gridShapeInfo, static_cast<sd::LongType>(1));
        auto W_out = shape::sizeAt(gridShapeInfo, static_cast<sd::LongType>(2));
        outputShapeVec = {batchSize, channels, H_out, W_out};
    } else {
        // 3D: output is [N, C, D_out, H_out, W_out]
        auto D_out = shape::sizeAt(gridShapeInfo, static_cast<sd::LongType>(1));
        auto H_out = shape::sizeAt(gridShapeInfo, static_cast<sd::LongType>(2));
        auto W_out = shape::sizeAt(gridShapeInfo, static_cast<sd::LongType>(3));
        outputShapeVec = {batchSize, channels, D_out, H_out, W_out};
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), 'c', outputShapeVec));
}

DECLARE_TYPES(grid_sample) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
