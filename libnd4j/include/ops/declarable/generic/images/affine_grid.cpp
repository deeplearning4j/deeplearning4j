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
// @author Eclipse Deeplearning4j Development Team
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_affine_grid)

#include <ops/declarable/headers/images.h>
#include <ops/declarable/helpers/affine_grid.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(affine_grid, 2, 1, false, 0, -2) {
    auto theta = INPUT_VARIABLE(0);  // [N, 2, 3] or [N, 3, 4]
    auto size = INPUT_VARIABLE(1);   // [N, C, H, W] or [N, C, D, H, W]
    auto output = OUTPUT_VARIABLE(0);

    bool alignCorners = block.numI() > 0 ? INT_ARG(0) != 0 : false;

    // Validate theta shape
    REQUIRE_TRUE(theta->rankOf() == 3, 0,
                 "affine_grid: theta must be rank 3, got rank %i", theta->rankOf());

    auto thetaDim1 = theta->sizeAt(1);
    auto thetaDim2 = theta->sizeAt(2);

    // Check for 2D or 3D case
    bool is2D = (thetaDim1 == 2 && thetaDim2 == 3);
    bool is3D = (thetaDim1 == 3 && thetaDim2 == 4);

    REQUIRE_TRUE(is2D || is3D, 0,
                 "affine_grid: theta must have shape [N,2,3] for 2D or [N,3,4] for 3D, got [%lld,%lld,%lld]",
                 theta->sizeAt(0), thetaDim1, thetaDim2);

    helpers::affineGrid(block.launchContext(), theta, size, alignCorners, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(affine_grid) {
    auto thetaShape = inputShape->at(0);
    auto sizeShape = inputShape->at(1);

    auto theta = INPUT_VARIABLE(0);
    auto size = INPUT_VARIABLE(1);

    sd::LongType batchSize = shape::sizeAt(thetaShape, static_cast<sd::LongType>(0));
    auto thetaDim1 = shape::sizeAt(thetaShape, static_cast<sd::LongType>(1));

    bool is2D = (thetaDim1 == 2);

    std::vector<sd::LongType> outputShape;

    if (is2D) {
        // 2D case: output is [N, H, W, 2]
        sd::LongType H = size->e<sd::LongType>(2);
        sd::LongType W = size->e<sd::LongType>(3);
        outputShape = {batchSize, H, W, 2};
    } else {
        // 3D case: output is [N, D, H, W, 3]
        sd::LongType D = size->e<sd::LongType>(2);
        sd::LongType H = size->e<sd::LongType>(3);
        sd::LongType W = size->e<sd::LongType>(4);
        outputShape = {batchSize, D, H, W, 3};
    }

    auto desc = new ShapeDescriptor(sd::DataType::FLOAT32, 'c', outputShape);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
}

DECLARE_TYPES(affine_grid) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
