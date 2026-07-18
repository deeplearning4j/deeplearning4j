/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Async gradient-checkpoint offload ops.
//
// checkpoint_offload_d2h  — copy a device tensor to pinned host memory (async D2H)
// checkpoint_prefetch_h2d — copy a host tensor back to device memory  (async H2D)
//

#include <system/op_boilerplate.h>

// ============================================================================
// checkpoint_offload_d2h
// ============================================================================
#if NOT_EXCLUDED(OP_checkpoint_offload_d2h)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/checkpoint_offload.h>
#include <helpers/ConstantShapeHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(checkpoint_offload_d2h, 1, 1, false, 0, 2) {
    auto src  = INPUT_VARIABLE(0);
    auto dst  = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(src->rankOf() >= 0, 0,
                 "checkpoint_offload_d2h: input must be a valid tensor");
    REQUIRE_TRUE(src->lengthOf() == dst->lengthOf(), 0,
                 "checkpoint_offload_d2h: input and output must have the same number of elements, "
                 "got %lld vs %lld", src->lengthOf(), dst->lengthOf());

    int usePinned = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
    int streamId  = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

    helpers::checkpointOffloadD2H(block.launchContext(), src, dst, usePinned, streamId);

    return sd::Status::OK;
}

DECLARE_TYPES(checkpoint_offload_d2h) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS, {DataType::BOOL}});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS, {DataType::BOOL}});
}

DECLARE_SHAPE_FN(checkpoint_offload_d2h) {
    // Output has the same shape and data type as the input.
    auto inShape = inputShape->at(0);
    auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
            ArrayOptions::dataType(inShape),
            shape::order(inShape),
            shape::rank(inShape),
            shape::shapeOf(inShape));
    return SHAPELIST(outShape);
}

}  // namespace ops
}  // namespace sd

#endif  // OP_checkpoint_offload_d2h


// ============================================================================
// checkpoint_prefetch_h2d
// ============================================================================
#if NOT_EXCLUDED(OP_checkpoint_prefetch_h2d)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/checkpoint_offload.h>
#include <helpers/ConstantShapeHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(checkpoint_prefetch_h2d, 1, 1, false, 0, 1) {
    auto src = INPUT_VARIABLE(0);
    auto dst = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(src->rankOf() >= 0, 0,
                 "checkpoint_prefetch_h2d: input must be a valid tensor");
    REQUIRE_TRUE(src->lengthOf() == dst->lengthOf(), 0,
                 "checkpoint_prefetch_h2d: input and output must have the same number of elements, "
                 "got %lld vs %lld", src->lengthOf(), dst->lengthOf());

    int targetDeviceId = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    helpers::checkpointPrefetchH2D(block.launchContext(), src, dst, targetDeviceId);

    return sd::Status::OK;
}

DECLARE_TYPES(checkpoint_prefetch_h2d) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS, {DataType::BOOL}});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS, {DataType::BOOL}});
}

DECLARE_SHAPE_FN(checkpoint_prefetch_h2d) {
    // Output has the same shape and data type as the input.
    auto inShape = inputShape->at(0);
    auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
            ArrayOptions::dataType(inShape),
            shape::order(inShape),
            shape::rank(inShape),
            shape::shapeOf(inShape));
    return SHAPELIST(outShape);
}

}  // namespace ops
}  // namespace sd

#endif  // OP_checkpoint_prefetch_h2d
