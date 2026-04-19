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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_floordiv)

#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/generic/helpers/BroadcastHelper.h>

namespace sd {
namespace ops {
BROADCASTABLE_OP_IMPL(floordiv, 0, 0) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  BROADCAST_CHECK_EMPTY(x, y, z);

  REQUIRE_TRUE(!y->isB(), 0, "FLOORDIV OP: you can't divide by bool array!");

  // Check for division by zero for scalar divisor - this is undefined behavior for integers
  // and will cause crashes. For scalar divisor, check the value directly.
  if (y->lengthOf() == 1) {
    // Sync to host to read the value
    y->syncToHost();
    bool isZero = false;
    switch (y->dataType()) {
      case DataType::INT64:
        isZero = y->e<sd::LongType>(0) == 0;
        break;
      case DataType::INT32:
        isZero = y->e<int>(0) == 0;
        break;
      case DataType::FLOAT32:
        isZero = y->e<float>(0) == 0.0f;
        break;
      case DataType::DOUBLE:
        isZero = y->e<double>(0) == 0.0;
        break;
      default:
        // For other types, use a generic check via double conversion
        isZero = y->e<double>(0) == 0.0;
        break;
    }
    REQUIRE_TRUE(!isZero, 0, "FLOORDIV OP: division by zero is not allowed! Divisor value is 0.");
  }

  // Fast path: same shape - skip BroadcastHelper dispatch overhead
  if (x->isSameShape(y)) {
    x->applyPairwiseTransform(pairwise::FloorDiv, y, z, nullptr);
    return Status::OK;
  }

  // Fast path: scalar or effectively-scalar (length 1) divisor
  const auto yLen = y->lengthOf();
  if (yLen == 1) {
    x->applyScalarArr(scalar::FloorDiv, y, z);
    return Status::OK;
  }

  // Fast path: 1D-like divisor broadcast along last dimension
  const auto xRank = x->rankOf();
  const auto yRank = y->rankOf();

  if (xRank > 1 && yLen == x->sizeAt(-1)) {
    bool compatible = true;
    for (int i = 0; i < yRank - 1; i++) {
      if (y->sizeAt(i) != 1) { compatible = false; break; }
    }
    if (compatible && (yRank == 1 || y->sizeAt(-1) == x->sizeAt(-1))) {
      std::vector<sd::LongType> dims = {xRank - 1};
      if (yRank > 1) {
        std::vector<sd::LongType> yShape = {yLen};
        auto yReshaped = y->reshape(y->ordering(), yShape, false);
        x->applyBroadcast(broadcast::FloorDiv, &dims, yReshaped, z);
        delete yReshaped;
      } else {
        x->applyBroadcast(broadcast::FloorDiv, &dims, y, z);
      }
      return Status::OK;
    }
  }

  auto tZ = BroadcastHelper::broadcastApply(
      BroadcastOpsTuple::custom(scalar::FloorDiv, pairwise::FloorDiv, broadcast::FloorDiv), x, y, z);
  if (tZ == nullptr)
    return Status::KERNEL_FAILURE;
  else if (tZ != z) {
    OVERWRITE_RESULT(tZ);
  }

  return Status::OK;
}

DECLARE_TYPES(floordiv) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes(0, INHERIT)
      ->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_TYPES(floordiv_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(floordiv_bp, 3, 2, false, 0, 0) {
  // PLEASE NOTE: we're just passing eps down the line here
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto epsNext = INPUT_VARIABLE(2);

  auto gradX = OUTPUT_VARIABLE(0);
  auto gradY = OUTPUT_VARIABLE(1);

  float zero = 0.0f;
  gradY->assign(zero);
  gradX->assign(zero);

  return Status::OK;
}

DECLARE_SHAPE_FN(floordiv_bp) {
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto e = inputShape->at(2);

  // eps always has shape of x
  // grad always has shape of y
  return SHAPELIST(CONSTANT(x), CONSTANT(y));
}
}  // namespace ops
}  // namespace sd

#endif
