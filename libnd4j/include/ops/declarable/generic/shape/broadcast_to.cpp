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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 03.09.2018
// Rewritten for CUDA graph replay safety and correct host/device sync.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_broadcast_to)

#include <ops/declarable/headers/shape.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// Read the target shape from input[1], syncing to host first.
// Returns the resolved output shape after applying numpy broadcast rules
// against the input's shape. This is shared between the op impl and
// the shape function to avoid divergence.
static std::vector<LongType> resolveTargetShape(
    const LongType* inputShapeInfo, NDArray* shapeArray) {

  // Force host sync — the shape tensor may be device-authoritative after
  // upstream ops (e.g. Where shape function) called syncToHost on their
  // inputs. Without this, getBufferAsVector reads stale device data,
  // producing wrong target shapes like [1,1] instead of [1,32].
  shapeArray->syncToHost();

  const LongType inputRank = shape::rank(inputShapeInfo);
  const LongType shapeLen = shapeArray->lengthOf();

  std::vector<LongType> outShape;

  if (shapeArray->isScalar()) {
    LongType val = shapeArray->e<LongType>(0);
    outShape.push_back(val);
  } else {
    outShape = shapeArray->getBufferAsVector<LongType>();
  }

  // Apply numpy broadcast rules (ONNX Expand semantics):
  //   result_dim = max(input_dim, target_dim) when one of them is 1 or ≤0
  for (LongType i = 1; i <= inputRank && i <= static_cast<LongType>(outShape.size()); ++i) {
    LongType& dim = outShape[outShape.size() - i];
    LongType inputDim = inputShapeInfo[inputRank + 1 - i];
    if (dim <= 0) {
      // -1 or 0 means keep input dimension
      dim = inputDim;
    } else if (dim == 1 && inputDim > 1) {
      // target=1, input>1: numpy broadcast = keep input (no shrinking)
      dim = inputDim;
    }
    // else: dim>1 and inputDim==1 → broadcast input to dim (normal case)
    // else: dim==inputDim → no-op
  }

  // Sync shape tensor back to device so subsequent GPU kernels see current data.
  // This prevents the "host-authoritative after syncToHost" stale-buffer problem
  // that cascades to downstream ops using this tensor.
  shapeArray->syncToDevice();

  return outShape;
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(broadcast_to, 2, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto shapeArr = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  const int inputRank = input->rankOf();
  const int shapeRank = shapeArr->rankOf();
  const LongType shapeLen = shapeArr->lengthOf();

  REQUIRE_TRUE(shapeRank <= 1, 0,
               "BROADCAST_TO op: rank of shape array should be <= 1, but got %i instead!", shapeRank);
  REQUIRE_TRUE(inputRank <= shapeLen, 0,
               "BROADCAST_TO op: rank of input (%i) should be <= length of shape array (%lli)!",
               inputRank, shapeLen);

  auto outShape = resolveTargetShape(input->shapeInfo(), shapeArr);

  for (int i = 1; i <= inputRank; ++i) {
    LongType inputDim = input->sizeAt(inputRank - i);
    LongType targetDim = outShape[outShape.size() - i];
    REQUIRE_TRUE(inputDim == targetDim || inputDim == 1, 0,
                 "BROADCAST_TO op: input shape %s can't be broadcasted to shape %s "
                 "(dimension %i: input=%lli target=%lli)",
                 ShapeUtils::shapeAsString(input).c_str(),
                 ShapeUtils::shapeAsString(outShape).c_str(),
                 i, inputDim, targetDim);
  }

  input->tile(*output);
  return Status::OK;
}

DECLARE_TYPES(broadcast_to) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(broadcast_to) {
  auto inputShapeInfo = inputShape->at(0);
  auto shapeArr = INPUT_VARIABLE(1);

  const LongType inputRank = shape::rank(inputShapeInfo);
  const LongType shapeRank = shapeArr->rankOf();
  const LongType shapeLen = shapeArr->lengthOf();

  REQUIRE_TRUE(shapeRank <= 1, 0,
               "BROADCAST_TO op: rank of shape array should be <= 1, but got %i instead!", shapeRank);
  REQUIRE_TRUE(inputRank <= shapeLen, 0,
               "BROADCAST_TO op: rank of input (%lli) should be <= length of shape array (%lli)!",
               inputRank, shapeLen);

  auto outShape = resolveTargetShape(inputShapeInfo, shapeArr);

  // DEBUG: trace shape resolution
  printf("BROADCAST_TO SHAPE_FN: inputRank=%lld inputShape=[", inputRank);
  for (LongType i = 0; i < inputRank; i++) printf("%s%lld", i?",":"", inputShapeInfo[i+1]);
  printf("] shapeLen=%lld target=[", shapeLen);
  for (size_t i = 0; i < outShape.size(); i++) printf("%s%lld", i?",":"", outShape[i]);
  printf("]\n");
  fflush(stdout);

  for (LongType i = 1; i <= inputRank; ++i) {
    LongType inputDim = inputShapeInfo[inputRank + 1 - i];
    LongType targetDim = outShape[outShape.size() - i];
    REQUIRE_TRUE(inputDim == targetDim || inputDim == 1, 0,
                 "BROADCAST_TO op: input shape %s can't be broadcasted to shape %s",
                 ShapeUtils::shapeAsString(inputShapeInfo).c_str(),
                 ShapeUtils::shapeAsString(outShape).c_str());
  }

  auto outShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inputShapeInfo), shape::order(inputShapeInfo), outShape);
  return SHAPELIST(outShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
