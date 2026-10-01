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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_smooth_quant) || NOT_EXCLUDED(OP_fused_norm_quantize)
#include <ops/declarable/helpers/symmetric_quant.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// Whether scale holds one value per row of input: input's shape without the
// last axis, or with the last axis kept as a unit extent.
static bool symmetricRowScaleShape(NDArray* input, NDArray* scale) {
  const int rows = input->rankOf() - 1;
  const int scaleRank = scale->rankOf();
  if (scaleRank != rows && !(scaleRank == rows + 1 && scale->sizeAt(rows) == 1)) return false;
  for (int axis = 0; axis < rows; ++axis) {
    if (scale->sizeAt(axis) != input->sizeAt(axis)) return false;
  }
  return true;
}

void symmetricRowScale(LaunchContext* context, NDArray* input, DataType quantizedType, NDArray* scale) {
  if (!(DataTypeUtils::isR(quantizedType) || DataTypeUtils::isZ(quantizedType)) || DataTypeUtils::isU(quantizedType)) {
    const std::string message =
        "symmetricRowScale: " + DataTypeUtils::asString(quantizedType) + " has no symmetric quantization grid";
    THROW_EXCEPTION(message.c_str());
  }
  if (!DataTypeUtils::isR(input->dataType()) || scale->dataType() != input->dataType()) {
    const std::string message = "symmetricRowScale: input must be floating and scale of its data type, got " +
                                DataTypeUtils::asString(input->dataType()) + " and " +
                                DataTypeUtils::asString(scale->dataType());
    THROW_EXCEPTION(message.c_str());
  }
  if (input->rankOf() < 1 || !symmetricRowScaleShape(input, scale)) {
    const std::string message = "symmetricRowScale: scale shape " + ShapeUtils::shapeAsString(scale) +
                                " does not hold one scale per row of input shape " + ShapeUtils::shapeAsString(input);
    THROW_EXCEPTION(message.c_str());
  }
  if (input->isEmpty()) {
    // Rows without elements have no magnitude, so their scale is zero.
    if (!scale->isEmpty()) scale->nullify();
    return;
  }
  const std::vector<LongType> lastAxis = {input->rankOf() - 1};
  // A scale of input's rank keeps the reduced axis as a unit extent.
  input->reduceAlongDimension(reduce::AMax, scale, &lastAxis, scale->rankOf() == input->rankOf());
  scale->applyScalar(scalar::Divide, DataTypeUtils::max(quantizedType), scale);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
