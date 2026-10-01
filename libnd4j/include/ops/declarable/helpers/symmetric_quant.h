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

#ifndef LIBND4J_SYMMETRIC_QUANT_H
#define LIBND4J_SYMMETRIC_QUANT_H

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_smooth_quant) || NOT_EXCLUDED(OP_fused_norm_quantize)
#include <array/DataTypeUtils.h>
#include <helpers/ShapeUtils.h>
#include <helpers/shape.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/reproducible_math.h>

#include <string>

namespace sd {
namespace ops {
namespace helpers {

// Symmetric quantization onto the grid of a target data type Q:
//   q = snap(clamp(x / scale, -qmax, qmax)),  qmax = DataTypeUtils::max(Q).
// The grid is symmetric, so an INT8 target spans [-127, 127]. Grids are policies:
//  - SymmetricIntegerGrid rounds to the nearest integer, ties to even (the
//    round-to-nearest-even of cvt.rni.sat), and maps NaN to zero;
//  - SymmetricFloatingGrid keeps the clamped value and leaves the rounding to the
//    storage conversion (round to nearest even for FP8), so NaN stays NaN.
// A zero scale quantizes every value to zero (the scale of an all-zero row).
struct SymmetricIntegerGrid {
  template <typename AccT>
  SD_HOST_DEVICE SD_INLINE static AccT snap(AccT value) {
    return math::sd_isnan<AccT>(value) ? static_cast<AccT>(0) : math::sd_rint<AccT, AccT>(value);
  }
};

struct SymmetricFloatingGrid {
  template <typename AccT>
  SD_HOST_DEVICE SD_INLINE static AccT snap(AccT value) {
    return value;
  }
};

template <typename Grid, typename AccT>
SD_HOST_DEVICE SD_INLINE AccT symmetricQuantizeValue(AccT value, AccT scale, AccT bound) {
  if (scale == static_cast<AccT>(0)) return static_cast<AccT>(0);
  const AccT scaled = reproducible::divide<AccT>(value, scale);
  // NaN fails both comparisons and reaches the grid unchanged.
  const AccT clamped = scaled > bound ? bound : (scaled < -bound ? -bound : scaled);
  return Grid::template snap<AccT>(clamped);
}

// Offset of the scale element that applies to an input element: scale broadcasts
// against the input by trailing-axis alignment (numpy rules), so a [rows, 1]
// scale applies one scale per row and a scalar applies everywhere.
SD_HOST_DEVICE SD_INLINE LongType symmetricScaleOffset(const LongType* inputCoords, int inputRank,
                                                       const LongType* scaleShape, const LongType* scaleStrides,
                                                       int scaleRank) {
  LongType offset = 0;
  for (int axis = 0; axis < scaleRank; ++axis) {
    if (scaleShape[axis] != 1) offset += inputCoords[inputRank - scaleRank + axis] * scaleStrides[axis];
  }
  return offset;
}

// Quantizes input onto quantized's grid: quantized = q(input, scale). quantized
// has input's shape and a signed integer or floating data type; scale has a
// floating data type and a shape that broadcasts to input's. Values are computed
// in the input's aggregate type and converted once into quantized's data type.
SD_LIB_HIDDEN void symmetricQuantize(LaunchContext* context, NDArray* input, NDArray* scale, NDArray* quantized);

// One scale per row of input's last axis: max|row| / qmax of quantizedType, in
// scale's data type. scale has input's data type and input's shape without the
// last axis, or with the last axis kept as a unit extent. A row of zeros, or a
// row without elements, gets scale zero, which symmetricQuantize maps to zero.
SD_LIB_HIDDEN void symmetricRowScale(LaunchContext* context, NDArray* input, DataType quantizedType, NDArray* scale);

// Checks that quantizedType has a symmetric grid (signed integer or floating)
// whose bound the accumulator AccT represents exactly, so a clamp in AccT never
// yields a value the target cannot hold.
template <typename AccT>
SD_INLINE AccT symmetricGridBound(DataType quantizedType) {
  if (!(DataTypeUtils::isR(quantizedType) || DataTypeUtils::isZ(quantizedType)) || DataTypeUtils::isU(quantizedType)) {
    const std::string message =
        "symmetricQuantize: " + DataTypeUtils::asString(quantizedType) + " has no symmetric quantization grid";
    THROW_EXCEPTION(message.c_str());
  }
  const double qmax = DataTypeUtils::max(quantizedType);
  const AccT bound = static_cast<AccT>(qmax);
  if (static_cast<double>(bound) != qmax) {
    const std::string message = "symmetricQuantize: the bound of " + DataTypeUtils::asString(quantizedType) +
                                " is not exact in the accumulator " +
                                DataTypeUtils::asString(DataTypeUtils::fromT<AccT>());
    THROW_EXCEPTION(message.c_str());
  }
  return bound;
}

// Checks the operands of symmetricQuantize: floating input and scale, a scale
// that broadcasts to input by trailing-axis alignment, and a quantized array of
// input's shape.
SD_INLINE void symmetricQuantizeCheck(NDArray* input, NDArray* scale, NDArray* quantized) {
  if (!DataTypeUtils::isR(input->dataType()) || !DataTypeUtils::isR(scale->dataType())) {
    const std::string message = "symmetricQuantize: input and scale must be floating, got " +
                                DataTypeUtils::asString(input->dataType()) + " and " +
                                DataTypeUtils::asString(scale->dataType());
    THROW_EXCEPTION(message.c_str());
  }
  const int inputRank = input->rankOf();
  const int scaleRank = scale->rankOf();
  bool broadcasts = scaleRank <= inputRank;
  for (int axis = 0; broadcasts && axis < scaleRank; ++axis) {
    const LongType extent = scale->sizeAt(axis);
    broadcasts = extent == 1 || extent == input->sizeAt(inputRank - scaleRank + axis);
  }
  if (!broadcasts) {
    const std::string message = "symmetricQuantize: scale shape " + ShapeUtils::shapeAsString(scale) +
                                " does not broadcast to input shape " + ShapeUtils::shapeAsString(input);
    THROW_EXCEPTION(message.c_str());
  }
  if (!quantized->isSameShape(input)) {
    const std::string message = "symmetricQuantize: quantized shape " + ShapeUtils::shapeAsString(quantized) +
                                " differs from input shape " + ShapeUtils::shapeAsString(input);
    THROW_EXCEPTION(message.c_str());
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
#endif  // LIBND4J_SYMMETRIC_QUANT_H
