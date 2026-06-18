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
// Created by raver119 on 24/09/18.
//
#include <array/NDArrayList.h>
#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>
#include <array/ArrayOptions.h>
#include <helpers/shape.h>

#if NOT_EXCLUDED(OP_Where)
#include <ops/declarable/helpers/where.h>

namespace sd {
namespace ops {
namespace helpers {
inline bool evaluateConditionValue(NDArray& condition, LongType index) {
  switch (condition.dataType()) {
    case DataType::BOOL:
    case DataType::INT8:
      return condition.e<int8_t>(index) != 0;
    case DataType::UINT8:
      return condition.e<uint8_t>(index) != 0;
    case DataType::INT16:
      return condition.e<int16_t>(index) != 0;
    case DataType::INT32:
      return condition.e<int32_t>(index) != 0;
    case DataType::INT64:
      return condition.e<LongType>(index) != 0;
    case DataType::FLOAT32:
      return condition.e<float>(index) != 0.0f;
    case DataType::DOUBLE:
      return condition.e<double>(index) != 0.0;
    default:
      return condition.e<int32_t>(index) != 0;
  }
}

template <typename T>
static void __where(NDArray &condition, NDArray &output, memory::Workspace *workspace) {
  // Early return if output is empty - check multiple ways for robustness
  // 1. Check isEmpty() which looks at ARRAY_EMPTY flag
  // 2. Check lengthOf() == 0 for zero-size arrays
  // 3. Check ARRAY_EMPTY flag directly as backup
  bool outputIsEmpty = output.isEmpty() || output.lengthOf() == 0;
  if (!outputIsEmpty) {
    // Direct flag check as additional safety
    auto* outShapeInfo = output.shapeInfo();
    if (outShapeInfo != nullptr) {
      outputIsEmpty = ArrayOptions::hasPropertyBitSet(outShapeInfo, ARRAY_EMPTY);
    }
  }
  if (outputIsEmpty) {
    return;
  }

  // Early return if condition is empty or invalid
  bool conditionIsEmpty = condition.isEmpty() || condition.lengthOf() == 0;
  if (!conditionIsEmpty) {
    auto* condShapeInfo = condition.shapeInfo();
    if (condShapeInfo != nullptr) {
      conditionIsEmpty = ArrayOptions::hasPropertyBitSet(condShapeInfo, ARRAY_EMPTY);
    }
  }
  if (conditionIsEmpty) {
    return;
  }

  NDArrayList list(0, true);
  sd::LongType cnt = 0;

  // Ensure condition data is accessible before iteration
  condition.syncToHost();

  for (sd::LongType e = 0; e < condition.lengthOf(); e++) {
    sd::LongType coords[SD_MAX_RANK];

    INDEX2COORDS(e, condition.rankOf(), condition.shapeOf(), coords);
    sd::LongType offset;
    COORDS2INDEX(condition.rankOf(), shape::stride(condition.shapeInfo()), coords, offset);

    if (evaluateConditionValue(condition, offset)) {
      std::vector<sd::LongType> arrShape = {1, condition.rankOf()};
      auto array = NDArrayFactory::create_('c', arrShape, output.dataType(), output.getContext());
      for (sd::LongType f = 0; f < condition.rankOf(); f++)  {
        array->p(f, (T)coords[f]);
      }
      list.write(cnt++, array);
    }
  }

  // Only assign if we have elements to assign
  if (list.elements() > 0) {
    auto s = list.stack();
    output.assign(s);
    delete s;
  }
}
BUILD_SINGLE_TEMPLATE( void __where, (NDArray & condition, NDArray &output, memory::Workspace *workspace),
                      SD_COMMON_TYPES);

void _where(sd::LaunchContext *context, NDArray &condition, NDArray &output, memory::Workspace *workspace) {
  // Early return if output is empty - check multiple ways for robustness
  bool outputIsEmpty = output.isEmpty() || output.lengthOf() == 0;
  if (!outputIsEmpty) {
    auto* outShapeInfo = output.shapeInfo();
    if (outShapeInfo != nullptr) {
      outputIsEmpty = ArrayOptions::hasPropertyBitSet(outShapeInfo, ARRAY_EMPTY);
    }
  }
  if (outputIsEmpty) {
    return;
  }

  // Early return if condition is empty or invalid
  bool conditionIsEmpty = condition.isEmpty() || condition.lengthOf() == 0;
  if (!conditionIsEmpty) {
    auto* condShapeInfo = condition.shapeInfo();
    if (condShapeInfo != nullptr) {
      conditionIsEmpty = ArrayOptions::hasPropertyBitSet(condShapeInfo, ARRAY_EMPTY);
    }
  }
  if (conditionIsEmpty) {
    return;
  }

  condition.syncToHost();
  BUILD_SINGLE_SELECTOR(output.dataType(), __where, (condition, output, workspace), SD_COMMON_TYPES);
  output.syncToDevice();
}

//////////////////////////////////////////////////////////////////////////
// Helper for 3-input where (condition, x, y) - CPU implementation
void _whereElementWise(LaunchContext* context, NDArray& condition, NDArray& x, NDArray& y,
                        NDArray& output) {
  // Early return if output is empty
  if (output.isEmpty() || output.lengthOf() == 0) {
    return;
  }

  // Sync inputs to host for CPU processing
  condition.syncToHost();
  x.syncToHost();
  y.syncToHost();

  auto zLen = output.lengthOf();
  auto condRank = condition.rankOf();
  auto xRank = x.rankOf();
  auto yRank = y.rankOf();
  auto zRank = output.rankOf();

  auto condShape = condition.shapeOf();
  auto xShape = x.shapeOf();
  auto yShape = y.shapeOf();
  auto zShape = output.shapeOf();

  auto condStride = shape::stride(condition.shapeInfo());
  auto xStride = shape::stride(x.shapeInfo());
  auto yStride = shape::stride(y.shapeInfo());
  auto zStride = shape::stride(output.shapeInfo());

  bool sameShapes = condition.isSameShape(x) && x.isSameShape(y) && x.isSameShape(output);

  for (LongType i = 0; i < zLen; i++) {
    LongType zCoords[SD_MAX_RANK];
    LongType zOffset, condOffset, xOffset, yOffset;

    INDEX2COORDS(i, zRank, zShape, zCoords);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    if (sameShapes) {
      // Fast path: all arrays have same shape
      COORDS2INDEX(condRank, condStride, zCoords, condOffset);
      COORDS2INDEX(xRank, xStride, zCoords, xOffset);
      COORDS2INDEX(yRank, yStride, zCoords, yOffset);
    } else {
      // Slow path: handle broadcasting
      LongType condCoords[SD_MAX_RANK];
      LongType xCoords[SD_MAX_RANK];
      LongType yCoords[SD_MAX_RANK];

      // Map z coordinates to each input with broadcasting
      for (LongType d = 0; d < zRank; d++) {
        LongType condDim = d - (zRank - condRank);
        LongType xDim = d - (zRank - xRank);
        LongType yDim = d - (zRank - yRank);

        if (condDim >= 0 && condDim < condRank) {
          condCoords[condDim] = (condShape[condDim] == 1) ? 0 : zCoords[d];
        }
        if (xDim >= 0 && xDim < xRank) {
          xCoords[xDim] = (xShape[xDim] == 1) ? 0 : zCoords[d];
        }
        if (yDim >= 0 && yDim < yRank) {
          yCoords[yDim] = (yShape[yDim] == 1) ? 0 : zCoords[d];
        }
      }

      COORDS2INDEX(condRank, condStride, condCoords, condOffset);
      COORDS2INDEX(xRank, xStride, xCoords, xOffset);
      COORDS2INDEX(yRank, yStride, yCoords, yOffset);
    }

    // Select x or y based on condition
    bool condVal = evaluateConditionValue(condition, condOffset);
    if (output.isR()) {
#ifdef HAS_DOUBLE
      output.p(zOffset, condVal ? x.e<double>(xOffset) : y.e<double>(yOffset));
#elif defined(HAS_FLOAT32)
      output.p(zOffset, condVal ? x.e<float>(xOffset) : y.e<float>(yOffset));
#endif
    } else {
      output.p(zOffset, condVal ? x.e<LongType>(xOffset) : y.e<LongType>(yOffset));
    }
  }

  output.syncToDevice();
}

//////////////////////////////////////////////////////////////////////////
// Helper for TAD-based where - CPU implementation
void _whereTad(LaunchContext* context, NDArray& condition, NDArray& x, NDArray& y,
               NDArray& output, const std::vector<LongType>& axis) {
  // Early return if output is empty
  if (output.isEmpty() || output.lengthOf() == 0) {
    return;
  }

  condition.syncToHost();

  std::vector<LongType> dimsToExclude;
  for (LongType i = 0; i < x.rankOf(); i++) {
    bool found = false;
    for (auto& a : axis) {
      if (a == i) {
        found = true;
        break;
      }
    }
    if (!found) dimsToExclude.push_back(i);
  }

  auto tadsX = x.allTensorsAlongDimension(dimsToExclude);
  auto tadsY = y.allTensorsAlongDimension(dimsToExclude);
  auto tadsZ = output.allTensorsAlongDimension(dimsToExclude);

  for (LongType e = 0; e < tadsX.size(); e++) {
    bool condVal = evaluateConditionValue(condition, e);
    if (condVal) {
      tadsZ.at(e)->assign(tadsX.at(e));
    } else {
      tadsZ.at(e)->assign(tadsY.at(e));
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Count true elements - CPU implementation
LongType countTrue(LaunchContext* context, NDArray& condition) {
  if (condition.isEmpty() || condition.lengthOf() == 0) {
    return 0;
  }

  condition.syncToHost();

  LongType count = 0;
  for (LongType i = 0; i < condition.lengthOf(); i++) {
    if (evaluateConditionValue(condition, i)) {
      count++;
    }
  }
  return count;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
