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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 16.04.2019
//

#ifndef LIBND4J_LOOPKIND_H
#define LIBND4J_LOOPKIND_H

#include <helpers/shape.h>


namespace sd {

class SD_LIB_EXPORT LoopKind {
 public:
  enum Kind {
    RANK1,
    RANK2,
    RANK3,
    RANK4,
    RANK5,
    BROADCAST_SCALAR_X,
    BROADCAST_SCALAR_Y,
    BROADCAST_2D,    // Added new 2D broadcast case
    BROADCAST_3D,
    BROADCAST_4D,
    BROADCAST_5D,
    COMMON,
    SMALLARR2DX
  };


  static SD_INLINE Kind deduceKindOfLoopXZ(const LongType* xShapeInfo, const LongType* zShapeInfo);
  static SD_INLINE Kind deduceKindOfLoopXYZ(const LongType* xShapeInfo, const LongType* yShapeInfo,
                                            const LongType* zShapeInfo);
  static SD_INLINE Kind deduceKindOfLoopTadXZ(const LongType* xShapeInfo, const LongType* zShapeInfo,
                                              const LongType* tadShapeInfo);
  static SD_INLINE Kind deduceKindOfLoopTadXYZ(const LongType* xTadShapeInfo, const LongType* yTadShapeInfo,
                                               const LongType* zShapeInfo);
  static SD_INLINE Kind deduceKindOfLoopBroadcast(const LongType* xShapeInfo, const LongType* yShapeInfo,
                                                  const LongType* zShapeInfo);
};

//////////////////////////////////////////////////////////////////////////////
LoopKind::Kind LoopKind::deduceKindOfLoopXZ(const LongType* xShapeInfo, const LongType* zShapeInfo) {
  const int xRank = shape::rank(xShapeInfo);
  const bool shapesSame = shape::shapeEquals(xShapeInfo, zShapeInfo);

  // Handle rank-specific optimizations when shapes match
  if (shapesSame) {
    switch(xRank) {
      case 1: return RANK1;
      case 2: return RANK2;
      case 3: return RANK3;
      case 4: return RANK4;
      case 5: return RANK5;
      default: return COMMON;
    }
  }

  return COMMON;
}

LoopKind::Kind LoopKind::deduceKindOfLoopBroadcast(const LongType* xShapeInfo, const LongType* yShapeInfo,
                                                   const LongType* zShapeInfo) {
  auto xRank = shape::rank(xShapeInfo);
  auto yRank = shape::rank(yShapeInfo);
  auto zRank = shape::rank(zShapeInfo);

  auto xOrder = shape::order(xShapeInfo);
  auto yOrder = shape::order(yShapeInfo);
  auto zOrder = shape::order(zShapeInfo);

  // First check scalar broadcast cases
  if (yRank < 1 && xRank == yRank && xRank == zRank && xOrder == 'c' && yOrder == 'c' && zOrder == 'c' && xRank >= 2) {
    // Validate shapes are equal till last dim
    for (int e = 0; e < xRank - 1; e++) {
      if (xShapeInfo[e + 1] != yShapeInfo[e + 1]) break;
    }

    // Check if one shape has 1 as last dim
    auto detect = xShapeInfo[xRank] == 1 ? -1 : (yShapeInfo[xRank] == 1) ? 1 : 0;

    if (detect == 1) return BROADCAST_SCALAR_Y;
    else if (detect == -1) return BROADCAST_SCALAR_X;
  }

  // Check for 2D broadcasting cases
  if (zRank == 2) {
    const auto zShape = shape::shapeOf(zShapeInfo);

    // Case 1: Matrix + row vector broadcasting
    if ((xRank == 2 && yRank == 1) || (yRank == 2 && xRank == 1)) {
      const auto vecShape = xRank == 1 ? shape::shapeOf(xShapeInfo) : shape::shapeOf(yShapeInfo);
      if (vecShape[0] == zShape[1]) return BROADCAST_2D;
    }

    // Case 2: Matrix + column vector broadcasting (nx1)
    if (xRank == 2 && yRank == 2) {
      const auto xShape = shape::shapeOf(xShapeInfo);
      const auto yShape = shape::shapeOf(yShapeInfo);

      // Improved check for column vector - explicitly check for yShape[1] == 1
      if (yShape[0] == zShape[0] && yShape[1] == 1) {
        return BROADCAST_2D;
      }

      // Also check for row vector - explicitly check for yShape[0] == 1
      if (yShape[0] == 1 && yShape[1] == zShape[1]) {
        return BROADCAST_2D;
      }
    }

    // Case 3: Regular 2D broadcasting with matching ranks
    if (xRank == 2 && yRank == 2) {
      const auto xShape = shape::shapeOf(xShapeInfo);
      const auto yShape = shape::shapeOf(yShapeInfo);

      // Check if one array can broadcast to the other
      bool canBroadcast = (xShape[0] == zShape[0] || xShape[0] == 1) &&
                          (xShape[1] == zShape[1] || xShape[1] == 1) &&
                          (yShape[0] == zShape[0] || yShape[0] == 1) &&
                          (yShape[1] == zShape[1] || yShape[1] == 1);

      if (canBroadcast) return BROADCAST_2D;
    }
  }

  // Check higher dimension cases
  bool bNDLoopsRanks = (xRank == zRank && yRank <= xRank && yRank >= 2);

  int countUnityDimsInY = 0, countUnityDimsInX = 0;
  for (LongType i = 0; i < xRank; i++) {
    if (i < yRank) countUnityDimsInY += (1 == shape::sizeAt(yShapeInfo, i)) ? 1 : 0;
    countUnityDimsInX += (1 == shape::sizeAt(xShapeInfo, i)) ? 1 : 0;
  }


  if (3 == xRank) return BROADCAST_3D;
  if (4 == xRank) return BROADCAST_4D;
  if (5 == xRank) return BROADCAST_5D;


  return COMMON;
}
//////////////////////////////////////////////////////////////////////////////
LoopKind::Kind LoopKind::deduceKindOfLoopXYZ(const LongType* xShapeInfo, const LongType* yShapeInfo,
                                             const LongType* zShapeInfo) {
  const int xRank = shape::rank(xShapeInfo);
  const char xOrder = shape::order(xShapeInfo);
  const char yOrder = shape::order(yShapeInfo);
  const char zOrder = shape::order(zShapeInfo);

  // Check if all shapes match
  const bool shapesSame = shape::shapeEquals(xShapeInfo, yShapeInfo, zShapeInfo);

  // Handle rank-specific optimizations when shapes match
  if (shapesSame) {
    if (xRank == 1) return RANK1;
    if (xRank == 2) return RANK2;
    if (xRank == 3) return RANK3;
    if (xRank == 4) return RANK4;
    if (xRank == 5) return RANK5;
  }

  // Default case
  return COMMON;
}

//////////////////////////////////////////////////////////////////////////////
LoopKind::Kind LoopKind::deduceKindOfLoopTadXZ(const LongType* xShapeInfo, const LongType* zShapeInfo,
                                               const LongType* tadShapeInfo) {
  // Check for small array optimization first
  if (shape::rank(xShapeInfo) == 2 &&
      shape::length(tadShapeInfo) * shape::length(zShapeInfo) <= Environment::getInstance().elementwiseThreshold()) {
    return SMALLARR2DX;
  }

  // Handle rank-specific optimizations
  switch(shape::rank(tadShapeInfo)) {
    case 1: return RANK1;
    case 2: return RANK2;
    case 3: return RANK3;
    case 4: return RANK4;
    case 5: return RANK5;
    default: return COMMON;
  }
}
//////////////////////////////////////////////////////////////////////////////
LoopKind::Kind LoopKind::deduceKindOfLoopTadXYZ(const LongType* xTadShapeInfo, const LongType* yTadShapeInfo,
                                                const LongType* zShapeInfo) {
  // both tad shapes are the same, but strides may be different
  const int tadRank = shape::rank(xTadShapeInfo);

  // Handle rank-specific optimizations
  switch(tadRank) {
    case 1: return RANK1;
    case 2: return RANK2;
    case 3: return RANK3;
    case 4: return RANK4;
    case 5: return RANK5;
    default: return COMMON;
  }
}

}  // namespace sd
#endif  // LIBND4J_LOOPKIND_H
