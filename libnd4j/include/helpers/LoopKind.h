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
    SMALLARR2DX,
    EWS1,
    EWSNONZERO,
    RANK1,
    RANK2,
    RANK3,
    RANK4,
    RANK5,
    X_EWSNONZERO,
    Y_EWSNONZERO,
    Z_EWSNONZERO,
    COMMON,
    BROADCAST_SCALAR_X,
    BROADCAST_SCALAR_Y,
    BROADCAST_3D,
    BROADCAST_4D,
    BROADCAST_5D
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
  const LongType xEws = shape::elementWiseStride(xShapeInfo);
  const LongType zEws = shape::elementWiseStride(zShapeInfo);

  const char xOrder = shape::order(xShapeInfo);
  const char zOrder = shape::order(zShapeInfo);

  LongType temp;
  const bool xVectorOrC = shape::isCommonVector(xShapeInfo, temp) || xOrder == 'c';
  const bool zVectorOrC = shape::isCommonVector(zShapeInfo, temp) || zOrder == 'c';
  const bool shapesSame = shape::shapeEquals(xShapeInfo, zShapeInfo);

  if (xEws == 1 && zEws == 1 && xOrder == zOrder
      && (shapesSame || xOrder == 'c')
      && !shape::isViewConst(xShapeInfo) && !shape::isViewConst(zShapeInfo)) {
    return EWS1;
  }
  if (xEws > 0 && zEws > 0
      && ((xOrder == zOrder && (shapesSame || xOrder == 'c'))
          || (xVectorOrC && zVectorOrC)) && !shape::isViewConst(xShapeInfo)
      && !shape::isViewConst(zShapeInfo)) {
    return EWSNONZERO;
  }
  if (xRank == 1 && shapesSame) {
    return RANK1;
  }
  if (xRank == 2 && shapesSame) {
    return RANK2;
  }
  if (xRank == 3 && shapesSame) {
    return RANK3;
  }
  if (xRank == 4 && shapesSame) {
    return RANK4;
  }
  if (xRank == 5 && shapesSame) {
    return RANK5;
  }
  if (xEws > 0 && xVectorOrC && !shape::isViewConst(xShapeInfo)) {
    return X_EWSNONZERO;
  }
  if (zEws > 0 && zVectorOrC && !shape::isViewConst(zShapeInfo)) {
    return Z_EWSNONZERO;
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

  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto yEws = shape::elementWiseStride(yShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  bool bNDLoopsRanks = (xRank == zRank && yRank <= xRank && yRank >= 2);

  int countUnityDimsInY = 0, countUnityDimsInX = 0;
  for (LongType i = 0; i < xRank; i++) {
    if (i < yRank) countUnityDimsInY += (1 == shape::sizeAt(yShapeInfo, i)) ? 1 : 0;
    countUnityDimsInX += (1 == shape::sizeAt(xShapeInfo, i)) ? 1 : 0;
  }

  bool bNotCommonVectorCase = (countUnityDimsInY != yRank - 1) && (countUnityDimsInX != xRank - 1);

  if (bNDLoopsRanks && bNotCommonVectorCase) {
    // case x[3,4,5] * y[1,4,5] = z[3,4,5] or reverse x[1,4,5] + y[3,4,5] = z[3,4,5]
    if (EWS1 == deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo) &&
        (1 == shape::sizeAt(yShapeInfo, static_cast<LongType>(0)) || 1 == shape::sizeAt(xShapeInfo, static_cast<LongType>(0)))) {
      return EWS1;
    }

    if (3 == xRank) return BROADCAST_3D;
    if (4 == xRank) return BROADCAST_4D;
    if (5 == xRank) return BROADCAST_5D;
  }

  if (xRank == yRank && xRank == zRank && xOrder == 'c' && yOrder == 'c' && zOrder == 'c' && xEws == 1 && yEws == 1 &&
      zEws == 1 && xRank >= 2) {
    // we validate that shapes are equal till the last dim
    for (int e = 0; e < xRank - 1; e++) {
      if (xShapeInfo[e + 1] != yShapeInfo[e + 1]) return COMMON;
    }

    // now, if one of the shapes has 1 as last dim
    auto detect = xShapeInfo[xRank] == 1 ? -1 : (yShapeInfo[xRank] == 1) ? 1 : 0;

    if (detect == 1)
      return BROADCAST_SCALAR_Y;
    else if (detect == -1)
      return BROADCAST_SCALAR_X;
  }

  return COMMON;
}

//////////////////////////////////////////////////////////////////////////////
LoopKind::Kind LoopKind::deduceKindOfLoopXYZ(const LongType* xShapeInfo, const LongType* yShapeInfo,
                                             const LongType* zShapeInfo) {
  const int xRank = shape::rank(xShapeInfo);
  const LongType xEws = shape::elementWiseStride(xShapeInfo);
  const LongType yEws = shape::elementWiseStride(yShapeInfo);
  const LongType zEws = shape::elementWiseStride(zShapeInfo);

  const char xOrder = shape::order(xShapeInfo);
  const char yOrder = shape::order(yShapeInfo);
  const char zOrder = shape::order(zShapeInfo);

  LongType temp;
  const bool xVectorOrC = shape::isCommonVector(xShapeInfo, temp) || xOrder == 'c';
  const bool yVectorOrC = shape::isCommonVector(yShapeInfo, temp) || yOrder == 'c';
  const bool zVectorOrC = shape::isCommonVector(zShapeInfo, temp) || zOrder == 'c';
  const bool shapesSame = shape::shapeEquals(xShapeInfo, yShapeInfo, zShapeInfo);

  if (xEws == 1 && yEws == 1 && zEws == 1 && xOrder == yOrder && xOrder == zOrder && (shapesSame || xOrder == 'c'))
    return EWS1;
  if (xEws > 0 && yEws > 0 && zEws > 0 &&
      ((xOrder == yOrder && xOrder == zOrder && (shapesSame || xOrder == 'c')) ||
       (xVectorOrC && yVectorOrC && zVectorOrC)))
    return EWSNONZERO;
  if (xRank == 1 && shapesSame) return RANK1;
  if (xRank == 2 && shapesSame) return RANK2;
  if (xRank == 3 && shapesSame) return RANK3;
  if (xRank == 4 && shapesSame) return RANK4;
  if (xRank == 5 && shapesSame) return RANK5;
  if (zEws > 0 && zVectorOrC) return Z_EWSNONZERO;
  return COMMON;
}

//////////////////////////////////////////////////////////////////////////////
LoopKind::Kind LoopKind::deduceKindOfLoopTadXZ(const LongType* xShapeInfo, const LongType* zShapeInfo,
                                               const LongType* tadShapeInfo) {
  const int xRank = shape::rank(xShapeInfo);
  const int tRank = shape::rank(tadShapeInfo);

  const LongType xEws = shape::elementWiseStride(xShapeInfo);
  const LongType tEws = shape::elementWiseStride(tadShapeInfo);
  const LongType zEws = shape::elementWiseStride(zShapeInfo);

  const char xOrder = shape::order(xShapeInfo);
  const char tOrder = shape::order(tadShapeInfo);
  const char zOrder = shape::order(zShapeInfo);

  const bool allC = (tOrder == zOrder && zOrder == 'c');

  LongType temp;
  const bool tVectorOrC = shape::isCommonVector(tadShapeInfo, temp) || tOrder == 'c';
  const bool zVectorOrC = shape::isCommonVector(zShapeInfo, temp) || zOrder == 'c';


  if (shape::length(tadShapeInfo) * shape::length(zShapeInfo) <= Environment::getInstance().elementwiseThreshold() &&
      xEws == 1 && xOrder == 'c' && xRank == 2 && tEws > 1 && zEws == 1 && (allC || (tVectorOrC && zVectorOrC)))
    return SMALLARR2DX;
  if (tEws == 1 && zEws == 1 && (allC || (tVectorOrC && zVectorOrC))) return EWS1;
  if (tEws > 0 && zEws > 0 && (allC || (tVectorOrC && zVectorOrC))) return EWSNONZERO;
  if (tRank == 1 && zEws == 1 && zVectorOrC) return RANK1;
  if (tRank == 2 && zEws == 1 && zVectorOrC) return RANK2;
  if (tRank == 3 && zEws == 1 && zVectorOrC) return RANK3;
  if (tRank == 4 && zEws == 1 && zVectorOrC) return RANK4;
  if (tRank == 5 && zEws == 1 && zVectorOrC) return RANK5;
  return COMMON;
}

//////////////////////////////////////////////////////////////////////////////
LoopKind::Kind LoopKind::deduceKindOfLoopTadXYZ(const LongType* xTadShapeInfo, const LongType* yTadShapeInfo,
                                                const LongType* zShapeInfo) {
  // both tad shapes are the same, but strides and ews may be different

  const int tadRank = shape::rank(xTadShapeInfo);

  const LongType xTadEws = shape::elementWiseStride(xTadShapeInfo);
  const LongType yTadEws = shape::elementWiseStride(yTadShapeInfo);
  const LongType zEws = shape::elementWiseStride(zShapeInfo);

  const char xTadOrder = shape::order(xTadShapeInfo);
  const char yTadOrder = shape::order(xTadShapeInfo);
  const char zOrder = shape::order(zShapeInfo);

  LongType position;
  const bool xTadVectorOrC = shape::isCommonVector(xTadShapeInfo, position) || xTadOrder == 'c';
  const bool yTadVectorOrC = shape::isCommonVector(yTadShapeInfo, position) || yTadOrder == 'c';
  const bool zVectorOrC = shape::isCommonVector(zShapeInfo, position) || zOrder == 'c';
  const bool allC = (xTadOrder == yTadOrder && xTadOrder == zOrder && zOrder == 'c');

  if (xTadEws == 1 && yTadEws == 1 && zEws == 1 && allC) return EWS1;
  if (xTadEws > 0 && yTadEws > 0 && zEws > 0 && (allC || (xTadVectorOrC && yTadVectorOrC && zVectorOrC)))
    return EWSNONZERO;
  if (tadRank == 1 && zEws > 0 && zVectorOrC) return RANK1;
  if (tadRank == 2 && zEws > 0 && zVectorOrC) return RANK2;
  if (tadRank == 3 && zEws > 0 && zVectorOrC) return RANK3;
  if (tadRank == 4 && zEws > 0 && zVectorOrC) return RANK4;
  if (tadRank == 5 && zEws > 0 && zVectorOrC) return RANK5;
  return COMMON;
}

}  // namespace sd
#endif  // LIBND4J_LOOPKIND_H
