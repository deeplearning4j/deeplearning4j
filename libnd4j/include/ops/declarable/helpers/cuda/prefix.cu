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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 12.06.2019
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/prefix.h>
#include <ops/ops.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
static void prefix_(scalar::Ops op, const void* vx, const LongType* xShapeInfo, void* vz, const LongType* zShapeInfo, bool exclusive, bool reverse) {
  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);

  const auto length = shape::length(xShapeInfo);
  const auto rankX = shape::rank(xShapeInfo);
  const auto rankZ = shape::rank(zShapeInfo);
  const auto shapeX = shape::shapeOf(xShapeInfo);
  const auto strideX = shape::stride(xShapeInfo);
  const auto shapeZ = shape::shapeOf(zShapeInfo);
  const auto strideZ = shape::stride(zShapeInfo);

  T prevSum = (op == scalar::Add) ? static_cast<T>(0) : static_cast<T>(1);
  T sum = prevSum;

  LongType coordsX[SD_MAX_RANK];
  LongType coordsZ[SD_MAX_RANK];

  if (reverse) {
    for (LongType e = length - 1; e >= 0; --e) {
      LongType offsetX, offsetZ;

      // Compute input and output offsets
      INDEX2COORDS(e, rankX, shapeX, coordsX);
      COORDS2INDEX(rankX, strideX, coordsX, offsetX);
      INDEX2COORDS(e, rankZ, shapeZ, coordsZ);
      COORDS2INDEX(rankZ, strideZ, coordsZ, offsetZ);

      // Perform operation
      sum = (op == scalar::Add) ? simdOps::Add<T, T, T>::op(sum, x[offsetX]) : simdOps::Multiply<T, T, T>::op(sum, x[offsetX]);

      if (!exclusive) prevSum = sum;

      z[offsetZ] = prevSum;
      prevSum = sum;
    }
  } else {
    for (LongType e = 0; e < length; ++e) {
      LongType offsetX, offsetZ;

      // Compute input and output offsets
      INDEX2COORDS(e, rankX, shapeX, coordsX);
      COORDS2INDEX(rankX, strideX, coordsX, offsetX);
      INDEX2COORDS(e, rankZ, shapeZ, coordsZ);
      COORDS2INDEX(rankZ, strideZ, coordsZ, offsetZ);

      // Perform operation
      sum = (op == scalar::Add) ? simdOps::Add<T, T, T>::op(sum, x[offsetX]) : simdOps::Multiply<T, T, T>::op(sum, x[offsetX]);

      if (!exclusive) prevSum = sum;

      z[offsetZ] = prevSum;
      prevSum = sum;
    }
  }
}

template <typename T>
static void prefix_(scalar::Ops op, NDArray* x, NDArray* z, const std::vector<LongType>& dims, bool exclusive,
                    bool reverse) {
  NDArray::preparePrimaryUse({z}, {x});
  auto xTads = x->allTensorsAlongDimension(dims);
  auto zTads = z->allTensorsAlongDimension(dims);
  auto t = xTads.size();

  for (int e = 0; e < t; e++) {
    auto tx = xTads.at(e);
    auto tz = zTads.at(e);

    prefix_<T>(op, tx->buffer(), tx->shapeInfo(), tz->buffer(), tz->shapeInfo(), exclusive, reverse);
  }

  NDArray::registerPrimaryUse({z}, {x});
};

///////////////////////////////////////////////////////////////////

template <typename T>
static void prefix_(scalar::Ops op, NDArray* x, NDArray* z, bool exclusive, bool reverse) {
  prefix_<T>(op, x->buffer(), x->shapeInfo(), z->buffer(), z->shapeInfo(), exclusive, reverse);
};

void prefix(LaunchContext* context, scalar::Ops op, NDArray* x, NDArray* z, bool exclusive, bool reverse) {
  BUILD_SINGLE_SELECTOR(x->dataType(), prefix_, (op, x, z, exclusive, reverse), SD_COMMON_TYPES);
}

void prefix(LaunchContext* context, scalar::Ops op, NDArray* x, NDArray* z, const std::vector<LongType>& dims,
            bool exclusive, bool reverse) {
  BUILD_SINGLE_SELECTOR(x->dataType(), prefix_, (op, x, z, dims, exclusive, reverse), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void prefix_,
                      (scalar::Ops op, const void* vx, sd::LongType const* xShapeInfo, void* vz,
                          sd::LongType const* zShapeInfo, bool exclusive, bool reverse),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void prefix_,
                      (scalar::Ops op, NDArray* x, NDArray* z, const std::vector<sd::LongType>& dims, bool exclusive,
                          bool reverse),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void prefix_,
                      (scalar::Ops op, NDArray* x, NDArray* z, bool exclusive, bool reverse), SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
