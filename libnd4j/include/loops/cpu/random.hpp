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
//  @author raver119@gmail.com, created on 15.12.17.
//  @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/OmpLaunchHelper.h>
#include <loops/random.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace randomOps;

namespace functions {
namespace random {

template <typename X>
template <typename OpClass>
void RandomFunction<X>::execTransform(sd::Pointer state, const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                      const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                      void *vextraArguments) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<X *>(vz);
  auto extraArguments = reinterpret_cast<X *>(vextraArguments);

  if (OpClass::requiresSpecial) {
    OpClass::specialOp(state, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraArguments);
    return;
  }

  auto length = shape::length(zShapeInfo);

  sd::graph::RandomGenerator *rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);

  if (shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo) &&
      shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        z[i] = OpClass::op(x[i], y[i], i, length, rng, extraArguments);
      }
    };
    samediff::Threads::parallel_for(func, 0, length, 1);
  } else {
    sd::LongType coords[SD_MAX_RANK];
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        INDEX2COORDS(i, shape::rank(xShapeInfo), xShapeInfo, coords);
        sd::LongType xOffset, yOffset, zOffset;
        COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, xOffset);
        COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), coords, yOffset);
        COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), coords, zOffset);
        z[zOffset] = OpClass::op(x[xOffset], y[yOffset], i, length, rng, extraArguments);
      }
    };
    samediff::Threads::parallel_for(func, 0, length, 1);
  }
}
template <typename X>
template <typename OpClass>
void RandomFunction<X>::execTransform(sd::Pointer state, const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                      const sd::LongType *zShapeInfo, void *vextraArguments) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<X *>(vz);
  auto extraArguments = reinterpret_cast<X *>(vextraArguments);

  auto length = shape::length(zShapeInfo);

  sd::graph::RandomGenerator *rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);

  if (shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
    sd::LongType coords[SD_MAX_RANK];
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        INDEX2COORDS(i, shape::rank(xShapeInfo), xShapeInfo, coords);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, offset);
        z[offset] = OpClass::op(x[offset], i, length, rng, extraArguments);
      }
    };
    samediff::Threads::parallel_for(func, 0, length, 1);
  } else {
    sd::LongType coords[SD_MAX_RANK];
    auto func = PRAGMA_THREADS_FOR {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        INDEX2COORDS(i, shape::rank(zShapeInfo), zShapeInfo, coords);
        sd::LongType xOffset, zOffset;
        COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, xOffset);
        COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), coords, zOffset);
        z[zOffset] = OpClass::op(x[xOffset], i, length, rng, extraArguments);
      }
    };
    samediff::Threads::parallel_for(func, 0, length, 1);
  }
}

template <typename X>
template <typename OpClass>
void RandomFunction<X>::execTransform(sd::Pointer state, void *vz, const sd::LongType *zShapeInfo,
                                      void *vextraArguments) {
  auto z = reinterpret_cast<X *>(vz);
  auto extraArguments = reinterpret_cast<X *>(vextraArguments);
  auto length = shape::length(zShapeInfo);

  sd::graph::RandomGenerator *rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);

  sd::LongType coords[SD_MAX_RANK];
  auto func = PRAGMA_THREADS_FOR {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) {
      INDEX2COORDS(i, shape::rank(zShapeInfo), zShapeInfo, coords);
      sd::LongType offset;
      COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), coords, offset);
      z[offset] = OpClass::op(i, length, rng, extraArguments);
    }
  };

  samediff::Threads::parallel_for(func, 0, length, 1);
}


template <typename X>
void RandomFunction<X>::execTransform(int opNum, sd::Pointer state, const void *x, const sd::LongType *xShapeInfo,
                                      void *z, const sd::LongType *zShapeInfo, void *extraArguments) {
  DISPATCH_BY_OPNUM_T(execTransform, PARAMS(state, x, xShapeInfo, z, zShapeInfo, extraArguments), RANDOM_OPS)
}

template <typename X>
void RandomFunction<X>::execTransform(int opNum, sd::Pointer state, const void *x, const sd::LongType *xShapeInfo,
                                      const void *y, const sd::LongType *yShapeInfo, void *z,
                                      const sd::LongType *zShapeInfo, void *extraArguments) {

  DISPATCH_BY_OPNUM_T(execTransform, PARAMS(state, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraArguments),
                      RANDOM_OPS)
}

template <typename X>
void RandomFunction<X>::execTransform(int opNum, sd::Pointer state, void *z, const sd::LongType *zShapeInfo,
                                      void *extraArguments) {
  DISPATCH_BY_OPNUM_T(execTransform, PARAMS(state, z, zShapeInfo, extraArguments), RANDOM_OPS)
}

}  // namespace random
}  // namespace functions