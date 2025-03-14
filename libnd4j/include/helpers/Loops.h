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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.03.2019
//

#ifndef LIBND4J_LOOPS_H
#define LIBND4J_LOOPS_H
#include <array/DataTypeUtils.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/LoopKind.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/shape.h>
#include <loops/indexreduce.h>
#include <ops/ops.h>


#include <functional>

namespace sd {

template <typename X, typename Z, typename E>
class SD_LIB_HIDDEN ReductionLoops {
 protected:
 public:
  template <typename OpType>
  static SD_INLINE void loopReduce(memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                                   const LongType* zShapeInfo, const LongType* dims, E* extraParams);
};

template <typename X, typename Z>
class SD_LIB_HIDDEN ReductionFloatLoops : public ReductionLoops<X, Z, Z> {
 public:
  static void wrapper(int opNum, memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                      const LongType* zShapeInfo, const LongType* dims, Z* extraParams);

  template <typename OpType>
  static void innerloopReduce(memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                              const LongType* zShapeInfo, const LongType* dims, Z* extraParams);
};

template <typename X, typename Z>
class SD_LIB_HIDDEN ReductionBoolLoops : public ReductionLoops<X, Z, X> {
 public:
  static void wrapper(int opNum, memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                      const LongType* zShapeInfo, const LongType* dims, X* extraParams);

  template <typename OpType>
  static void innerloopReduce(memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                              const LongType* zShapeInfo, const LongType* dims, X* extraParams);
};

template <typename X, typename Z>
class SD_LIB_HIDDEN ReductionLongLoops : public ReductionLoops<X, Z, X> {
 public:
  static void wrapper(int opNum, memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                      const LongType* zShapeInfo, const LongType* dims, X* extraParams);

  template <typename OpType>
  static void innerloopReduce(memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                              const LongType* zShapeInfo, const LongType* dims, X* extraParams);
};

template <typename X>
class SD_LIB_HIDDEN ReductionSameLoops : public ReductionLoops<X, X, X> {
 public:
  static void wrapper(int opNum, memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, X* z,
                      const LongType* zShapeInfo, const LongType* dims, X* extraParams);

  template <typename OpType>
  static void innerloopReduce(memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, X* z,
                              const LongType* zShapeInfo, const LongType* dims, X* extraParams);
};

template <typename X, typename Z>
class SD_LIB_HIDDEN IndexReductionLoops {
 public:
  static void wrapIndexReduce(int opNum, const void* x, const LongType* xShapeInfo, void* z,
                              const LongType* zShapeInfo, const LongType* tadShapeInfo,
                              const LongType* tadOffsets, void* extraParams);

  template <typename OpType>
  static void loopIndexReduce(X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                              const LongType* tadShapeInfo, const LongType* tadOffsets, void* extraParams);
};

template <typename X, typename Z, typename E>
class SD_LIB_HIDDEN TransformLoops {
 public:
  template <typename OpType>
  static SD_INLINE void loopTransform(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                      E* extraParams, LongType threadId, LongType numThreads);
};

template <typename X, typename Z>
class SD_LIB_HIDDEN Reduction3Loops {
 public:
  template <typename OpType>
  static SD_INLINE void loopReduce3(const X* x, const LongType* xShapeInfo, const X* y,
                                    const LongType* yShapeInfo, Z* z, const LongType* zShapeInfo,
                                    LongType* dims,
                                    int dimsLen, Z* extraParams, int64_t start, int64_t stop);

  template <typename OpType>
  static SD_INLINE void loopReduce3All(const X* x, const LongType* xShapeInfo, const X* y,
                                       const LongType* yShapeInfo, Z* z, const LongType* zShapeInfo,
                                       const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                                       const LongType* yTadShapeInfo, const LongType* yTadOffsets,
                                       Z* extraParams, int64_t start, int64_t stop);

  static void wrapper(int opNum, const X* x, const LongType* xShapeInfo, const X* y, const LongType* yShapeInfo,
                      Z* z, const LongType* zShapeInfo, LongType* dims, int dimsLen, Z* extraParams, int64_t start,
                      int64_t stop);

  static void wrapperAll(int opNum, const X* x, const LongType* xShapeInfo, const X* y,
                         const LongType* yShapeInfo, Z* z, const LongType* zShapeInfo,
                         const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                         const LongType* yTadShapeInfo, const LongType* yTadOffsets, Z* extraParams,
                         int64_t start, int64_t stop);

  template <typename OpType>
  static void innerloopReduce3(const X* x, const LongType* xShapeInfo, const X* y, const LongType* yShapeInfo,
                               Z* z, const LongType* zShapeInfo, LongType* dims, int dimsLen, Z* extraParams,
                               int64_t start, int64_t stop);

  template <typename OpType>
  static void innerloopReduce3All(const X* x, const LongType* xShapeInfo, const X* y,
                                  const LongType* yShapeInfo, Z* z, const LongType* zShapeInfo,
                                  const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                                  const LongType* yTadShapeInfo, const LongType* yTadOffsets, Z* extraParams,
                                  int64_t start, int64_t stop);
};

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
static void reduceExec21(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                         const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, dims[0]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, dims[0]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, static_cast<LongType>(0));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, dims[1]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, dims[1]);
  auto func = PRAGMA_THREADS_FOR {
    for (auto i0 = start; i0 < stop; ++i0) {
      auto x0 = x + i0 * xStrd0;
      auto z0 = z + i0 * zStrd0;

      auto s = OpType::startingValue(x0);

      if (xStrd1 == 1)
        for (LongType i1 = 0; i1 < xAxis1; ++i1) {
#if defined(PRINT_INDICES)
          shape::printShapeInfo(xShapeInfo);
          shape::printShapeInfo(zShapeInfo);
          printf("Index i0,i1 is %lld,%lld\n", i0,i1);
#endif
          s = OpType::update(s, OpType::op(x0[i1], extraParams), extraParams);
        }
      else
        for (LongType i1 = 0; i1 < xAxis1; ++i1) {
#if defined(PRINT_INDICES)
          shape::printShapeInfo(xShapeInfo);
          shape::printShapeInfo(zShapeInfo);
          printf("Index i0,i1 is %lld,%lld\n", i0,i1);
#endif
          s = OpType::update(s, OpType::op(x0[i1 * xStrd1], extraParams), extraParams);
        }

      *z0 = OpType::postProcess(s, static_cast<LongType>(xAxis1), extraParams);
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
static void reduceExec31(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                         const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, dims[0]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, dims[0]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, static_cast<LongType>(0));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, dims[1]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, dims[1]);

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, dims[2]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, dims[2]);

  const LongType tadLen = static_cast<LongType>(xAxis1 * xAxis2);
  auto func = PRAGMA_THREADS_FOR {
    for (auto i0 = start; i0 < stop; ++i0) {
      auto x0 = x + i0 * xStrd0;
      auto z0 = z + i0 * zStrd0;

      auto s = OpType::startingValue(x0);

      if (xStrd1 == 1)
        for (LongType i2 = 0; i2 < xAxis2; ++i2)
          for (LongType i1 = 0; i1 < xAxis1; ++i1) {
#if defined(PRINT_INDICES)
            shape::printShapeInfo(xShapeInfo);
            shape::printShapeInfo(zShapeInfo);
            printf("Index i0,i1,i2 is %lld,%lld,%lld reduceExec31\n", i0,i1,i2);
#endif
            s = OpType::update(s, OpType::op(x0[i1 + i2 * xStrd2], extraParams), extraParams);
          }
      else if (xStrd2 == 1)
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
            shape::printShapeInfo(xShapeInfo);
            shape::printShapeInfo(zShapeInfo);
            printf("Index i0,i1,i2 is %lld,%lld,%lld offset  is %lld reduceExec31\n", i0,i1,i2,i1 * xStrd1 + i2);
#endif
            s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2], extraParams), extraParams);
          }
      else
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
            shape::printShapeInfo(xShapeInfo);
            shape::printShapeInfo(zShapeInfo);
            printf("Index i0,i1,i2 is %lld,%lld,%lld offset is %lld reduceExec31\n", i0,i1,i2,i1 * xStrd1 + i2 * xStrd2);
#endif

            s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2 * xStrd2], extraParams), extraParams);
          }

      *z0 = OpType::postProcess(s, tadLen, extraParams);
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec32(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[1]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[1]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(0) : static_cast<LongType>(1));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[0]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[0]);
  const LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(1) : static_cast<LongType>(0));

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, dims[2]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, dims[2]);

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        auto x1 = x + i0 * xStrd0 + i1 * xStrd1;
        auto z1 = z + i0 * zStrd0 + i1 * zStrd1;

        auto s = OpType::startingValue(x1);

        if (xStrd2 == 1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
            shape::printShapeInfo(xShapeInfo);
            shape::printShapeInfo(zShapeInfo);
            printf("Index i0,i1,i2 is %lld,%lld,%lld reduceExec32\n", i0,i1,i2);
#endif
            s = OpType::update(s, OpType::op(x1[i2], extraParams), extraParams);
          }
        else
          for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
            shape::printShapeInfo(xShapeInfo);
            shape::printShapeInfo(zShapeInfo);
            printf("Index i0,i1,i2 is %lld,%lld,%lld reduceExec32\n", i0,i1,i2);
#endif
            s = OpType::update(s, OpType::op(x1[i2 * xStrd2], extraParams), extraParams);
          }
        *z1 = OpType::postProcess(s, static_cast<LongType>(xAxis2), extraParams);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0, 1, 0, xAxis1, 1);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec41(const X* x,
                                const LongType* xShapeInfo,
                                Z* z, const
                                LongType* zShapeInfo,
                                const LongType* dims,
                                E* extraParams) {
  LongType xRank = shape::rank(xShapeInfo);
  LongType zRank = shape::rank(zShapeInfo);

  const LongType xAxis0 = shape::sizeAt(xShapeInfo, dims[0]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, dims[0]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, static_cast<LongType>(0));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, dims[1]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, dims[1]);

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, dims[2]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, dims[2]);

  const LongType xAxis3 = shape::sizeAt(xShapeInfo, dims[3]);
  const LongType xStrd3 = shape::strideAt(xShapeInfo, dims[3]);

  const LongType tadLen = static_cast<LongType>(xAxis1 * xAxis2 * xAxis3);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i0 = start; i0 < stop; ++i0) {
      auto x0 = x + i0 * xStrd0;
      auto z0 = z + i0 * zStrd0;

      auto s = OpType::startingValue(x0);

      if (xStrd1 == 1)
        for (LongType i3 = 0; i3 < xAxis3; ++i3)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i1 = 0; i1 < xAxis1; ++i1) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld offset is %lld reduceExec41\n", i0,i1,i2,i3,i1 + i2 * xStrd2 + i3 * xStrd3);
#endif
              s = OpType::update(s, OpType::op(x0[i1 + i2 * xStrd2 + i3 * xStrd3], extraParams), extraParams);
            }
      else if (xStrd2 == 1)
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i3 = 0; i3 < xAxis3; ++i3)
            for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld offset is %lld reduceExec41\n", i0,i1,i2,i3,i1 * xStrd1 + i2 + i3 * xStrd3);
#endif
              s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2 + i3 * xStrd3], extraParams), extraParams);
            }

      else if (xStrd3 == 1)
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld offset is %lld reduceExec41\n", i0,i1,i2,i3,i1 * xStrd1 + i2 * xStrd2 + i3);
#endif
              s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2 * xStrd2 + i3], extraParams), extraParams);
            }
      else
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld offset is %lld reduceExec41\n", i0,i1,i2,i3,i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3);
#endif
              s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3], extraParams), extraParams);
            }
      *z0 = OpType::postProcess(s, tadLen, extraParams);
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec42(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[1]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[1]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(0) : static_cast<LongType>(1));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[0]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[0]);
  const LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(1) : static_cast<LongType>(0));

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, dims[2]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, dims[2]);

  const LongType xAxis3 = shape::sizeAt(xShapeInfo, dims[3]);
  const LongType xStrd3 = shape::strideAt(xShapeInfo, dims[3]);

  const LongType tadLen = static_cast<LongType>(xAxis2 * xAxis3);

  LongType xRank = shape::rank(xShapeInfo);

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        auto x1 = x + i0 * xStrd0 + i1 * xStrd1;
        auto z1 = z + i0 * zStrd0 + i1 * zStrd1;

        auto s = OpType::startingValue(x1);

        if (xStrd2 == 1)
          for (LongType i3 = 0; i3 < xAxis3; ++i3)
            for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld reduceExec42\n", i0,i1,i2,i3);
#endif
              s = OpType::update(s, OpType::op(x1[i2 + i3 * xStrd3], extraParams), extraParams);
            }
        else if (xStrd3 == 1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld offset %lld reduceExec42\n", i0,i1,i2,i3,i2 * xStrd2 + i3);
#endif
              s = OpType::update(s, OpType::op(x1[i2 * xStrd2 + i3], extraParams), extraParams);
            }
        else
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld offset %lld reduceExec42\n", i0,i1,i2,i3,i2 * xStrd2 + i3 * xStrd3);
#endif
              s = OpType::update(s, OpType::op(x1[i2 * xStrd2 + i3 * xStrd3], extraParams), extraParams);
            }

        *z1 = OpType::postProcess(s, tadLen, extraParams);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0, 1, 0, xAxis1, 1);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec43(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[2]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[2]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(0) : static_cast<LongType>(2));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, dims[1]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, dims[1]);
  const LongType zStrd1 = shape::strideAt(zShapeInfo, static_cast<LongType>(1));

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[2] : dims[0]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[2] : dims[0]);
  const LongType zStrd2 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(2) : static_cast<LongType>(0));

  const LongType xAxis3 = shape::sizeAt(xShapeInfo, dims[3]);
  const LongType xStrd3 = shape::strideAt(xShapeInfo, dims[3]);
  LongType xRank = shape::rank(xShapeInfo);

  auto func = PRAGMA_THREADS_FOR_3D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        for (auto i2 = start_z; i2 < stop_z; ++i2) {
          auto x2 = x + i0 * xStrd0 + i1 * xStrd1 + i2 * xStrd2;
          auto z2 = z + i0 * zStrd0 + i1 * zStrd1 + i2 * zStrd2;

          auto s = OpType::startingValue(x2);

          if (xStrd3 == 1)
            for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld reduceExec43\n", i0,i1,i2,i3);
#endif
              s = OpType::update(s, OpType::op(x2[i3], extraParams), extraParams);
            }
          else
            for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
              shape::printShapeInfo(xShapeInfo);
              shape::printShapeInfo(zShapeInfo);
              printf("Index i0,i1,i2,i3 is %lld,%lld,%lld,%lld reduceExec43\n", i0,i1,i2,i3);
#endif
              s = OpType::update(s, OpType::op(x2[i3 * xStrd3], extraParams), extraParams);
            }

          *z2 = OpType::postProcess(s, static_cast<LongType>(xAxis3), extraParams);
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0, 1, 0, xAxis1, 1, 0, xAxis2, 1);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec51(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, dims[0]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, dims[0]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, static_cast<LongType>(0));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, dims[1]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, dims[1]);

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, dims[2]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, dims[2]);

  const LongType xAxis3 = shape::sizeAt(xShapeInfo, dims[3]);
  const LongType xStrd3 = shape::strideAt(xShapeInfo, dims[3]);

  const LongType xAxis4 = shape::sizeAt(xShapeInfo, dims[4]);
  const LongType xStrd4 = shape::strideAt(xShapeInfo, dims[4]);

  const LongType tadLen = static_cast<LongType>(xAxis1 * xAxis2 * xAxis3 * xAxis4);

  LongType xRank = shape::rank(xShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i0 = start; i0 < stop; ++i0) {
      auto x0 = x + i0 * xStrd0;
      auto z0 = z + i0 * zStrd0;

      auto s = OpType::startingValue(x0);

      if (xStrd1 == 1)
        for (LongType i4 = 0; i4 < xAxis4; ++i4)
          for (LongType i3 = 0; i3 < xAxis3; ++i3)
            for (LongType i2 = 0; i2 < xAxis2; ++i2)
              for (LongType i1 = 0; i1 < xAxis1; ++i1) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld reduceExec51\n", i0,i1,i2,i3,i4);
#endif
                s = OpType::update(s, OpType::op(x0[i1 + i2 * xStrd2 + i3 * xStrd3 + i4 * xStrd4], extraParams),
                                   extraParams);
              }
      else if (xStrd2 == 1)
        for (LongType i4 = 0; i4 < xAxis4; ++i4)
          for (LongType i3 = 0; i3 < xAxis3; ++i3)
            for (LongType i1 = 0; i1 < xAxis1; ++i1)
              for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld  offset %lldreduceExec51\n", i0,i1,i2,i3,i4,i1 * xStrd1 + i2 + i3 * xStrd3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2 + i3 * xStrd3 + i4 * xStrd4], extraParams),
                                   extraParams);
              }
      else if (xStrd3 == 1)
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i4 = 0; i4 < xAxis4; ++i4)
              for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld  offset %lld reduceExec51\n", i0,i1,i2,i3,i4,i1 * xStrd1 + i2 * xStrd2 + i3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2 * xStrd2 + i3 + i4 * xStrd4], extraParams),
                                   extraParams);
              }
      else if (xStrd4 == 1)
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3)
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld  offset %lld reduceExec51\n", i0,i1,i2,i3,i4,i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3 + i4);
#endif
                s = OpType::update(s, OpType::op(x0[i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3 + i4], extraParams),
                                   extraParams);
              }
      else
        for (LongType i1 = 0; i1 < xAxis1; ++i1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3)
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld  offset %lld reduceExec51\n", i0,i1,i2,i3,i4,i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3 + i4 * xStrd4);
#endif
                s = OpType::update(
                    s, OpType::op(x0[i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3 + i4 * xStrd4], extraParams), extraParams);
              }
      *z0 = OpType::postProcess(s, tadLen, extraParams);
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec52(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[1]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[1]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(0) : static_cast<LongType>(1));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[0]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[0]);
  const LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(1) : static_cast<LongType>(0));

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, dims[2]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, dims[2]);

  const LongType xAxis3 = shape::sizeAt(xShapeInfo, dims[3]);
  const LongType xStrd3 = shape::strideAt(xShapeInfo, dims[3]);

  const LongType xAxis4 = shape::sizeAt(xShapeInfo, dims[4]);
  const LongType xStrd4 = shape::strideAt(xShapeInfo, dims[4]);

  const LongType tadLen = static_cast<LongType>(xAxis2 * xAxis3 * xAxis4);

  LongType xRank = shape::rank(xShapeInfo);

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        auto x1 = x + i0 * xStrd0 + i1 * xStrd1;
        auto z1 = z + i0 * zStrd0 + i1 * zStrd1;

        auto s = OpType::startingValue(x1);

        if (xStrd2 == 1)
          for (LongType i4 = 0; i4 < xAxis4; ++i4)
            for (LongType i3 = 0; i3 < xAxis3; ++i3)
              for (LongType i2 = 0; i2 < xAxis2; ++i2) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld offset %lld reduceExec52\n", i0,i1,i2,i3,i4,i2 + i3 * xStrd3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x1[i2 + i3 * xStrd3 + i4 * xStrd4], extraParams), extraParams);
              }
        else if (xStrd3 == 1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i4 = 0; i4 < xAxis4; ++i4)
              for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld offset %lld reduceExec52\n", i0,i1,i2,i3,i4,i2 + i3 * xStrd3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x1[i2 * xStrd2 + i3 + i4 * xStrd4], extraParams), extraParams);
              }
        else if (xStrd4 == 1)
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3)
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld offset %lld reduceExec52\n", i0,i1,i2,i3,i4,i2 * xStrd2 + i3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x1[i2 * xStrd2 + i3 * xStrd3 + i4], extraParams), extraParams);
              }
        else
          for (LongType i2 = 0; i2 < xAxis2; ++i2)
            for (LongType i3 = 0; i3 < xAxis3; ++i3)
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld offset %lld reduceExec52\n", i0,i1,i2,i3,i4,i2 * xStrd2 + i3 * xStrd3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x1[i2 * xStrd2 + i3 * xStrd3 + i4 * xStrd4], extraParams),
                                   extraParams);
              }

        *z1 = OpType::postProcess(s, tadLen, extraParams);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0, 1, 0, xAxis1, 1);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec53(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[2]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[2]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(0) : static_cast<LongType>(2));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, dims[1]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, dims[1]);
  const LongType zStrd1 = shape::strideAt(zShapeInfo, static_cast<LongType>(1));

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[2] : dims[0]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[2] : dims[0]);
  const LongType zStrd2 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(2) : static_cast<LongType>(0));

  const LongType xAxis3 = shape::sizeAt(xShapeInfo, dims[3]);
  const LongType xStrd3 = shape::strideAt(xShapeInfo, dims[3]);

  const LongType xAxis4 = shape::sizeAt(xShapeInfo, dims[4]);
  const LongType xStrd4 = shape::strideAt(xShapeInfo, dims[4]);

  const LongType tadLen = static_cast<LongType>(xAxis3 * xAxis4);

  LongType xRank = shape::rank(xShapeInfo);
  auto func = PRAGMA_THREADS_FOR_3D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        for (auto i2 = start_z; i2 < stop_z; ++i2) {
          auto x2 = x + i0 * xStrd0 + i1 * xStrd1 + i2 * xStrd2;
          auto z2 = z + i0 * zStrd0 + i1 * zStrd1 + i2 * zStrd2;

          auto s = OpType::startingValue(x2);

          if (xStrd3 == 1)
            for (LongType i4 = 0; i4 < xAxis4; ++i4)
              for (LongType i3 = 0; i3 < xAxis3; ++i3) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld offset %lld reduceExec53\n", i0,i1,i2,i3,i4,i3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x2[i3 + i4 * xStrd4], extraParams), extraParams);
              }
          else if (xStrd4 == 1)
            for (LongType i3 = 0; i3 < xAxis3; ++i3)
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld offset %lld reduceExec53\n", i0,i1,i2,i3,i4,i3 * xStrd3 + i4);
#endif
                s = OpType::update(s, OpType::op(x2[i3 * xStrd3 + i4], extraParams), extraParams);
              }
          else
            for (LongType i3 = 0; i3 < xAxis3; ++i3)
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                printf("Index i0,i1,i2,i3,i4 is %lld,%lld,%lld,%lld,%lld offset %lld reduceExec53\n", i0,i1,i2,i3,i4,i3 * xStrd3 + i4 * xStrd4);
#endif
                s = OpType::update(s, OpType::op(x2[i3 * xStrd3 + i4 * xStrd4], extraParams), extraParams);
              }
          *z2 = OpType::postProcess(s, tadLen, extraParams);
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0, 1, 0, xAxis1, 1, 0, xAxis2, 1);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceExec54(const X* x, const LongType* xShapeInfo, Z* z, const LongType* zShapeInfo,
                                const LongType* dims, E* extraParams) {
  const LongType xAxis0 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[3]);
  const LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[0] : dims[3]);
  const LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(0) : static_cast<LongType>(3));

  const LongType xAxis1 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[2]);
  const LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[1] : dims[2]);
  const LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(1) : static_cast<LongType>(2));

  const LongType xAxis2 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[2] : dims[1]);
  const LongType xStrd2 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[2] : dims[1]);
  const LongType zStrd2 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(2) : static_cast<LongType>(1));

  const LongType xAxis3 = shape::sizeAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[3] : dims[0]);
  const LongType xStrd3 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? dims[3] : dims[0]);
  const LongType zStrd3 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<LongType>(3) : static_cast<LongType>(0));

  const LongType xAxis4 = shape::sizeAt(xShapeInfo, dims[4]);
  const LongType xStrd4 = shape::strideAt(xShapeInfo, dims[4]);

  LongType xRank = shape::rank(xShapeInfo);

  auto func = PRAGMA_THREADS_FOR_3D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        for (auto i2 = start_z; i2 < stop_z; ++i2) {
          for (auto i3 = 0; i3 < xAxis3; ++i3) {
            auto x3 = x + i0 * xStrd0 + i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3;
            auto z3 = z + i0 * zStrd0 + i1 * zStrd1 + i2 * zStrd2 + i3 * zStrd3;

            auto s = OpType::startingValue(x3);

            if (xStrd4 == 1)
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
#endif
                s = OpType::update(s, OpType::op(x3[i4], extraParams), extraParams);
              }
            else
              for (LongType i4 = 0; i4 < xAxis4; ++i4) {
#if defined(PRINT_INDICES)
                shape::printShapeInfo(xShapeInfo);
                shape::printShapeInfo(zShapeInfo);
#endif
                s = OpType::update(s, OpType::op(x3[i4 * xStrd4], extraParams), extraParams);
              }
            *z3 = OpType::postProcess(s, static_cast<LongType>(xAxis4), extraParams);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, xAxis0, 1, 0, xAxis1, 1, 0, xAxis2, 1);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E, typename OpType>
SD_LIB_HIDDEN void reduceDefault(memory::Workspace* workspace, const X* x, const LongType* xShapeInfo, Z* z,
                                 const LongType* zShapeInfo, const LongType* dims, E* extraParams) {
  const int zRank = shape::rank(zShapeInfo);
  const int tadRank = shape::rank(xShapeInfo) - zRank;

  LongType* outerXTadShapeInfo = ShapeBuilders::createSubArrShapeInfo(xShapeInfo, dims, zRank);
  LongType* innerXTadShapeInfo = ShapeBuilders::createSubArrShapeInfo(xShapeInfo, dims + zRank, tadRank);

  const bool sameOffsets1 = shape::haveSameShapeAndStrides(zShapeInfo, outerXTadShapeInfo);
  const bool sameOffsets2 = shape::haveSameShapeAndStrides(zShapeInfo, innerXTadShapeInfo);

  const LongType zLen = shape::length(zShapeInfo);
  const LongType tadLen = shape::length(innerXTadShapeInfo);

  LongType* zOffsets = nullptr;
  ALLOCATE(zOffsets, workspace, zLen, sd::LongType);
  shape::calcOffsets(zShapeInfo, zOffsets);

  LongType* outerXTadOffsets = zOffsets;
  if (!sameOffsets1) {
    ALLOCATE(outerXTadOffsets, workspace, zLen, sd::LongType);
    shape::calcOffsets(outerXTadShapeInfo, outerXTadOffsets);
  }

  LongType* innerXTadOffsets = zOffsets;
  if (!sameOffsets2) {
    ALLOCATE(innerXTadOffsets, workspace, tadLen, sd::LongType);
    shape::calcOffsets(innerXTadShapeInfo, innerXTadOffsets);
  }

  LongType xRank = shape::rank(xShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; ++i) {
      const auto tad = x + outerXTadOffsets[i];
      auto s = OpType::startingValue(tad);

      for (LongType j = 0; j < tadLen; j++) {
#if defined(PRINT_INDICES)
        shape::printShapeInfo(outerXTadShapeInfo);
        shape::printShapeInfo(innerXTadShapeInfo);
        printf("Index i,j is %lld,%lld  offset %lld reduceDefault\n", i,j,innerXTadOffsets[j]);
#endif
        s = OpType::update(s, OpType::op(tad[innerXTadOffsets[j]], extraParams), extraParams);
      }
      z[zOffsets[i]] = OpType::postProcess(s, tadLen, extraParams);
    }
  };

  samediff::Threads::parallel_for(func, 0, shape::length(zShapeInfo));

  RELEASE(outerXTadShapeInfo, workspace);
  RELEASE(innerXTadShapeInfo, workspace);
  RELEASE(zOffsets, workspace);
  if (!sameOffsets1) RELEASE(outerXTadOffsets, workspace);
  if (!sameOffsets2) RELEASE(innerXTadOffsets, workspace);
}

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E>
template <typename OpType>
SD_LIB_HIDDEN void ReductionLoops<X, Z, E>::loopReduce(memory::Workspace* workspace, const X* x,
                                                       const LongType* xShapeInfo, Z* z,
                                                       const LongType* zShapeInfo, const LongType* dims,
                                                       E* extraParams) {
  const LongType xRank = shape::rank(xShapeInfo);
  const LongType zRank = shape::rank(zShapeInfo);

  if (xRank == 2 && zRank == 1)
    reduceExec21<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 3 && zRank == 1)
    reduceExec31<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 3 && zRank == 2)
    reduceExec32<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 4 && zRank == 1)
    reduceExec41<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 4 && zRank == 2)
    reduceExec42<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 4 && zRank == 3)
    reduceExec43<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 5 && zRank == 1)
    reduceExec51<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 5 && zRank == 2)
    reduceExec52<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 5 && zRank == 3)
    reduceExec53<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else if (xRank == 5 && zRank == 4)
    reduceExec54<X, Z, E, OpType>(x, xShapeInfo, z, zShapeInfo, dims, extraParams);
  else
    reduceDefault<X, Z, E, OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, extraParams);
}

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E>
template <typename OpType>
SD_LIB_HIDDEN void TransformLoops<X, Z, E>::loopTransform(const X* x,
                                                          const LongType* xShapeInfo, Z* z,
                                                          const LongType* zShapeInfo,
                                                          E* extraParams,
                                                          LongType threadId, LongType numThreads) {
  const LoopKind::Kind kindOfLoop = LoopKind::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);
  if(xShapeInfo == nullptr) {
    THROW_EXCEPTION("Input x shape info was null!");
  }

  if(xShapeInfo[0] > SD_MAX_RANK || xShapeInfo[0] < 0) {
    THROW_EXCEPTION("x shape info appears to be corrupt. This is likely due to deallocation.");
  }

  if(zShapeInfo[0] > SD_MAX_RANK || zShapeInfo[0] < 0) {
    THROW_EXCEPTION("z shape info appears to be corrupt. This is likely due to deallocation.");
  }

  if(zShapeInfo == nullptr) {
    THROW_EXCEPTION("Input z shape info was null!");
  }


  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType zRank = shape::rank(zShapeInfo);

  const LongType* xShape = shape::shapeOf(const_cast<LongType*>(xShapeInfo));
  const LongType* zShape = shape::shapeOf(const_cast<LongType*>(zShapeInfo));
  const LongType* xStride = shape::stride(const_cast<LongType*>(xShapeInfo));
  const LongType* zStride = shape::stride(const_cast<LongType*>(zShapeInfo));
  const LongType len = shape::length(xShapeInfo);
  switch (kindOfLoop) {
    //*********************************************//
    default: {
      if(shape::shapeEquals(xShapeInfo, zShapeInfo)) {
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        auto xLen = shape::length(xShapeInfo);
        auto zLen = shape::length(zShapeInfo);
        auto span = samediff::Span::build(threadId, numThreads, 0, len, 1);

        for (auto i = span.startX(); i < span.stopX(); i++) {
          INDEX2COORDS(i, xRank, xShape, xCoords);
          INDEX2COORDS(i,zRank, zShape, zCoords);

          sd::LongType xOffset;
          sd::LongType zOffset;
          COORDS2INDEX(xRank, xStride, xCoords, xOffset);
          COORDS2INDEX(zRank, zStride, zCoords, zOffset);


          auto opResult = OpType::op(x[xOffset], extraParams);
          z[zOffset] = static_cast<Z>(opResult);
        }


      } else {
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        auto xLen = shape::length(xShapeInfo);
        auto zLen = shape::length(zShapeInfo);
        auto span = samediff::Span::build(threadId, numThreads, 0, len, 1);

        for (auto i = span.startX(); i < span.stopX(); i++) {
          INDEX2COORDS(i, xRank, xShape, xCoords);
          INDEX2COORDS(i, zRank, zShape, zCoords);

          LongType xOffset;
          LongType zOffset;
          COORDS2INDEX(xRank, xStride, xCoords, xOffset);
          COORDS2INDEX(zRank, zStride, zCoords, zOffset);


          auto opResult = OpType::op(x[xOffset], extraParams);
          z[zOffset] = static_cast<Z>(opResult);
        }
      }
    }

  }

}

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void Reduction3Loops<X, Z>::loopReduce3(const X* x, const LongType* xShapeInfo, const X* y,
                                        const LongType* yShapeInfo, Z* z, const LongType* zShapeInfo,
                                        LongType* dims, int dimsLen, Z* extraParameters, int64_t start, int64_t stop) {
  // both tads have same shape, however strides and ews may differ

  Z param0(OpType::startingValue(x)), param1(OpType::startingValue(x)),
      param2(extraParameters ? extraParameters[0] : OpType::startingValue(x));

  const LongType xLen = shape::length(xShapeInfo);
  const LongType yLen = shape::length(yShapeInfo);

  const LongType *xTadShapeInfo = nullptr, *yTadShapeInfo = nullptr, *xTadOffsets = nullptr, *yTadOffsets = nullptr;
  TadPack *tadPackX, *tadPackY;
  std::vector<LongType> zeroOffsets;

  if (xLen == yLen) {
    tadPackX = ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType*>(xShapeInfo), dims, dimsLen);
    tadPackY = ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType*>(yShapeInfo), dims, dimsLen);
    xTadShapeInfo = tadPackX->primaryShapeInfo();
    yTadShapeInfo = tadPackY->primaryShapeInfo();
    xTadOffsets = tadPackX->primaryOffsets();
    yTadOffsets = tadPackY->primaryOffsets();
  } else if (yLen > xLen) {
    tadPackY = ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType*>(yShapeInfo), dims, dimsLen);
    xTadShapeInfo = xShapeInfo;
    yTadShapeInfo = tadPackY->primaryShapeInfo();
    yTadOffsets = tadPackY->primaryOffsets();
  } else {
    tadPackX = ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType*>(xShapeInfo), dims, dimsLen);
    yTadShapeInfo = yShapeInfo;
    xTadShapeInfo = tadPackX->primaryShapeInfo();
    xTadOffsets = tadPackX->primaryOffsets();
  }

  const auto zLen = shape::length(zShapeInfo);
  const auto tadLen = shape::length(xTadShapeInfo);

  const auto tadShape = shape::shapeOf(xTadShapeInfo);
  const auto xTadStride = shape::stride(xTadShapeInfo);
  const auto yTadStride = shape::stride(xTadShapeInfo);

  int numThreads = OmpLaunchHelper::tadThreads(tadLen, zLen);

  LongType castXTadShapeInfo[SD_MAX_RANK];
  const bool canCastXTad = DataTypeUtils::castShapeInfo<LongType>(xTadShapeInfo, castXTadShapeInfo);
  LongType castYTadShapeInfo[SD_MAX_RANK];
  const bool canCastYTad = DataTypeUtils::castShapeInfo<LongType>(yTadShapeInfo, castYTadShapeInfo);

  sd::LongType zRank = shape::rank(zShapeInfo);
  sd::LongType *zShape = shape::shapeOf(zShapeInfo);
  sd::LongType *zStride = shape::stride(zShapeInfo);
  Z extraParams[3];
  for (auto i = start; i < stop; i++) {
    extraParams[0] = param0;
    extraParams[1] = param1;
    extraParams[2] = param2;

    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
    auto s = OpType::startingValue(xTad);

    // Calculate z coordinates for this iteration
    LongType zCoords[SD_MAX_RANK];
    INDEX2COORDS(i,zRank, zShape, zCoords);
    LongType zOffset;
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    sd::LongType xTadRank = shape::rank(xTadShapeInfo);
    sd::LongType yTadRank = shape::rank(yTadShapeInfo);
    sd::LongType *xTadShape = shape::shapeOf(xTadShapeInfo);
    sd::LongType *yTadShape = shape::shapeOf(yTadShapeInfo);

    for (LongType j = 0; j < tadLen; ++j) {
      LongType coords[SD_MAX_RANK];
      LongType  yCoords[SD_MAX_RANK];
      INDEX2COORDS(j, xTadRank, xTadShape, coords);
      INDEX2COORDS(j, yTadRank,yTadShape, yCoords);
      LongType xTadOffset, yTadOffset;
      COORDS2INDEX(xTadRank, xTadStride, coords, xTadOffset);
      COORDS2INDEX(yTadRank,yTadStride, yCoords, yTadOffset);

#if defined(PRINT_INDICES)
      shape::printShapeInfo(xTadShapeInfo);
      shape::printShapeInfo(yTadShapeInfo);
#endif
      s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
    }

    z[zOffset] = OpType::postProcess(s, tadLen, extraParams);
  };
}

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void Reduction3Loops<X, Z>::loopReduce3All(const X* x, const LongType* xShapeInfo, const X* y,
                                           const LongType* yShapeInfo, Z* z, const LongType* zShapeInfo,
                                           const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                                           const LongType* yTadShapeInfo, const LongType* yTadOffsets,
                                           Z* extraParameters, int64_t start, int64_t stop) {
  // both tads have same shape, however strides and ews may differ

  Z param0(OpType::startingValue(x)), param1(OpType::startingValue(x)),
      param2(extraParameters ? extraParameters[0] : OpType::startingValue(x));



  const auto zLen = shape::length(zShapeInfo);
  const auto tadLen = shape::length(xTadShapeInfo);

  const auto numXTads = shape::length(xShapeInfo) / tadLen;
  const auto numYTads = shape::length(yShapeInfo) / tadLen;

  const auto tadShape = shape::shapeOf(xTadShapeInfo);
  const auto xTadStride = shape::stride(xTadShapeInfo);
  const auto yTadStride = shape::stride(yTadShapeInfo);

  const auto startVal = OpType::startingValue(x);

  int numThreads = OmpLaunchHelper::tadThreads(tadLen, numXTads * numYTads);

  //*********************************************//
  LongType castXTadShapeInfo[SD_MAX_RANK];
  LongType castYTadShapeInfo[SD_MAX_RANK];
  const bool canCastYTad = DataTypeUtils::castShapeInfo<LongType>(yTadShapeInfo, castYTadShapeInfo);

  Z extraParams[3];
  sd::LongType *xTadShape = shape::shapeOf(xTadShapeInfo);
  sd::LongType *yTadShape = shape::shapeOf(yTadShapeInfo);
  sd::LongType xTadRank = shape::rank(xTadShapeInfo);
  sd::LongType zRank = shape::rank(zShapeInfo);
  sd::LongType *zShape = shape::shapeOf(zShapeInfo);
  sd::LongType *zStride = shape::stride(zShapeInfo);
  sd::LongType yTadRank = shape::rank(yTadShapeInfo);

  for (LongType ix = 0; ix < numXTads; ix++) {
    for (LongType iy = 0; iy < numYTads; iy++) {
      extraParams[0] = param0;
      extraParams[1] = param1;
      extraParams[2] = param2;

      const auto xTad = x + xTadOffsets[ix];
      const auto yTad = y + yTadOffsets[iy];
      auto s = startVal;

      for (LongType j = 0; j < tadLen; ++j) {
        LongType coords[SD_MAX_RANK];
        INDEX2COORDS(j, xTadRank, xTadShape, coords);
        LongType xTadOffset, yTadOffset;
        COORDS2INDEX(xTadRank, xTadStride, coords, xTadOffset);
        COORDS2INDEX(yTadRank, yTadStride, coords, yTadOffset);
#if defined(PRINT_INDICES)
        shape::printShapeInfo(xTadShapeInfo);
        shape::printShapeInfo(yTadShapeInfo);
        printf("Index is %lld offset is %lld loop kind: default Reduction3Loops<X, Z>::loopReduce3All\n", ix * numYTads + iy, j);
#endif
        s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
      }
      z[ix * numYTads + iy] = OpType::postProcess(s, tadLen, extraParams);
    }
  }

}

}  // namespace sd

#endif  // LIBND4J_LOOPS_H
