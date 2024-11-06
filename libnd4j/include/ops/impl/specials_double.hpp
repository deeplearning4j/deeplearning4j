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
// @author raver119@gmail.com, created on 07.10.2017.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <array/NDArray.h>
#include <helpers/Loops.h>
#include <helpers/TAD.h>
#include <helpers/shape.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/specials.h>
#include <types/types.h>

namespace sd {

template <typename S, typename T>
void SpecialTypeConverter::convertGeneric(sd::Pointer *extras, void *dx, sd::LongType N, void *dz) {
  auto x = reinterpret_cast<S *>(dx);
  auto z = reinterpret_cast<T *>(dz);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      z[i] = static_cast<T>(x[i]);
    }
  };

  samediff::Threads::parallel_for(func, 0, N);
};

template <typename X, typename Y>
void quickSort_parallel_internal_key(X *key, sd::LongType const *xShapeInfo, Y *values, sd::LongType const *yShapeInfo,
                                     LongType left, LongType right, LongType cutoff, bool descending) {
  sd::LongType i = left, j = right;
  X ktmp;
  X pivot = key[shape::getIndexOffset((left + right) / 2, xShapeInfo)];

  Y vtmp;

  {
    /* PARTITION PART */
    while (i <= j) {
      if (descending) {
        while (key[shape::getIndexOffset(i, xShapeInfo)] > pivot) i++;
        while (key[shape::getIndexOffset(j, xShapeInfo)] < pivot) j--;
        if (i <= j) {
          ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
          key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
          key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

          vtmp = values[shape::getIndexOffset(i, yShapeInfo)];
          values[shape::getIndexOffset(i, yShapeInfo)] = values[shape::getIndexOffset(j, yShapeInfo)];
          values[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

          i++;
          j--;
        }
      } else {
        while (key[shape::getIndexOffset(i, xShapeInfo)] < pivot) i++;
        while (key[shape::getIndexOffset(j, xShapeInfo)] > pivot) j--;
        if (i <= j) {
          ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
          key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
          key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

          vtmp = values[shape::getIndexOffset(i, yShapeInfo)];
          values[shape::getIndexOffset(i, yShapeInfo)] = values[shape::getIndexOffset(j, yShapeInfo)];
          values[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

          i++;
          j--;
        }
      }
    }
  }

  //

  if (((right - left) < cutoff)) {
    if (left < j) {
      quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, left, j, cutoff, descending);
    }
    if (i < right) {
      quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, i, right, cutoff, descending);
    }

  } else {
    PRAGMA_OMP_TASK {
      quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, left, j, cutoff, descending);
    }
    PRAGMA_OMP_TASK {
      quickSort_parallel_internal_key(key, xShapeInfo, values, yShapeInfo, i, right, cutoff, descending);
    }
  }
}

template <typename X, typename Y>
void quickSort_parallel_internal_value(X *key, sd::LongType const *xShapeInfo, Y *value, sd::LongType const *yShapeInfo,
                                       LongType left, LongType right, LongType cutoff, bool descending) {
  sd::LongType i = left, j = right;
  X ktmp;
  Y pivot = value[shape::getIndexOffset((left + right) / 2, yShapeInfo)];

  Y vtmp;

  {
    /* PARTITION PART */
    while (i <= j) {
      if (descending) {
        while (value[shape::getIndexOffset(i, yShapeInfo)] > pivot) i++;
        while (value[shape::getIndexOffset(j, yShapeInfo)] < pivot) j--;
        if (i <= j) {
          ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
          key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
          key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

          vtmp = value[shape::getIndexOffset(i, yShapeInfo)];
          value[shape::getIndexOffset(i, yShapeInfo)] = value[shape::getIndexOffset(j, yShapeInfo)];
          value[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

          i++;
          j--;
        }
      } else {
        while (value[shape::getIndexOffset(i, yShapeInfo)] < pivot) i++;
        while (value[shape::getIndexOffset(j, yShapeInfo)] > pivot) j--;
        if (i <= j) {
          ktmp = key[shape::getIndexOffset(i, xShapeInfo)];
          key[shape::getIndexOffset(i, xShapeInfo)] = key[shape::getIndexOffset(j, xShapeInfo)];
          key[shape::getIndexOffset(j, xShapeInfo)] = ktmp;

          vtmp = value[shape::getIndexOffset(i, yShapeInfo)];
          value[shape::getIndexOffset(i, yShapeInfo)] = value[shape::getIndexOffset(j, yShapeInfo)];
          value[shape::getIndexOffset(j, yShapeInfo)] = vtmp;

          i++;
          j--;
        }
      }
    }
  }

  //

  if (((right - left) < cutoff)) {
    if (left < j) {
      quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, left, j, cutoff, descending);
    }
    if (i < right) {
      quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, i, right, cutoff, descending);
    }

  } else {
    PRAGMA_OMP_TASK {
      quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, left, j, cutoff, descending);
    }
    PRAGMA_OMP_TASK {
      quickSort_parallel_internal_value(key, xShapeInfo, value, yShapeInfo, i, right, cutoff, descending);
    }
  }
}

template <typename X, typename Y>
static void quickSort_parallel_key(NDArray *x, NDArray *y, sd::LongType lenArray, int numThreads,
                                   bool descending) {
  auto array = reinterpret_cast<X *>(x->bufferAsT<X>());
  auto values = reinterpret_cast<Y *>(y->bufferAsT<Y>());
  int cutoff = 1000;

  PRAGMA_OMP_PARALLEL_THREADS(numThreads) {
    PRAGMA_OMP_SINGLE_ARGS(nowait) {
      quickSort_parallel_internal_key(array, x->shapeInfo(), values, y->shapeInfo(), 0, lenArray - 1, cutoff, descending);
    }
  }
}

template <typename X, typename Y>
static void quickSort_parallel_value(NDArray *x, NDArray *y, sd::LongType lenArray, int numThreads,
                                     bool descending) {
  auto array = reinterpret_cast<X *>(x->bufferAsT<X>());
  auto values = reinterpret_cast<Y *>(y->bufferAsT<Y>());
  int cutoff = 1000;

  PRAGMA_OMP_PARALLEL_THREADS(numThreads) {
    PRAGMA_OMP_SINGLE_ARGS(nowait) {
      quickSort_parallel_internal_value(array, x->shapeInfo(), values,y->shapeInfo(), 0, lenArray - 1, cutoff, descending);
    }
  }
}

template <typename X, typename Y>
void DoubleMethods<X, Y>::sortByKey(NDArray *x,NDArray *y,
                                    bool descending) {
  quickSort_parallel_key<X, Y>(x,y, x->lengthOf(),Environment::getInstance().maxMasterThreads(),
                               descending);
}

template <typename X, typename Y>
void DoubleMethods<X, Y>::sortByValue(NDArray *x,NDArray *y,
                                      bool descending) {
  quickSort_parallel_value<X, Y>(x,y,x->lengthOf(),Environment::getInstance().maxMasterThreads(),
                                 descending);
}

template <typename X, typename Y>
void DoubleMethods<X, Y>::sortTadByKey(NDArray *xArr,NDArray *yArr,
                                       NDArray *dimension, bool descending) {
  auto x = xArr->bufferAsT<X>();
  auto y = yArr->bufferAsT<Y>();
  auto dimensionData = dimension->bufferAsT<sd::LongType>();
  auto dimensionLength = dimension->lengthOf();
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(xArr->shapeInfo(), dimensionData, dimensionLength);
  auto packY = ConstantTadHelper::getInstance().tadForDimensions(yArr->shapeInfo(), dimensionData, dimensionLength);

  auto xLength = xArr->lengthOf();
  auto xTadLength = shape::length(packX->primaryShapeInfo());
  auto numTads = packX->numberOfTads();

  auto func = PRAGMA_THREADS_FOR {
    for (auto r = start; r < stop; r++) {
      NDArray *xView = packX->extractTadView(xArr,r);
      NDArray *yView = packY->extractTadView(yArr,r);
      quickSort_parallel_key<X, Y>(xView,
                                   yView, xTadLength, 1,
                                   descending);
      delete xView;
      delete yView;
    }
  };

  samediff::Threads::parallel_tad(func, 0, numTads);
}

template <typename X, typename Y>
void DoubleMethods<X, Y>::sortTadByValue(NDArray *xArr, NDArray *yArr,
                                         NDArray *dimension, bool descending) {
  auto x = reinterpret_cast<X *>(xArr->bufferAsT<X>());
  auto y = reinterpret_cast<Y *>(yArr->bufferAsT<Y>());
  auto dimensionData = dimension->bufferAsT<sd::LongType>();
  auto len = dimension->lengthOf();
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(xArr->shapeInfo(), dimensionData, len);
  auto packY = ConstantTadHelper::getInstance().tadForDimensions(yArr->shapeInfo(), dimensionData, len);

  auto xLength = xArr->lengthOf();
  auto xTadLength = shape::length(packX->primaryShapeInfo());
  auto numTads = packX->numberOfTads();

  auto func = PRAGMA_THREADS_FOR {
    for (auto r = start; r < stop; r++) {
      NDArray *xView = packX->extractTadView(xArr,r);
      NDArray *yView = packY->extractTadView(yArr,r);
      quickSort_parallel_value<X, Y>(xView,
                                   yView, xTadLength, 1,
                                   descending);
      delete xView;
      delete yView;
    }
  };

  samediff::Threads::parallel_tad(func, 0, numTads);
}
}  // namespace sd
