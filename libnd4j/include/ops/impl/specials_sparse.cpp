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
// @author raver119@gmail.com
//
#include <helpers/shape.h>
#include <ops/specials_sparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <types/float16.h>
#include <types/types.h>

namespace sd {
namespace sparse {

template <typename T>
void SparseUtils<T>::printIndex(sd::LongType *indices, int rank, int x) {
  printf(" [");
  for (int e = 0; e < rank; e++) {
    if (e > 0) printf(", ");

    printf("%lld", (long long)indices[x * rank + e]);
  }
  printf("] ");
}

template <typename T>
bool SparseUtils<T>::ltIndices(sd::LongType *indices, int rank, sd::LongType x, sd::LongType y) {
  for (int e = 0; e < rank; e++) {
    sd::LongType idxX = indices[x * rank + e];
    sd::LongType idxY = indices[y * rank + e];
    // we're comparing indices one by one, starting from outer dimension
    if (idxX < idxY) {
      return true;
    } else if (idxX == idxY) {
      // do nothing, continue to next dimension
    } else
      return false;
  }

  return false;
}

template <typename T>
bool SparseUtils<T>::gtIndices(sd::LongType *indices, int rank, sd::LongType x, sd::LongType y) {
  for (int e = 0; e < rank; e++) {
    // we're comparing indices one by one, starting from outer dimension
    sd::LongType idxX = indices[x * rank + e];
    sd::LongType idxY = indices[y * rank + e];
    if (idxX > idxY) {
      return true;
    } else if (idxX == idxY) {
      // do nothing, continue to next dimension
    } else
      return false;
  }
  return false;
}

template <typename T>
void SparseUtils<T>::swapEverything(sd::LongType *indices, T *array, int rank, sd::LongType x, sd::LongType y) {
  // swap indices
  for (int e = 0; e < rank; e++) {
    sd::LongType tmp = indices[x * rank + e];
    indices[x * rank + e] = indices[y * rank + e];
    indices[y * rank + e] = tmp;
  }

  // swap values
  T tmp = array[x];
  array[x] = array[y];
  array[y] = tmp;
}

template <typename T>
sd::LongType SparseUtils<T>::coo_quickSort_findPivot(sd::LongType *indices, T *array, sd::LongType left,
                                                     sd::LongType right, int rank) {
  sd::LongType mid = (left + right) / 2;

  // ensure left < mid
  if (ltIndices(indices, rank, mid, left)) {  // ensure lo < mid
    swapEverything(indices, array, rank, mid, left);
  }

  // ensure left < right
  if (ltIndices(indices, rank, right, left)) {
    swapEverything(indices, array, rank, right, left);
  }

  // ensure mid < right
  if (ltIndices(indices, rank, right, mid)) {
    swapEverything(indices, array, rank, right, mid);
  }

  // mid is the median of the 3, and is the optimal pivot point
  return mid;
}

template <typename T>
void SparseUtils<T>::coo_quickSort_parallel_internal(sd::LongType *indices, T *array, sd::LongType left,
                                                     sd::LongType right, int cutoff, int rank) {
  sd::LongType span = right - left;  // elements to be partitioned - 1

  if (span == 1) {
    // only 2 elements to partition. swap if needed and return directly without further sorting.
    if (ltIndices(indices, rank, right, left)) {
      swapEverything(indices, array, rank, left, right);
    }
    return;
  }

  // find optimal pivot and sort left < right < right
  sd::LongType pvt = coo_quickSort_findPivot(indices, array, left, right, rank);

  if (span == 2) {
    // only 3 elements to partition. findPivot has already sorted them. no further sorting is needed.
    return;
  }

  // index that is greater than pivot - leftmost element is already partitioned because of findPivot.
  sd::LongType i = left + 1;

  // index that is smaller than pivot - rightmost element is already partitioned because of findPivot.
  sd::LongType j = right - 1;

  {
    // flag that indicates that pivot index lies between i and j and *could* be swapped.
    bool checkPivot = true;
    /* PARTITION PART */
    while (i <= j) {
      while (ltIndices(indices, rank, i, pvt)) i++;

      while (gtIndices(indices, rank, j, pvt)) j--;

      if (i <= j) {
        if (i != j) {  // swap can be fairly expensive. don't swap i -> i
          swapEverything(indices, array, rank, i, j);
        }

        // only check pivot if it hasn't already been swapped.
        if (checkPivot) {
          // check if we moved the pivot, if so, change pivot index accordingly
          if (pvt == j) {
            pvt = i;
            checkPivot = false;
          } else if (pvt == i) {
            pvt = j;
            checkPivot = false;
          }
        }

        i++;
        j--;
      }
    }
  }

  if ((span < cutoff)) {
    if (left < j) {
      coo_quickSort_parallel_internal(indices, array, left, j, cutoff, rank);
    }
    if (i < right) {
      coo_quickSort_parallel_internal(indices, array, i, right, cutoff, rank);
    }

  } else {
    PRAGMA_OMP_TASK { coo_quickSort_parallel_internal(indices, array, left, j, cutoff, rank); }
    PRAGMA_OMP_TASK { coo_quickSort_parallel_internal(indices, array, i, right, cutoff, rank); }
  }
}

template <typename T>
void SparseUtils<T>::coo_quickSort_parallel(sd::LongType *indices, T *array, sd::LongType lenArray, int numThreads,
                                            int rank) {
  int cutoff = 1000;

  PRAGMA_OMP_PARALLEL_THREADS(numThreads) {
    PRAGMA_OMP_SINGLE_ARGS(nowait) { coo_quickSort_parallel_internal(indices, array, 0, lenArray - 1, cutoff, rank); }
  }
}

template <typename T>
void SparseUtils<T>::sortCooIndicesGeneric(sd::LongType *indices, void *vx, sd::LongType length, int rank) {
  auto values = reinterpret_cast<T *>(vx);
#ifdef _OPENMP
  coo_quickSort_parallel(indices, values, length, omp_get_max_threads(), rank);
#else
  coo_quickSort_parallel(indices, values, length, 1, rank);
#endif
}

BUILD_SINGLE_TEMPLATE(template class SparseUtils, , SD_COMMON_TYPES);

void IndexUtils::ravelMultiIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                                 sd::LongType *shapeInfo, int mode) {
  sd::LongType *shape = shape::shapeOf(shapeInfo);
  sd::LongType *stride = shape::stride(shapeInfo);
  sd::LongType rank = shape::rank(shapeInfo);
  int errorCount = 0;

  PRAGMA_OMP_PARALLEL_FOR
  for (sd::LongType i = 0; i < length; ++i) {
    sd::LongType raveledIndex = 0;
    for (sd::LongType j = 0; j < rank; ++j) {
      sd::LongType idx = indices[i * rank + j];
      if (idx >= shape[j]) {
        // index does not fit into shape at j dimension.
        if (mode == ND4J_CLIPMODE_CLIP) {
          // set idx to largest possible value (clip to shape)
          idx = shape[j] - 1;
        } else if (mode == ND4J_CLIPMODE_WRAP) {
          idx %= shape[j];
        } else {
          // mode is ND4J_CLIPMODE_THROW or is unknown. either way. throw an error later.
          // cannot throw here because of parallel region
          sd_printf(
              "sparse::IndexUtils::ravelMultiIndex Cannot ravel index at element %d, does not fit into specified "
              "shape.\n",
              i);
          ++errorCount;
        }
      }
      raveledIndex += idx * stride[j];
    }
    flatIndices[i] = raveledIndex;
  }

  if (errorCount > 0) {
    // throw error if one ocurred in loop
    throw std::runtime_error("sparse::IndexUtils::ravelMultiIndex Cannot ravel index");
  }
}

void IndexUtils::unravelIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                              sd::LongType *shapeInfo) {
  sd::LongType *shape = shape::shapeOf(shapeInfo);
  sd::LongType *stride = shape::stride(shapeInfo);
  sd::LongType rank = shape::rank(shapeInfo);
  int errorCount = 0;

  // unravelOrder ensures that the dimensions with largest stride are unraveled first.
  // create vector with elements 0..rank
  int *unravelOrder = shape::range<int>(0, rank);

  // sort order according to stride length.
  std::sort(unravelOrder, unravelOrder + rank, [&](int i1, int i2) { return stride[i1] > stride[i2]; });

  // calculate the largest raveled index that will fit into passed shape
  sd::LongType maxRaveledIndex = shape[unravelOrder[0]] * stride[unravelOrder[0]] - 1;

  PRAGMA_OMP_PARALLEL_FOR
  for (sd::LongType i = 0; i < length; ++i) {
    sd::LongType raveledIndex = flatIndices[i];
    if (raveledIndex > maxRaveledIndex) {
      // cannot throw here because of parallel region
      sd_printf(
          "sparse::IndexUtils::unravelIndex Cannot unravel index at element %d. raveled index of %d does not fit into "
          "specified shape.\n",
          i, raveledIndex);
      ++errorCount;
    }

    for (int *it = unravelOrder; it != unravelOrder + rank; it++) {
      int j = *it;
      // how many strides of this size?
      indices[i * rank + j] = raveledIndex / stride[j];

      // remainder for subsequent smaller strides.
      raveledIndex %= stride[j];
    }
  }

  if (errorCount > 0) {
    // throw error if one ocurred in loop
    sd_printf("Largest raveled index is: %d, ", maxRaveledIndex) std::vector<sd::LongType> v(shape, shape + rank);
    sd_printv("Shape: ", v);
    throw std::runtime_error("sparse::IndexUtils::unravelIndex Cannot unravel index");
  }

  delete[] unravelOrder;
}
}  // namespace sparse
}  // namespace sd
