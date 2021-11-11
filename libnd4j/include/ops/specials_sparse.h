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
// This file contains various helper methods/classes suited for sparse support
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIALS_SPARSE_H
#define LIBND4J_SPECIALS_SPARSE_H

#include <system/common.h>

#define ND4J_CLIPMODE_THROW 0
#define ND4J_CLIPMODE_WRAP 1
#define ND4J_CLIPMODE_CLIP 2

namespace sd {
namespace sparse {

template <typename T>
class SD_LIB_EXPORT SparseUtils {
 public:
  /**
   * Just simple helper for debugging :)
   *
   * @param indices
   * @param rank
   * @param x
   */
  static void printIndex(sd::LongType *indices, int rank, int x);
  static bool ltIndices(sd::LongType *indices, int rank, sd::LongType x, sd::LongType y);

  /**
   * Returns true, if x > y, false otherwise
   * @param indices
   * @param rank
   * @param x
   * @param y
   * @return
   */
  static bool gtIndices(sd::LongType *indices, int rank, sd::LongType x, sd::LongType y);

  static void swapEverything(sd::LongType *indices, T *array, int rank, sd::LongType x, sd::LongType y);

  static void coo_quickSort_parallel_internal(sd::LongType *indices, T *array, sd::LongType left, sd::LongType right,
                                              int cutoff, int rank);

  static void coo_quickSort_parallel(sd::LongType *indices, T *array, sd::LongType lenArray, int numThreads, int rank);

  static sd::LongType coo_quickSort_findPivot(sd::LongType *indices, T *array, sd::LongType left, sd::LongType right,
                                              int rank);

  static void sortCooIndicesGeneric(sd::LongType *indices, void *vx, sd::LongType length, int rank);
};

class SD_LIB_EXPORT IndexUtils {
 public:
  /**
   * Converts indices in COO format into an array of flat indices
   *
   * based on numpy.ravel_multi_index
   */
  static void ravelMultiIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                              sd::LongType *shapeInfo, int mode);

  /**
   * Converts flat indices to index matrix in COO format
   *
   * based on numpy.unravel_index
   */
  static void unravelIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                           sd::LongType *shapeInfo);
};
}  // namespace sparse
}  // namespace sd

#endif  // LIBND4J_SPECIALS_SPARSE_H
