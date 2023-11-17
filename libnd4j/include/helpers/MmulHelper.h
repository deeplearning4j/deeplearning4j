/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.06.2018
//

#ifndef LIBND4J_MMULHELPER_H
#define LIBND4J_MMULHELPER_H
#include "array/NDArray.h"

namespace sd {
class SD_LIB_EXPORT MmulHelper {
 private:
  // multiptication N-dimensions tensor on other N-dimensions one
  static NDArray* mmulNxN(const NDArray* A, const NDArray* B, NDArray* C, const double alpha = 1.0,
                              const double beta = 0.0, const char outOrder = 'f');

  // dot product of vectors (X * Y) = Z[0]
  static NDArray* dot(const NDArray* X, const NDArray* Y, NDArray* Z, const double alpha = 1.0,
                          const double beta = 0.0);

  // multiptication Matrix to Matrix
  static NDArray* mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, double alpha = 1.0,
                              double beta = 0.0, const char outOrder = 'f');

  // multiptication Matrix to vector
  static NDArray* mmulMxV(const NDArray* A, const NDArray* B, NDArray* C, double alpha = 1.0,
                              double beta = 0.0, const char outOrder = 'f');

 public:
  static NDArray* mmul(const NDArray* A, const NDArray* B, NDArray* C = nullptr,
                           const double alpha = 1.0, const double beta = 0.0, const char outOrder = 'f');

  static NDArray* tensorDot(const NDArray* A, const NDArray* B,
                                const std::initializer_list<LongType>& axesA,
                                const std::initializer_list<LongType>& axesB = {});

  static NDArray* tensorDot(const NDArray* A, const NDArray* B, const std::vector<LongType>& axesA,
                                const std::vector<LongType>& axesB);

  static void tensorDot(const NDArray* a, const NDArray* b, NDArray* c, const std::vector<LongType>& axes_a,
                        const std::vector<LongType>& axes_b, const std::vector<LongType>& permutForC = {});

  static void computeNewShapesAndAxes(
      const NDArray& as_, const std::vector<LongType>& axes_a,
      const NDArray& bs, const std::vector<LongType>& axes_b,
      std::vector<LongType>& newshape_a, std::vector<LongType>& newaxes_a,
      std::vector<LongType>& newshape_b, std::vector<LongType>& newaxes_b
      );
#ifndef __JAVACPP_HACK__
  /**
   *  modif - (can be empty) vector containing a subsequence of permutation/reshaping arrays (in any order), user must
   * take care of correctness of such arrays by himself
   */
  static void tensorDot(const NDArray* a, const NDArray* b, NDArray* c,
                        const std::vector<std::vector<LongType>>& modifA,
                        const std::vector<std::vector<LongType>>& modifB,
                        const std::vector<std::vector<LongType>>& modifC);
  static NDArray* tensorDot(const NDArray* a, const NDArray* b,
                                const std::vector<std::vector<LongType>>& modifA,
                                const std::vector<std::vector<LongType>>& modifB);

  static void tensorDot2(const NDArray* a, const NDArray* b, NDArray* c,
                         const std::vector<LongType>& axes_a, const std::vector<LongType>& axes_b,
                         std::vector<LongType>& permutAt, std::vector<LongType>& permuteBt,
                         std::vector<LongType>& permuteCt);
#endif

  static void matmul(const NDArray* x, const NDArray* y, NDArray* z, const bool transX, const bool transY,
                     double alpha = 1.0, double beta = 0.0);
};
}  // namespace sd

#endif  // LIBND4J_MMULHELPER_H
