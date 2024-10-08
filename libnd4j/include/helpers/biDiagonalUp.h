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
// Created by Yurii Shyrma on 18.12.2017.
//

#ifndef LIBND4J_BIDIAGONALUP_H
#define LIBND4J_BIDIAGONALUP_H
#include <array/NDArray.h>
#include <helpers/hhSequence.h>

namespace sd {
namespace ops {
namespace helpers {

class BiDiagonalUp {
 public:
  NDArray _HHmatrix;  // 2D Householder matrix
  NDArray _HHbidiag;  // vector which contains Householder coefficients
  NDArray _hhCoeffs;  // vector of Householder coefficients

  /**
   *  constructor
   *
   *  matrix - input matrix expected to be bi-diagonalized, remains unaffected
   */
  BiDiagonalUp(NDArray& matrix);

  /**
   *  this method evaluates data (coeff, normX, tail) used in Householder transformation
   *  formula for Householder matrix: P = identity_matrix - coeff * w * w^T
   *  P * x = [normX, 0, 0 , 0, ...]
   *  coeff - scalar
   *  w = [1, w1, w2, w3, ...], "tail" is w except first unity element, that is "tail" = [w1, w2, w3, ...]
   *  tail and coeff are stored in _HHmatrix
   *  normX are stored in _HHbidiag
   */
  template <typename T>
  void _evalData();

  void evalData();

  /**
   *  this method evaluates product of Householder sequence matrices (transformations) acting on columns
   *
   *  type - type of sequence, type = 'u' (acting on columns) or type = 'v' (acting on rows)
   */
  template <typename T>
  HHsequence makeHHsequence_(const char type);

  HHsequence makeHHsequence(const char type);
};

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_BIDIAGONALUP_H
