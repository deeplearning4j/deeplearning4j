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
// Created by Yurii Shyrma on 02.01.2018
//

#ifndef LIBND4J_HHSEQUENCE_H
#define LIBND4J_HHSEQUENCE_H
#include "array/NDArray.h"

namespace sd {
namespace ops {
namespace helpers {

class HHsequence {
 public:
  /*
   *  matrix containing the Householder vectors
   */
  NDArray& _vectors;

  /*
   *  vector containing the Householder coefficients
   */
  NDArray& _coeffs;

  /*
   *  shift of the Householder sequence
   */
  int _shift;

  /*
   *  length of the Householder sequence
   */
  int _diagSize;

  /*
   *  type of sequence, type = 'u' (acting on columns, left) or type = 'v' (acting on rows, right)
   */
  char _type;

  /*
   *  constructor
   */
  HHsequence(NDArray& vectors, NDArray& coeffs, const char type);
  HHsequence() = delete;
  /**
   *  this method mathematically multiplies input matrix on Householder sequence from the left H0*H1*...Hn * matrix
   *
   *  matrix - input matrix to be multiplied
   */
  template <typename T>
  void mulLeft_(NDArray& matrix);

  void mulLeft(NDArray& matrix);

  NDArray getTail(const int idx) const;

  template <typename T>
  void applyTo_(NDArray& dest);

  void applyTo(NDArray& dest);

  SD_INLINE int rows();
};

//////////////////////////////////////////////////////////////////////////
SD_INLINE int HHsequence::rows() { return _type == 'u' ? _vectors.sizeAt(0) : _vectors.sizeAt(1); }

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HHSEQUENCE_H
