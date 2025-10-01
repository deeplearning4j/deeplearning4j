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
// Created by Yurii Shyrma on 12.01.2018
//

#ifndef LIBND4J_HHCOLPICQR_H
#define LIBND4J_HHCOLPICQR_H
#include <array/NDArray.h>
#include <helpers/hhColPivQR.h>

namespace sd {
namespace ops {
namespace helpers {

class HHcolPivQR {
 public:
  NDArray *_qr;
  NDArray *_coeffs;
  NDArray *_permut;
  int _diagSize;

  HHcolPivQR() = delete;
  HHcolPivQR(NDArray &matrix);

  template <typename T>
  void _evalData();

  void evalData();
};

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HHCOLPICQR_H
