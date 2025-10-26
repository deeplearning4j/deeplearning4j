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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_GAMMAMATHFUNC_H
#define LIBND4J_GAMMAMATHFUNC_H
#include <ops/declarable/helpers/helpers.h>

#include "array/NDArray.h"

namespace sd {
namespace ops {
namespace helpers {

// calculate the digamma function for each element for array
SD_LIB_HIDDEN void diGamma(LaunchContext* context, NDArray& x, NDArray& z);

// calculate the polygamma function
SD_LIB_HIDDEN void polyGamma(LaunchContext* context, NDArray& n, NDArray& x, NDArray& z);

// calculate the digamma function for one element
// implementation is based on serial representation written in terms of the Hurwitz zeta function as polygamma =
// (-1)^{n+1} * n! * zeta(n+1, x)
template <typename T>
SD_HOST_DEVICE T diGammaScalar(T x) {
  const int xInt = static_cast<int>(x);

  // negative and zero
  if (x <= 0) {
    if (x == xInt)  // integer
      return DataTypeUtils::infOrMax<T>();
    else
      return diGammaScalar<T>(1 - x) -
             M_PI / math::sd_tan<T, T>(M_PI * x);  // use reflection formula psi(1-x) = psi(x) + pi*cot(pi*x)
  }

  // positive integer
  if (x == xInt &&
      xInt <= 20) {  // psi(n) = -Euler_Mascheroni_const + sum_from_k=1_to_n-1( 1/k ), for n = 1,2,3,...inf, we use this
                     // formula only for n <= 20 to avoid time consuming sum calculation for bigger n
    T result = static_cast<T>(-0.577215664901532);
    for (LongType i = 1; i <= xInt - 1; ++i) {
      result += static_cast<T>(1) / i;
    }
    return result;
  }

  // positive half-integer
  if (x - xInt == 0.5 && xInt <= 20) {  // psi(n+0.5) = -Euler_Mascheroni_const - 2*ln(2) + sum_from_k=1_to_n( 2/(2*k-1)
                                        // )    , for n = 1,2,3,...inf, we use this formula only for n <= 20 to avoid
                                        // time consuming sum calculation for bigger n
    T result = static_cast<T>(-0.577215664901532 - static_cast<T>(2) * math::sd_log<T, T>(static_cast<T>(2)));
    for (LongType i = 1; i <= xInt; ++i) {
      result += static_cast<T>(2) / (2 * i - 1);
    }
    return result;
  }

  // positive, smaller then 5; we should use number > 5 in order to have satisfactory accuracy in asymptotic expansion
  if (x < 5) return diGammaScalar<T>(1 + x) - static_cast<T>(1) / x;  // recurrence formula  psi(x) = psi(x+1) - 1/x.

  // *** other positive **** //

  // truncated expansion formula (from wiki)
  // psi(x) = log(x) - 1/(2*x) - 1/(12*x^2) + 1/(120*x^4) - 1/(252*x^6) + 1/(240*x^8) - 5/(660*x^10) + 691/(32760*x^12)
  // - 1/(12*x^14) + ...

  if (x >= (sizeof(T) > 4 ? 1.e16 : 1.e8))  // if x is too big take into account only log(x)
    return math::sd_log<T, T>(x);

  // coefficients used in truncated asymptotic expansion formula
  const T coeffs[7] = {-(T)1 / 12, (T)1 / 120, -(T)1 / 252, (T)1 / 240, -(T)5 / 660, (T)691 / 32760, -(T)1 / 12};
  // const T coeffs[7] = {-0.0833333333333333, 0.00833333333333333, -0.00396825396825397, 0.00416666666666667,
  // -0.00757575757575758, 0.0210927960927961, -0.0833333333333333};

  const T x2Inv = static_cast<T>(1) / (x * x);
  T result = static_cast<T>(0);

  for (int i = 6; i >= 0; --i) result = (result + coeffs[i]) * x2Inv;
  return result + math::sd_log<T, T>(x) - static_cast<T>(0.5) / x;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_GAMMAMATHFUNC_H
