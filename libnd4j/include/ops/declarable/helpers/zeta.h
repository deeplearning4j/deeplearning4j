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
// Created by Yurii Shyrma on 12.12.2017.
//

#ifndef LIBND4J_ZETA_H
#define LIBND4J_ZETA_H
#include <ops/declarable/helpers/helpers.h>

#include "array/NDArray.h"

namespace sd {
namespace ops {
namespace helpers {

// calculate the Hurwitz zeta function for arrays
SD_LIB_HIDDEN void zeta(LaunchContext* context, NDArray& x, NDArray& q, NDArray& output);

// calculate the Hurwitz zeta function for scalars
// fast implementation, it is based on Euler-Maclaurin summation formula
template <typename T>
SD_LIB_HIDDEN SD_HOST_DEVICE T zetaScalar(const T x, const T q) {
  const T machep = T(1.11022302462515654042e-16);

  // FIXME: @raver119
  // expansion coefficients for Euler-Maclaurin summation formula (2k)! / B2k, where B2k are Bernoulli numbers
  const T coeffZeta[] = {T(12.0),
                         T(-720.0),
                         T(30240.0),
                         T(-1209600.0),
                         T(47900160.0),
                         T(-1.8924375803183791606e9),
                         T(7.47242496e10),
                         T(-2.950130727918164224e12),
                         T(1.1646782814350067249e14),
                         T(-4.5979787224074726105e15),
                         T(1.8152105401943546773e17),
                         T(-7.1661652561756670113e18)};

  T a, b = T(0.0), k, s, t, w;

  s = math::sd_pow<T, T, T>(q, -x);
  a = q;
  int i = 0;

  while (i < 9 || a <= T(9.0)) {
    i += 1;
    a += T(1.0);
    b = math::sd_pow<T, T, T>(a, -x);
    s += b;
    if (math::sd_abs<T,T>(b / s) < machep) return s;
  }

  w = a;
  s += b * (w / (x - T(1.0)) - T(0.5));
  a = T(1.0);
  k = T(0.0);

  for (i = 0; i < 12; ++i) {
    a *= x + k;
    b /= w;
    t = a * b / coeffZeta[i];
    s += t;
    t = math::sd_abs<T,T>(t / s);

    if (t < machep) return s;

    k += T(1.0);
    a *= x + k;
    b /= w;
    k += T(1.0);
  }

  return s;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_ZETA_H
