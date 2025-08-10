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
// Created by Yurii Shyrma on 12.12.2017
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/zeta.h>
#if NOT_EXCLUDED(OP_zeta)
namespace sd {
namespace ops {
namespace helpers {

const int maxIter = 1000000;  // max number of loop iterations

//////////////////////////////////////////////////////////////////////////
// slow implementation
template <typename T>
static SD_INLINE T zetaScalarSlow(const T x, const T q) {
  const T precision = (T)1e-7;  // function stops the calculation of series when next item is <= precision

  // if (x <= (T)1.)
  //     throw("zeta function: x must be > 1 !");

  // if (q <= (T)0.)
  //     throw("zeta function: q must be > 0 !");

  T item;
  T result = (T)0.;
  for (int i = 0; i < maxIter; ++i) {
    item = math::sd_pow((q + i), -x);
    result += item;

    if (item <= precision) break;
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////
// calculate the Hurwitz zeta function for arrays
template <typename T>
static void zeta_(sd::LaunchContext* context, NDArray& x, NDArray& q, NDArray& z) {
  // auto result = NDArray(&x, false, context);
  int xLen = x.lengthOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) z.p(i, zetaScalar<T>(x.e<T>(i), q.e<T>(i)));
  };

  samediff::Threads::parallel_for(func, 0, xLen);
}

void zeta(sd::LaunchContext* context, NDArray& x, NDArray& q, NDArray& z) {
  BUILD_SINGLE_SELECTOR(x.dataType(), zeta_, (context, x, q, z), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template  void zeta_,
                      (sd::LaunchContext * context, NDArray& x, NDArray& q, NDArray& z), SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif