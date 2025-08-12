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
// Created by Yurii Shyrma on 11.12.2017
//
#include <array/DataTypeUtils.h>
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <ops/declarable/helpers/betaInc.h>

#include <cmath>
#if NOT_EXCLUDED(OP_betainc)
namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// modified Lentz’s algorithm for continued fractions,
// reference: Lentz, W.J. 1976, “Generating Bessel Functions in Mie Scattering Calculations Using Continued Fractions”

template <typename T>
static T continuedFraction(const T a, const T b, const T x) {
  const T min = DataTypeUtils::min_positive<T>() / static_cast<T>(DataTypeUtils::eps<T>());
  const T aPlusb = a + b;
  T val, aPlus2i;

  T t2 = static_cast<T>(1);
  T t1 = static_cast<T>(1) - aPlusb * x / (a + static_cast<T>(1));
  if (math::sd_abs<T,T>(t1) < min) t1 = min;
  t1 = static_cast<T>(1) / t1;
  T result = t1;

  for (sd::LongType i = 1; i <= maxIter; ++i) {
    aPlus2i = a + static_cast<T>(2 * i);
    val = i * (b - i) * x / ((aPlus2i - static_cast<T>(1)) * aPlus2i);
    // t1
    t1 = static_cast<T>(1) + val * t1;
    if (math::sd_abs<T,T>(t1) < min) t1 = min;
    t1 = static_cast<T>(1) / t1;
    // t2
    t2 = static_cast<T>(1) + val / t2;
    if (math::sd_abs<T,T>(t2) < min) t2 = min;
    // result
    result *= t2 * t1;
    val = -(a + i) * (aPlusb + i) * x / ((aPlus2i + static_cast<T>(1)) * aPlus2i);
    // t1
    t1 = static_cast<T>(1) + val * t1;
    if (math::sd_abs<T,T>(t1) < min) t1 = min;
    t1 = static_cast<T>(1) / t1;
    // t2
    t2 = static_cast<T>(1) + val / t2;
    if (math::sd_abs<T,T>(t2) < min) t2 = min;
    // result
    val = t2 * t1;
    result *= val;

    // condition to stop loop
    if (math::sd_abs<T,T>(val - static_cast<T>(1)) <= DataTypeUtils::eps<T>()) return result;
  }

  return DataTypeUtils::infOrMax<T>();  // no convergence, more iterations is required, return infinity
}

///////////////////////////////////////////////////////////////////
// evaluates incomplete beta function for positive a and b, and x between 0 and 1.
template <typename T>
static T betaIncCore(T a, T b, T x) {
  // t^{n-1} * (1 - t)^{n-1} is symmetric function with respect to x = 0.5
  if (a == b && x == static_cast<T>(0.5)) return static_cast<T>(0.5);

  if (x == static_cast<T>(0) || x == static_cast<T>(1)) return x;

  const T gammaPart = static_cast<T>(lgamma(a) + lgamma(b) - lgamma(a + b));
  const T front = math::sd_exp<T, T>(math::sd_log<T, T>(x) * a + math::sd_log<T, T>(1.f - x) * b - gammaPart);

  if (x <= (a + static_cast<T>(1)) / (a + b + static_cast<T>(2)))
    return front * continuedFraction<T>(a, b, x) / a;
  else  // symmetry relation
    return static_cast<T>(1) - front * continuedFraction<T>(b, a, static_cast<T>(1) - x) / b;
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void betaIncForArray(sd::LaunchContext* context, NDArray& a, NDArray& b, NDArray& x,
                            NDArray& output) {
  int xLen = x.lengthOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) output.r<T>(i) = betaIncCore<T>(a.t<T>(i), b.t<T>(i), x.t<T>(i));
  };

  samediff::Threads::parallel_for(func, 0, xLen);
}

///////////////////////////////////////////////////////////////////
// overload betaInc for arrays, shapes of a, b and x must be the same !!!
void betaInc(sd::LaunchContext* context, NDArray& a, NDArray& b, NDArray& x, NDArray& output) {
  auto xType = a.dataType();
  BUILD_SINGLE_SELECTOR(xType, betaIncForArray, (context, a, b, x, output), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void betaIncForArray,
                      (sd::LaunchContext * context, NDArray& a, NDArray& b, NDArray& x,
                       NDArray& output),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif