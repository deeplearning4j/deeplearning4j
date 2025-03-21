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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_polygamma)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/gammaMathFunc.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(polygamma, 2, 1, false, 0, 0) {
  auto n = INPUT_VARIABLE(0);
  auto x = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(n->isSameShape(x), 0,
               "POLYGAMMA op: two input arrays n and x must have the same shapes, but got n=%s and x=%s instead !",
               ShapeUtils::shapeAsString(n).c_str(), ShapeUtils::shapeAsString(x).c_str());

  void* extraArgs = nullptr;
  auto nNegative = n->reduceNumber(reduce::IsNegative, extraArgs);
  auto xPositive = x->reduceNumber(reduce::IsPositive, extraArgs);
  bool nPositiveFlag = !nNegative.e<bool>(0);  // require all n >= 0
  bool xPositiveFlag = xPositive.e<bool>(0);   // require all x > 0
  REQUIRE_TRUE(nPositiveFlag, 0, "POLYGAMMA op: all elements of n array must be >= 0 !");
  REQUIRE_TRUE(xPositiveFlag, 0, "POLYGAMMA op: all elements of x array must be > 0 !");

  helpers::polyGamma(block.launchContext(), *n, *x, *output);
  return Status::OK;
}

DECLARE_SYN(polyGamma, polygamma);
DECLARE_SYN(PolyGamma, polygamma);

DECLARE_TYPES(polygamma) { getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true); }
}  // namespace ops
}  // namespace sd

#endif
