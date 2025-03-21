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
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_testop2i2o)

#include <ops/declarable/headers/tests.h>

namespace sd {
namespace ops {
//////////////////////////////////////////////////////////////////////////
// test op, non-divergent
OP_IMPL(testop2i2o, 2, 2, true) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  auto xO = OUTPUT_VARIABLE(0);
  auto yO = OUTPUT_VARIABLE(1);

  x->applyScalar(scalar::Add, 1.0, xO);
  y->applyScalar(scalar::Add, 2.0, yO);

  STORE_2_RESULTS(*xO, *yO);

  return Status::OK;
}
DECLARE_SYN(TestOp2i2o, testop2i2o);

DECLARE_TYPES(testop2i2o) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }
}  // namespace ops
}  // namespace sd

#endif
