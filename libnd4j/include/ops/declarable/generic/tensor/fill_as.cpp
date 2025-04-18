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
// Created by raver119 on 01.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_fill_as)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CONFIGURABLE_OP_IMPL(fill_as, 1, 1, true, 0, 0) {
  auto output = OUTPUT_VARIABLE(0);

  if (block.width() > 1) {
    auto s = INPUT_VARIABLE(0);
    output->assign(s);
  } else if (block.numT() > 0) {
    output->assign(T_ARG(0));
  } else if (block.numI() > 0) {
    output->assign(INT_ARG(0));
  }

  STORE_RESULT(output);

  return Status::OK;
}
DECLARE_SYN(filllike, fill_as);
DECLARE_SYN(fill_like, fill_as);

DECLARE_TYPES(fill_as) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }
}  // namespace ops
}  // namespace sd

#endif
