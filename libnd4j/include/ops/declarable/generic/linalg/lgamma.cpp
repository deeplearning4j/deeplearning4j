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
// @author George A. Shulinok <sgazeos@gmail.com>
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lgamma)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lgamma.h>

namespace sd {
namespace ops {

OP_IMPL(lgamma, 1, 1, true) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);
  helpers::lgamma(block.launchContext(), x, z);

  return Status::OK;
}

DECLARE_TYPES(lgamma) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS})  // as TF says
      ->setSameMode(true);
}

}  // namespace ops
}  // namespace sd

#endif
