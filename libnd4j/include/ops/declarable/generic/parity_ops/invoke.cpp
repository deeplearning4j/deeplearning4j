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
//  @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_invoke)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(invoke, 2, 2, false, 0, 0) {
  sd_printf("This operation is unimplemented and is present for only metadata purposes.\n",0);
  return Status::VALIDATION;
};

DECLARE_SHAPE_FN(invoke) {
  sd_printf("This shape function is unimplemented and is present for only metadata purposes.\n",0);
  return SHAPELIST();
}

DECLARE_TYPES(invoke) {
  getOpDescriptor()
      ->setAllowedInputTypes(ANY)
      ->setAllowedOutputTypes(ANY);
}
}  // namespace ops
}  // namespace sd

#endif
