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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_Floor)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
OP_IMPL(Floor, 1, 1, true) {
  auto first = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  first->applyTransform(transform::Floor, z);

  STORE_RESULT(*z);

  return sd::Status::OK;
}
DECLARE_SYN(floor, Floor);

DECLARE_TYPES(Floor) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setSameMode(true); }
}  // namespace ops
}  // namespace sd

#endif
