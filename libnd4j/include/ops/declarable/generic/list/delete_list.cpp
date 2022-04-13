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
#if NOT_EXCLUDED(OP_delete_list)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
LIST_OP_IMPL(delete_list, -2, 1, 0, -2) {
  auto list = INPUT_LIST(0);
  auto output = OUTPUT_VARIABLE(0);
  int idx = -1;
  // nd4j mode
  if (block.width() >= 2) {
    auto idxArr = INPUT_VARIABLE(block.width() - 1);

    REQUIRE_TRUE(idxArr->isScalar(), 0, "Index should be Scalar");
    idx = idxArr->e<int>(0);

  }

  //allow negative indexing from the end
  if(idx < 0) {
    idx += list->elements();
  }


  list->remove(idx);
  auto result = list->remove(idx);
  output->assign(result);
  return sd::Status::OK;


}

}  // namespace ops
}  // namespace sd

#endif
