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
// Created by raver119 on 06.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_split_list)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
LIST_OP_IMPL(split_list, 2, 1, 0, -2) {
  NDArrayList *list = nullptr;
  NDArray *array = nullptr;
  NDArray *sizes = nullptr;

  bool hasList = false;

  if (block.width() >= 3) {
    list = INPUT_LIST(0);
    array = INPUT_VARIABLE(1);
    sizes = INPUT_VARIABLE(2);
    hasList = true;
  } else {
    array = INPUT_VARIABLE(0);
    sizes = INPUT_VARIABLE(1);
    list = new NDArrayList(sizes->lengthOf(), false);
    block.trackList(list);
  }

  REQUIRE_TRUE(sizes->isZ(), 0, "split_list: sizes array must have one of integer types");
  REQUIRE_TRUE(sizes->rankOf() == 1, 0, "split_list: sizes array must be 1D")

  auto* arrayShape = array->getShapeAsVector();
  list->shape() = *arrayShape;
  delete arrayShape;

  // now let's build subarrays
  int cnt = 0;
  std::vector<LongType> indices(2 * array->rankOf(), 0);
  for (LongType e = 0; e < sizes->lengthOf(); e++) {
    int c_size = sizes->e<int>(e);

    REQUIRE_TRUE(c_size > 0, 0, "Slice size should have postive value, but got %i instead", c_size);
    REQUIRE_TRUE(cnt < array->sizeAt(0) && cnt + c_size <= array->sizeAt(0), 0,
                 "Slices size should NOT be higher then number of TADs of source array. Source size: [%i]; Slice "
                 "start: [%i]; Slice size: [%i]",
                 array->sizeAt(0), cnt, c_size);

    // we're adding our interval along zeroth dimension
    indices[0] = cnt;
    indices[1] = cnt + c_size;
    cnt += c_size;

    auto subarray = (*array)(indices);

    auto status = list->write(e, new NDArray(subarray->dup(array->ordering(), false)));
    delete subarray;
    if (status != Status::OK) return status;
  }

  if (!hasList) {
    setupResultList(list, block);
  }

  return Status::OK;
}
DECLARE_SYN(TensorArraySplitV3, split_list);
DECLARE_SYN(tensorarraysplitv3, split_list);
}  // namespace ops
}  // namespace sd

#endif
