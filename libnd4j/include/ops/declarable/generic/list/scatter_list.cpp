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
#if NOT_EXCLUDED(OP_scatter_list)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
LIST_OP_IMPL(scatter_list, 1, 1, 0, -2) {
  NDArrayList *list = nullptr;
  NDArray *array = nullptr;
  NDArray *indices = nullptr;

  bool hasList = false;
  auto w = block.width();

  if (w >= 3) {
    list = INPUT_LIST(0);
    indices = INPUT_VARIABLE(1);
    array = INPUT_VARIABLE(2);
    hasList = true;
  } else {
    array = INPUT_VARIABLE(1);
    indices = INPUT_VARIABLE(2);
    list = new NDArrayList(indices->lengthOf(), false);
    block.trackList(list);
  }

  REQUIRE_TRUE(indices->isVector() || indices->rankOf() == 1, 0, "ScatterList: Indices for Scatter should be a vector")
  REQUIRE_TRUE(indices->lengthOf() == array->sizeAt(0), 0,
               "ScatterList: Indices length should be equal number of TADs along dim0, but got %i instead",
               indices->lengthOf());

  std::vector<sd::LongType> zero;
  zero.push_back(0);
  std::vector<LongType> *axis = ShapeUtils::evalDimsToExclude(array->rankOf(),1,zero.data());
  auto tads = array->allTensorsAlongDimension(*axis);
  for (sd::LongType e = 0; e < tads.size(); e++) {
    auto idx = indices->e<sd::LongType>(e);
    if (idx >= tads.size()) return sd::Status::BAD_ARGUMENTS;

    auto arr = new NDArray(tads.at(e)->dup(array->ordering()));
    auto res = list->write(idx, arr);


    if (res != sd::Status::OK) {
      delete axis;
      return res;
    }
  }



  if (!hasList)
    setupResultList(list, block);

  delete axis;


  return sd::Status::OK;
}
DECLARE_SYN(TensorArrayScatterV3, scatter_list);
DECLARE_SYN(tensorarrayscatterv3, scatter_list);
}  // namespace ops
}  // namespace sd

#endif
