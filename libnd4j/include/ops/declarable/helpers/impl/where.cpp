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
// Created by raver119 on 24/09/18.
//
#include <array/NDArrayList.h>
#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_Where)
#include <ops/declarable/helpers/where.h>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static void __where(NDArray &condition, NDArray &output, memory::Workspace *workspace) {
  NDArrayList list(0, true);
  sd::LongType cnt = 0;


  for (sd::LongType e = 0; e < condition.lengthOf(); e++) {
    sd::LongType coords[SD_MAX_RANK];

    INDEX2COORDS(e, condition.rankOf(), condition.shapeOf(), coords);
    sd::LongType offset;
    COORDS2INDEX(condition.rankOf(), shape::stride(condition.shapeInfo()), coords, offset);

    if (condition.e<bool>(offset)) {
      std::vector<sd::LongType> arrShape = {1, condition.rankOf()};
      auto array = NDArrayFactory::create_('c', arrShape, output.dataType(), output.getContext());
      for (int f = 0; f < condition.rankOf(); f++) array->p(f, (T)coords[f]);

      list.write(cnt++, array);
    }
  }

  auto s = list.stack();
  output.assign(s);
  delete s;
}
BUILD_SINGLE_TEMPLATE(template void __where, (NDArray & condition, NDArray &output, memory::Workspace *workspace),
                      SD_COMMON_TYPES);

void _where(sd::LaunchContext *context, NDArray &condition, NDArray &output, memory::Workspace *workspace) {
  condition.syncToHost();
  BUILD_SINGLE_SELECTOR(output.dataType(), __where, (condition, output, workspace), SD_COMMON_TYPES);
  output.syncToDevice();
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif