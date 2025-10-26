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
// @author GS (sgazeos@gmail.com), created on 10/1/2018
//

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/cross.h>
#if NOT_EXCLUDED(OP_cross)
namespace sd {
namespace ops {
namespace helpers {

void crossBatched(sd::LaunchContext *context, NDArray *a, NDArray *b, NDArray *o) {
  std::vector<sd::LongType> shape2= {-1,3};
  auto _a = a->reshape(a->ordering(), shape2);
  auto _b = b->reshape(b->ordering(), shape2);
  auto _o = o->reshape(o->ordering(), shape2, false);

  auto tadsA = _a->allTensorsAlongDimension({1});
  auto tadsB = _b->allTensorsAlongDimension({1});
  auto tadsO = _o->allTensorsAlongDimension({1});

  int tads = tadsA.size();

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      auto a_ = tadsA.at(e);
      auto b_ = tadsB.at(e);
      auto o_ = tadsO.at(e);

      helpers::cross(context, a_, b_, o_);
    }
  };

  samediff::Threads::parallel_tad(func, 0, tads);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif