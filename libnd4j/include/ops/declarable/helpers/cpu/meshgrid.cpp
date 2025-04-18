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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.04.2018
//

#include <array/ResultSet.h>
#include <ops/declarable/helpers/meshgrid.h>

#include <numeric>
#if NOT_EXCLUDED(OP_meshgrid)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////

void meshgrid(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs,
              const bool swapFirst2Dims) {
  const int rank = inArrs.size();
  int inIndices[SD_MAX_RANK];
  std::iota(inIndices, inIndices + rank, 0);
  if (swapFirst2Dims && rank > 1) {
    inIndices[0] = 1;
    inIndices[1] = 0;
  }

  for (int i = 0; i < rank; ++i) {
    auto list = outArrs[i]->allTensorsAlongDimension({inIndices[i]});
    for (int j = 0; j < list.size(); ++j) list.at(j)->assign(inArrs[i]);
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif