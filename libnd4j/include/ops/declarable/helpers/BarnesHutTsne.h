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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019
//

#ifndef LIBND4J_BARNES_HUT_TSNE_HELPERS_H
#define LIBND4J_BARNES_HUT_TSNE_HELPERS_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN LongType barnes_row_count(NDArray* rowP, NDArray* colP, LongType N,
                                            NDArray& rowCounts);
SD_LIB_HIDDEN void barnes_symmetrize(NDArray* rowP, NDArray* colP, NDArray* valP, LongType N,
                                     NDArray* outputRows, NDArray* outputCols, NDArray* outputVals,
                                     NDArray* rowCounts = nullptr);
SD_LIB_HIDDEN void barnes_edge_forces(NDArray* rowP, NDArray * colP, NDArray * valP, int N,
                                      NDArray* output, NDArray& data);
SD_LIB_HIDDEN void barnes_gains(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output);
SD_LIB_HIDDEN bool cell_contains(NDArray* corner, NDArray* width, NDArray* point, LongType dimension);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_ACTIVATIONS_H
