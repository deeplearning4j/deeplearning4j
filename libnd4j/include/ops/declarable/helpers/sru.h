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
// @author Yurii Shyrma (iuriish@yahoo.com), created by on 06.04.2018
//

#ifndef LIBND4J_SRU_H
#define LIBND4J_SRU_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void sruCell(LaunchContext* context, NDArray* x, NDArray* c0, NDArray* w,
                           NDArray* b, NDArray* h, NDArray* c);

SD_LIB_HIDDEN void sruTimeLoop(LaunchContext* context, NDArray* x, NDArray* c0, NDArray* w,
                               NDArray* b, NDArray* h, NDArray* c);

SD_LIB_HIDDEN void sruBI(LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
                         NDArray* mask, NDArray* ht, NDArray* ct);

SD_LIB_HIDDEN void sruBIBP(LaunchContext* context, NDArray* x, NDArray* w, NDArray* b,
                           NDArray* c0, NDArray* ct, NDArray* inGradC0, NDArray* inGradH,
                           NDArray* mask, NDArray* gradI, NDArray* gradWeights, NDArray* gradB, NDArray* gradC0);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_SRU_H
