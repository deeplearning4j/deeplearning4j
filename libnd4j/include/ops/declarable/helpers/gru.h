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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.02.2018
//

#ifndef LIBND4J_GRU_H
#define LIBND4J_GRU_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void gruCell(LaunchContext* context, NDArray* x, NDArray* hLast, NDArray* Wru,
                           NDArray* Wc, NDArray* bru, NDArray* bc, NDArray* r, NDArray* u, NDArray* c,
                           NDArray* h);

SD_LIB_HIDDEN void gruCell(NDArray* x, NDArray* hLast, NDArray* Wru, NDArray* Wc,
                           NDArray* b, NDArray* gates, NDArray* h, bool linearBeforeReset);

SD_LIB_HIDDEN void gruTimeLoop(LaunchContext* context, NDArray* x, NDArray* h0, NDArray* Wx,
                               NDArray* Wh, NDArray* b, NDArray* h, bool linearBeforeReset);

SD_LIB_HIDDEN void gruCellBp(LaunchContext* context, NDArray* x, NDArray* hLast, NDArray* W,
                             NDArray* Wc, NDArray* b, NDArray* bc, NDArray* dLdr,
                             NDArray* dLdu, NDArray* dLdc, NDArray* dLdh, NDArray* dLdx,
                             NDArray* dLdhLast, NDArray* dLdW, NDArray* dLdWc, NDArray* dLdb, NDArray* dLdbc);

SD_LIB_HIDDEN void gruCellBp(LaunchContext* context, NDArray* x, NDArray* hI, NDArray* Wx,
                             NDArray* Wh, NDArray* b, NDArray* dLdh, NDArray* gates,
                             NDArray* dLdx, NDArray* dLdhI, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb);

SD_LIB_HIDDEN void gruTimeLoopBp(LaunchContext* context, NDArray* x, NDArray* hI, NDArray* Wx,
                                 NDArray* Wh, NDArray* b, NDArray* dLdh, NDArray* dLdx,
                                 NDArray* dLdhI, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_GRU_H
