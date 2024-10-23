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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_LSTMLAYER_H
#define LIBND4J_LSTMLAYER_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
SD_LIB_HIDDEN void lstmLayerCell(NDArray* x, NDArray* Wx, NDArray* Wr, NDArray* b,
                                 NDArray* hI, NDArray* cI, NDArray* Wp,
                                 const std::vector<float>& params, NDArray* h, NDArray* c);

//////////////////////////////////////////////////////////////////////////
// this auxiliary ff should be running before backprop
SD_LIB_HIDDEN void lstmLayerCell(NDArray* x, NDArray* Wx, NDArray* Wr, NDArray* b,
                                 NDArray* hI, NDArray* cI, NDArray* Wp,
                                 const std::vector<float>& params, NDArray* z, NDArray* a, NDArray* h, NDArray* c);

//////////////////////////////////////////////////////////////////////////
SD_LIB_HIDDEN void lstmLayerCellBp(NDArray* x, NDArray* Wx, NDArray* Wr, NDArray* b,
                                   NDArray* hI, NDArray* cI, NDArray* Wp, NDArray* dLdh,
                                   NDArray* dLdhL, NDArray* dLdcL, NDArray* z, NDArray* a,
                                   NDArray* c, const std::vector<float>& params, NDArray* dLdx, NDArray* dLdWx,
                                   NDArray* dLdWr, NDArray* dLdhI, NDArray* dLdcI, NDArray* dLdb, NDArray* dLdWp);

//////////////////////////////////////////////////////////////////////////
SD_LIB_HIDDEN void lstmLayerTimeLoop(NDArray* x, NDArray* Wx, NDArray* Wr, NDArray* b,
                                     NDArray* seqLen, NDArray* hI, NDArray* cI, NDArray* Wp,
                                     const std::vector<float>& params, const bool forward, NDArray* h, NDArray* hL,
                                     NDArray* cL);

//////////////////////////////////////////////////////////////////////////
SD_LIB_HIDDEN void lstmLayerTimeLoopBp(NDArray* x, NDArray* Wx, NDArray* Wr, NDArray* b,
                                       NDArray* seqLen, NDArray* hI, NDArray* cI, NDArray* Wp,
                                       NDArray* dLdh, NDArray* dLdhL, NDArray* dLdcL,
                                       const std::vector<float>& params, const bool forward, NDArray* dLdx,
                                       NDArray* dLdWx, NDArray* dLdWr, NDArray* dLdb, NDArray* dLdhI, NDArray* dLdcI,
                                       NDArray* dLdWp);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LSTMLAYER_H
