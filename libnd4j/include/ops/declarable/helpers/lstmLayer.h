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

namespace sd    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
void ND4J_EXPORT lstmLayerCell(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                   const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                   const std::vector<float>& params,
                         NDArray* h, NDArray* c);

//////////////////////////////////////////////////////////////////////////
// this auxiliary ff should be running before backprop
void ND4J_EXPORT lstmLayerCell(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                   const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                   const std::vector<float>& params,
                   NDArray* z, NDArray* a, NDArray* h, NDArray* c);

//////////////////////////////////////////////////////////////////////////
void ND4J_EXPORT lstmLayerCellBp(const NDArray* x, const NDArray* Wx, const NDArray* Wr, const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                     const NDArray* dLdh, const NDArray* dLdhL, const NDArray* dLdcL,
                     const NDArray* z, const NDArray* a, const NDArray* c, const std::vector<float>& params,
                     NDArray* dLdx, NDArray* dLdWx, NDArray* dLdWr, NDArray* dLdhI, NDArray* dLdcI, NDArray* dLdb, NDArray* dLdWp);


//////////////////////////////////////////////////////////////////////////
void ND4J_EXPORT lstmLayerTimeLoop(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                        const NDArray* b, const NDArray* seqLen, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                        const std::vector<float>& params,
                        const bool forward,
                        NDArray* h, NDArray* hL, NDArray* cL);

//////////////////////////////////////////////////////////////////////////
void ND4J_EXPORT lstmLayerTimeLoopBp(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                        const NDArray* b, const NDArray* seqLen, NDArray* hI, NDArray* cI, const NDArray* Wp,
                        const NDArray* dLdh, const NDArray* dLdhL, const NDArray* dLdcL,
                        const std::vector<float>& params, const bool forward,
                        NDArray* dLdx, NDArray* dLdWx, NDArray* dLdWr, NDArray* dLdb, NDArray* dLdhI, NDArray* dLdcI, NDArray* dLdWp);


}
}
}


#endif //LIBND4J_LSTMLAYER_H
