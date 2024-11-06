/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma, created on 14.02.2018
//

#ifndef LIBND4J_LSTM_H
#define LIBND4J_LSTM_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void lstmBlockCell(NDArray* xt, NDArray* cLast, NDArray* yLast, NDArray* W,
                                 NDArray* Wci, NDArray* Wcf, NDArray* Wco, NDArray* b,
                                 NDArray* i, NDArray* c, NDArray* f, NDArray* o, NDArray* z, NDArray* h, NDArray* y,
                                 const std::vector<double>& params);

SD_LIB_HIDDEN void lstmBlockTimeLoop(NDArray* maxSeqLength, NDArray* xSeq, NDArray* c0,
                                     NDArray* y0, NDArray* W, NDArray* Wci, NDArray* Wcf,
                                     NDArray* Wco, NDArray* b, NDArray* iSeq, NDArray* cSeq,
                                     NDArray* fSeq, NDArray* oSeq, NDArray* zSeq, NDArray* hSeq,
                                     NDArray* ySeq, const std::vector<double>& params, const int dataFormat);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LSTM_H
