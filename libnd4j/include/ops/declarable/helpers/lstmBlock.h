/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma, created on 14.02.2018
//

#ifndef LIBND4J_LSTM_H
#define LIBND4J_LSTM_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


	void lstmBlockCell(const NDArray* xt, const NDArray* cLast, const NDArray* yLast,
					   const NDArray* W, const NDArray* Wci, const NDArray* Wcf, const NDArray* Wco, const NDArray* b,
					   NDArray* i, NDArray* c, NDArray* f, NDArray* o, NDArray* z, NDArray* h, NDArray* y, const std::vector<double>& params);

    void lstmBlockTimeLoop(const NDArray* maxSeqLength, const NDArray* xSeq, const NDArray* c0, const NDArray* y0,
                           const NDArray* W, const NDArray* Wci, const NDArray* Wcf, const NDArray* Wco, const NDArray* b,
                           const NDArray* iSeq, const NDArray* cSeq, const NDArray* fSeq, const NDArray* oSeq, const NDArray* zSeq,
                           const NDArray* hSeq, const NDArray* ySeq, const std::vector<double>& params, const int dataFormat);

}
}
}


#endif //LIBND4J_LSTM_H
