/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

    //////////////////////////////////////////////////////////////////////////
    static FORCEINLINE NDArray sigmoid(const NDArray& arr) {
        return (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
    }

    static FORCEINLINE void sigmoidInplace(const NDArray& arr) {
        (const_cast<NDArray&>(arr)).applyTransform(transform::Sigmoid);
    }

//////////////////////////////////////////////////////////////////////////
    static FORCEINLINE NDArray tanh(const NDArray& arr) {
        return (const_cast<NDArray&>(arr)).transform(transform::Tanh);
    }

    static FORCEINLINE void tanhInplace(const NDArray& arr) {
        (const_cast<NDArray&>(arr)).applyTransform(transform::Tanh);
    }

//////////////////////////////////////////////////////////////////////////
    static NDArray timeSubset(const NDArray* arr, const int t, const int dataFormat){

        if(dataFormat == 0) { // TNS: shape [timeLength, numExamples, inOutSize]
            return (*arr)({t,t+1, 0,0, 0,0});
        }
        else if(dataFormat == 1) {   //NST: shape [numExamples, inOutSize, timeLength]
            return (*arr)({0,0, 0,0, t,t+1});
        }
        else {          //NTS: shape [numExamples, timeLength, inOutSize] - TF "time_major=false" layout
            return (*arr)({0,0, t,t+1, 0,0});
        }
    }

	void lstmCell(nd4j::LaunchContext * context, const NDArray* xt, const NDArray* ht_1, const NDArray* ct_1, const NDArray* Wx, const NDArray* Wh, const NDArray* Wc, const NDArray* Wp, const NDArray* b,
                  NDArray* ht, NDArray* ct, const std::vector<double>& params);

	void lstmTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* h0, const NDArray* c0, const NDArray* Wx, const NDArray* Wh, const NDArray* Wc, const NDArray* Wp, const NDArray* b,
                      NDArray* h, NDArray* c, const std::vector<double>& params);

    void lstmBlockCell(const NDArray* xt, const NDArray* cLast, const NDArray* yLast,
                       const NDArray* W, const NDArray* Wci, const NDArray* Wcf, const NDArray* Wco, const NDArray* b,
                       NDArray* i, NDArray* c, NDArray* f, NDArray* o, NDArray* z, NDArray* h, NDArray* y, const std::vector<double>& params);



}
}
}


#endif //LIBND4J_LSTM_H
