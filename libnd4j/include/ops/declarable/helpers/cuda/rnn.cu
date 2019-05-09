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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 16.04.2018
//

// function nnCell implements an Elman RNN cell: output = activation(Wx*x + bx  +  Wh*ht  + bh)

#include<ops/declarable/helpers/rnn.h>
#include <helpers/BlasHelper.h>


namespace nd4j    {
namespace ops     {
namespace helpers {


    //////////////////////////////////////////////////////////////////////////
    static FORCEINLINE NDArray activation(const NDArray& arr) {
        return (const_cast<NDArray&>(arr)).transform(transform::Tanh);
    }


    //////////////////////////////////////////////////////////////////////////
    void rnnCell(nd4j::LaunchContext * context, const NDArray* xt, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* ht_1, NDArray* ht) {

    }


    //////////////////////////////////////////////////////////////////////////
    void rnnTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* h0, const NDArray* maxTimeStep, NDArray* h, NDArray* hFinal) {

    }

}
}
}

