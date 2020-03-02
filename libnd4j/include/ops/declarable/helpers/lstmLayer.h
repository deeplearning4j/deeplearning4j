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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_LSTMLAYER_H
#define LIBND4J_LSTMLAYER_H

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/activations.h>

namespace sd    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
void ND4J_EXPORT lstmLayerCell(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                   const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                   const std::vector<float>& params,
                         NDArray* h, NDArray* c);

//////////////////////////////////////////////////////////////////////////
void ND4J_EXPORT lstmLayerTimeLoop(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                        const NDArray* b, const NDArray* seqLen, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                        const std::vector<float>& params,
                        const bool forward,
                        NDArray* h, NDArray* hL, NDArray* cL);

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE void applyActivation(NDArray& x, const int opId, const float alpha, const float beta, NDArray& z) {

    switch (opId) {
        case 0:
            (const_cast<NDArray&>(x)).applyTransform(transform::Tanh, z);
            break;
        case 1:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::RELU, 0, z);
            break;
        case 2:
            (const_cast<NDArray&>(x)).applyTransform(transform::Sigmoid, z);
            break;
        case 3: {
            ExtraArguments args({ static_cast<double>(alpha), static_cast<double>(beta)});
            (const_cast<NDArray&>(x)).applyTransform(transform::Affine, z, &args);
            break;
        }
        case 4:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::LeakyRELU, alpha, z);
            break;
        case 5:
            helpers::thresholdRelu(x.getContext(), x, alpha, z);
            break;
        case 6: {
            ExtraArguments args({ static_cast<double>(alpha), static_cast<double>(beta)});
            (const_cast<NDArray&>(x)).applyTransform(transform::ScaledTanh, z, &args);
            break;
        }
        case 7:
            (const_cast<NDArray&>(x)).applyTransform(transform::HardSigmoid, z);
            break;
        case 8:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::ELU, alpha, z);
            break;
        case 9:
            (const_cast<NDArray&>(x)).applyTransform(transform::SoftSign, z);
            break;
        case 10:
            (const_cast<NDArray&>(x)).applyTransform(transform::SoftPlus, z);
            break;
        default:
            throw std::invalid_argument("LSTM_LAYER operation: wrong id number of activation !");
    }
}

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray tensorAlongTimeBatchDims(const NDArray& arr, const int dataFormat, const int t1, const int t2, const int b1, const int b2) {

    if(dataFormat == 0 || dataFormat == 3)
        return arr({t1,t2, b1,b2, 0,0});    // TNS: [sL, bS, nIn]

    if(dataFormat == 1)
        return arr({b1,b2, t1,t2, 0,0});    // NTS: [bS, sL ,nIn]

    return arr({b1,b2, 0,0, t1,t2});        // NST: [bS, nIn, sL]
}

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE int getBatchTimeTotalIndex(const int dataFormat, const int sL, const int bS, const int t, const int b) {

    if(dataFormat == 0 || dataFormat == 3)
        return t * bS + b;   // TNS: shape [sL, bS, nIn]

    return b * sL + t;       // NTS, NST: shape [bS, sL, nIn], [bS, nIn, sL]
}


}
}
}


#endif //LIBND4J_LSTMLAYER_H
