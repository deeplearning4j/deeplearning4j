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
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
//  @author Yurii Shyrma, created on 05.12.2017
//

#include<ops/declarable/helpers/sru.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

    //////////////////////////////////////////////////////////////////////////
    static FORCEINLINE NDArray activation(const NDArray& arr) {
        // return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();
        auto result = NDArray(&arr, false, arr.getContext());
        (const_cast<NDArray&>(arr)).applyTransform(transform::Tanh, &result);
        return result;
    }


    //////////////////////////////////////////////////////////////////////////
    static FORCEINLINE NDArray sigmoid(const NDArray& arr) {
        return (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
    }


//////////////////////////////////////////////////////////////////////////
void sruCell(nd4j::LaunchContext * context, const NDArray* x, const NDArray* c0, const NDArray* w, const NDArray* b, NDArray* h, NDArray* c) {

    // x   input [bS x inSize], bS - batch size, inSize - number of features
    // c0  previous cell state c  [bS x inSize], that is at previous time step t-1
    // w   weights [inSize x 3*inSize]
    // b   biases [2*inSize]

    // h   current cell output [bS x inSize], that is at current time step t
    // c   current cell state  [bS x inSize], that is at current time step t

    const int inSize = x->sizeAt(1);           // inSize - number of features

    auto z = mmul(*x, *w);               //  [bS x 3*inSize]

    // forget gate = sigmoid(x*Wf + bf)
    auto f = sigmoid(z({0,0, inSize,   2*inSize}) + (*b)({0, inSize}));

    // reset gate = sigmoid(x*Wr + br)
    auto r = sigmoid(z({0,0, 2*inSize, 3*inSize}) + (*b)({inSize, 2*inSize}));

    // ◦ means element-wise product or so called Hadamard product
    // current sell state = f◦c0 + (1 - f)◦(x*Wc)
    c->assign(f * (*c0) + (1.f - f) * z({0, 0 ,0, inSize}) );
    // *c = f*(*c0 - z({},{0, inSize})) + z({{},{0, inSize}});

    // current cell output = r◦activation(c) + (1 - r)◦x
    h->assign( r * activation(*c) + (1.f - r) * (*x) );
    // *h = r * (activation<T>(c) - *x) + *x;
}

//////////////////////////////////////////////////////////////////////////
void sruTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* c0, const NDArray* w, const NDArray* b, NDArray* h, NDArray* c) {

    // x   input [bS x inSize x time]
    // c0  initial cell state  (at time step = 0) [bS x inSize],
    // w   weights, [3*inSize x inSize]
    // b   biases,  [2*inSize]

    // h   cell outputs [bS x inSize x time]
    // c   cell states  [bS x inSize x time]

    auto wT = w->transpose();                             // [3*inSize x inSize] -> [inSize x 3*inSize]

    const int time  = x->sizeAt(2);

    NDArray ct_1(*c0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        auto xt = (*x)({0,0, 0,0, t,t+1});
        auto ht = (*h)({0,0, 0,0, t,t+1});
        auto ct = (*c)({0,0, 0,0, t,t+1});

        helpers::sruCell(context, &xt, &ct_1, &wT, b,  &ht, &ct);
        ct_1.assign(ct);
    }
}


    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void sruBI_(NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* mask, NDArray* ht, NDArray* ct) {

    }

    //////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void sruBIBP_(NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* ct, const NDArray* inGradC0, const NDArray* inGradHt, const NDArray* mask,
                     NDArray* gradI, NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
    }


    void sruBI(nd4j::LaunchContext * context, NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* mask, NDArray* ht, NDArray* ct) {
        BUILD_SINGLE_SELECTOR(x->dataType(), sruBI_, (x, w, b, c0, mask, ht, ct), FLOAT_TYPES);
    }

    void sruBIBP(nd4j::LaunchContext * context, NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* ct, const NDArray* inGradC0, const NDArray* inGradH, const NDArray* mask, NDArray* gradI, NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
        BUILD_SINGLE_SELECTOR(x->dataType(), sruBIBP_, (x, w, b, c0, ct, inGradC0, inGradH, mask, gradI, gradW, gradB, gradC0), FLOAT_TYPES);
    }


    BUILD_SINGLE_TEMPLATE(template void sruBI_,   (NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* mask, NDArray* ht, NDArray* ct), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template void sruBIBP_, (NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* ct, const NDArray* inGradC0, const NDArray* inGradH, const NDArray* mask, NDArray* gradI, NDArray* gradW, NDArray* gradB, NDArray* gradC0), FLOAT_TYPES);

}
}
}