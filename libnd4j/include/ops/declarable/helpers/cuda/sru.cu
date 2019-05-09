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

    }

    //////////////////////////////////////////////////////////////////////////
    void sruTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* c0, const NDArray* w, const NDArray* b, NDArray* h, NDArray* c) {

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