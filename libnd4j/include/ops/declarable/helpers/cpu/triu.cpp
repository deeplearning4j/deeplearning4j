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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include <ops/declarable/helpers/transforms.h>
#include <helpers/Loops.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void triuBP_(sd::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {

    auto dOdI = NDArray(&gradO);                // dO/dI
    const_cast<NDArray&>(input).fillAsTriangular<T>(0, diagonal, dOdI.sizeAt(-1), dOdI, 'b');
    int dLen = dOdI.lengthOf();

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            if (dOdI.t<T>(i) != static_cast<T>(0.f))
                dOdI.t<T>(i) = static_cast<T>(1.f);
        }
    };
    samediff::Threads::parallel_for(func, 0, dLen);

    // FIXME: !!!
    gradI.assign(dOdI * gradO);                          // chain rule: dLoss/dI = dO/dI * dLoss/dO
}

    void triuBP(sd::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const int diagonal) {
        BUILD_SINGLE_SELECTOR(gradO.dataType(), triuBP_, (context, input, gradO, gradI, diagonal), LIBND4J_TYPES);
    }

}
}
}
