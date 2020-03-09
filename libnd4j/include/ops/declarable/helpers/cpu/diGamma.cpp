/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

#include<ops/declarable/helpers/gammaMathFunc.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// calculate digamma function for array elements
template <typename T>
static void diGamma_(const NDArray& x, NDArray& z) {

	auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++)
            z.p(i, diGammaScalar<T>(x.e<T>(i)));
    };
	sd::Threads::parallel_for(func, 0, x.lengthOf());
}

void diGamma(sd::LaunchContext* context, const NDArray& x, NDArray& z) {

	BUILD_SINGLE_SELECTOR(x.dataType(), diGamma_, (x, z), FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void diGamma_, (const NDArray& x, NDArray& z), FLOAT_TYPES);



}
}
}

