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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 27.08.2018
//


#include <ops/declarable/helpers/range.h>

namespace sd {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void global_range(void *output, Nd4jLong length, T start, T delta) {
        auto buff = reinterpret_cast<T*>(output);
        const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for(Nd4jLong i = tid; i < length; i += step)
            buff[i] = start + i * delta;
    }

    //////////////////////////////////////////////////////////////////////////
    // be careful: outVector must have c-order and ews = 1 !!!
    template <typename T>
    static void _range(sd::LaunchContext * context, const NDArray& start, const NDArray& delta, NDArray& outVector) {
        global_range<T><<<512, 512, 2048, *context->getCudaStream()>>>(outVector.specialBuffer(), outVector.lengthOf(), start.e<T>(0), delta.e<T>(0));
    }

    void range(sd::LaunchContext * context, const NDArray& start, const NDArray& delta, NDArray& outVector) {
        NDArray::prepareSpecialUse({&outVector}, {&start, &delta});
        BUILD_SINGLE_SELECTOR(outVector.dataType(), _range, (context, start, delta, outVector), LIBND4J_TYPES);
        NDArray::registerSpecialUse({&outVector}, {&start, &delta});
    }

}
}
}