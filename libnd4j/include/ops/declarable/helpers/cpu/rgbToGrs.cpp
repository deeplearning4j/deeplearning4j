/*******************************************************************************
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
 
#include <ops/declarable/helpers/rgbToGrs_conf_op.h>
#include <execution/Threads.h>

namespace nd4j {
namespace ops {
namespace helpers {
template <typename T1, typename T2>
    FORCEINLINE static void rgb_to_grs(const NDArray* input, NDArray* output, const int dimLast) {
        const T1* x = input->bufferAsT<T>();
        T2* z = output->bufferAsT<T>();
// TODO for c-order add check point for it
        auto funcGood = PRAGMA_THREADS_FOR{
            for (auto i = start; i < stop; i += increment) {
                z[i] = 0.2989f*x[i*3] + 0.5870f*x[(i*3) + 1] + 0.1140f*x[(i*3) + 2];
            }
        };
// different orders getIndexOffset
        samediff::Threads::parallel_for(funcGood, 0, input->lengthOf(), 1);
    }
    BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), rgb_to_grs, (input, output, dimLast), FLOAT_TYPES);
}
}
}
