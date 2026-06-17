/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/segment_gemm.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void segmentGemmCpu_(LaunchContext* context,
                             NDArray* input, NDArray* weights,
                             NDArray* segmentOffsets, NDArray* segmentSizes,
                             NDArray* output) {
    auto numExperts = weights->sizeAt(0);
    auto inDim = weights->sizeAt(1);
    auto outDim = weights->sizeAt(2);

    // For each expert, perform matmul on its token segment
    PRAGMA_OMP_PARALLEL_FOR
    for (LongType e = 0; e < numExperts; e++) {
        LongType offset = segmentOffsets->e<LongType>(e);
        LongType size = segmentSizes->e<LongType>(e);

        if (size <= 0) continue;

        // output[offset:offset+size, :] = input[offset:offset+size, :] @ weights[e, :, :]
        for (LongType t = 0; t < size; t++) {
            PRAGMA_OMP_PARALLEL_FOR
            for (LongType j = 0; j < outDim; j++) {
                T sum = static_cast<T>(0);
                for (LongType k = 0; k < inDim; k++) {
                    sum += input->e<T>(offset + t, k) * weights->e<T>(e, k, j);
                }
                output->p<T>(offset + t, j, sum);
            }
        }
    }
}

void segmentGemmCpu(LaunchContext* context,
                     NDArray* input, NDArray* weights,
                     NDArray* segmentOffsets, NDArray* segmentSizes,
                     NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), segmentGemmCpu_,
                          (context, input, weights, segmentOffsets, segmentSizes, output),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
