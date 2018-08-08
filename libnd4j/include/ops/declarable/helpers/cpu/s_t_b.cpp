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
// Created by raver119 on 19.01.18.
//

#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {
namespace helpers {


    template <int N, bool B2S>
    struct SpaceToBatchHelper {
        template <typename T>
        static void run(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
            for (int batch_pos = 0; batch_pos < batch_shape[0]; ++batch_pos) {
                const int space_pos = batch_pos * block_shape[0] + block_offsets[0] - pad_start[0];
                if (space_pos >= 0 && space_pos < space_shape[0]) {
                    SpaceToBatchHelper<N - 1, B2S>::run(ptrSpace + space_pos * space_strides[0], space_shape + 1, space_strides + 1, block_shape + 1, pad_start + 1, block_offsets + 1, ptrBatch, batch_shape + 1, batch_strides + 1);
                } else {
                    if (!B2S)
                        for (int i = 0; i < batch_strides[0]; i++)
                            ptrBatch[i] = (T) 0.f;
                }

                ptrBatch += batch_strides[0];
            }
        }
    };

    template <bool B2S>
    struct SpaceToBatchHelper<0, B2S> {
        template <typename T>
        static void run(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
            int str = batch_strides[-1];
            for (int i = 0; i < str; i++)
                if (B2S)
                    ptrSpace[i] = ptrBatch[i];
                else
                    ptrBatch[i] = ptrSpace[i];
        }
    };

    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
        SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(ptrSpace, space_shape, space_strides, block_shape, pad_start, block_offsets, ptrBatch, batch_shape, batch_strides);
    };


    template void _execute<float, 4, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float, 3, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float, 2, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float, 1, false>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<float16, 4, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float16, 3, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float16, 2, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float16, 1, false>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<double, 4, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<double, 3, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<double, 2, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<double, 1, false>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<int, 4, false>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<int, 3, false>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<int, 2, false>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<int, 1, false>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<Nd4jLong, 4, false>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<Nd4jLong, 3, false>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<Nd4jLong, 2, false>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<Nd4jLong, 1, false>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);


    template void _execute<float, 4, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float, 3, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float, 2, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float, 1, true>(float *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<float16, 4, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float16, 3, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float16, 2, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<float16, 1, true>(float16 *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, float16 *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<double, 4, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<double, 3, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<double, 2, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<double, 1, true>(double *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, double *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<int, 4, true>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<int, 3, true>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<int, 2, true>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<int, 1, true>(int *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, int *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

    template void _execute<Nd4jLong, 4, true>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<Nd4jLong, 3, true>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<Nd4jLong, 2, true>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);
    template void _execute<Nd4jLong, 1, true>(Nd4jLong *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, Nd4jLong *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);

}
}
}