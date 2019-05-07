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
    void _execute(graph::LaunchContext* context, void *vptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, void *vptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides) {
        auto ptrSpace = reinterpret_cast<T *>(vptrSpace);
        auto ptrBatch = reinterpret_cast<T *>(vptrBatch);
        SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(ptrSpace, space_shape, space_strides, block_shape, pad_start, block_offsets, ptrBatch, batch_shape, batch_strides);
    };

    Nd4jStatus _spaceToBatch(graph::LaunchContext* context, int internal_block_dims, NDArray *input, NDArray *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *paddings) {
        auto in = input->reshape('c', internal_input_shape);
        auto out = output->reshape('c', internal_output_shape);
        switch (internal_block_dims) {
            case 1:
                _prepare<1, false>(context, in, out, block_shape, paddings);
                break;
            case 2:
                _prepare<2, false>(context, in, out, block_shape, paddings);
                break;
            case 3:
                _prepare<3, false>(context, in, out, block_shape, paddings);
                break;
            case 4:
                _prepare<4, false>(context, in, out, block_shape, paddings);
                break;
            default: {
                return Status::THROW("SpaceToBatch: Wrong number of internal_block_dims");
            }
        }

        delete in;
        delete out;

        return Status::OK();
    }

    Nd4jStatus _batchToSpace(graph::LaunchContext* context, int internal_block_dims, NDArray *input, NDArray *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *crops) {
        auto in = input->reshape('c', internal_input_shape);
        auto out = output->reshape('c', internal_output_shape);
        switch (internal_block_dims) {
            case 1:
                _prepare<1, true>(context, in, out, block_shape, crops);
                break;
            case 2:
                _prepare<2, true>(context, in, out, block_shape, crops);
                break;
            case 3:
                _prepare<3, true>(context, in, out, block_shape, crops);
                break;
            case 4:
                _prepare<4, true>(context, in, out, block_shape, crops);
                break;
            default: {
                return Status::THROW("BatchToSpace: Wrong number of internal_block_dims");
            }
        }

        delete in;
        delete out;

        return Status::OK();
    }

#define STB_DIM (0, 1),\
                (1, 2),\
                (2, 3),\
                (3, 4)

#define STB_BOOL (0, false),\
                 (1, true)

    BUILD_TRIPLE_TEMPLATE(template void _execute, (graph::LaunchContext* context, void *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, void *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides), LIBND4J_TYPES, STB_DIM, STB_BOOL);

#undef STB_BOOL
#undef STB_DIM
}
}
}