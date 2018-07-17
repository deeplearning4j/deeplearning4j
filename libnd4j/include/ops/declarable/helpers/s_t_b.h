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

#ifndef LIBND4J_S_T_B_H
#define LIBND4J_S_T_B_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {
    // this method MUST be platform-specific

    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(T *ptrSpace, const Nd4jLong *space_shape, const Nd4jLong *space_strides, const Nd4jLong *block_shape, const Nd4jLong *pad_start, const Nd4jLong *block_offsets, T *ptrBatch, const Nd4jLong *batch_shape, const Nd4jLong *batch_strides);


    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    FORCEINLINE void _prepare(NDArray<T> * space, NDArray<T> *batch, const Nd4jLong block_array[NUM_BLOCK_DIMS], const Nd4jLong padding_array[NUM_BLOCK_DIMS * 2]) {

        Nd4jLong pad_start[NUM_BLOCK_DIMS];
        Nd4jLong block_shape[NUM_BLOCK_DIMS];
        Nd4jLong space_shape[NUM_BLOCK_DIMS];
        Nd4jLong batch_shape[NUM_BLOCK_DIMS];

        const int batch_size = batch->sizeAt(0);
        const int space_size = space->sizeAt(0);

#pragma unroll
        for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; block_dim++) {
            pad_start[block_dim] = padding_array[block_dim * 2];
            block_shape[block_dim] = block_array[block_dim];
            space_shape[block_dim] = space->sizeAt(block_dim + 1);
            batch_shape[block_dim] = batch->sizeAt(block_dim + 1);
        }

        auto space_strides = space->stridesOf();
        auto batch_strides = batch->stridesOf();

        // TODO: this loop should be moved to _execute phase
        for (int batch_b = 0; batch_b < batch_size; ++batch_b) {
            const Nd4jLong space_b = batch_b % space_size;
            Nd4jLong block_index = batch_b / space_size;
            Nd4jLong block_offsets[NUM_BLOCK_DIMS];
            for (Nd4jLong block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
                block_offsets[block_dim] = block_dim > 0 ? block_index % block_shape[block_dim] : block_index;
                block_index /= block_shape[block_dim];
            }

            Nd4jLong space_offset = space_b * space_strides[0];
            Nd4jLong batch_offset = batch_b * batch_strides[0];

            _execute<T, NUM_BLOCK_DIMS, B2S>(space->buffer() + space_offset, space_shape, &space_strides[1], block_shape, pad_start, block_offsets, batch->buffer() + batch_offset, batch_shape, &batch_strides[1]);
        }
    };


    template <typename T>
    FORCEINLINE Nd4jStatus _spaceToBatch(int internal_block_dims, NDArray<T> *input, NDArray<T> *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *paddings) {
        auto in = input->reshape('c', internal_input_shape);
        auto out = output->reshape('c', internal_output_shape);
        switch (internal_block_dims) {
            case 1:
                _prepare<T, 1, false>(in, out, block_shape, paddings);
                break;
            case 2:
                _prepare<T, 2, false>(in, out, block_shape, paddings);
                break;
            case 3:
                _prepare<T, 3, false>(in, out, block_shape, paddings);
                break;
            case 4:
                _prepare<T, 4, false>(in, out, block_shape, paddings);
                break;
            default: {
                return Status::THROW("SpaceToBatch: Wrong number of internal_block_dims");
            }
        }

        delete in;
        delete out;

        return Status::OK();
    }


    template <typename T>
    FORCEINLINE Nd4jStatus _batchToSpace(int internal_block_dims, NDArray<T> *input, NDArray<T> *output, std::vector<Nd4jLong> &internal_input_shape, std::vector<Nd4jLong> &internal_output_shape, Nd4jLong *block_shape, Nd4jLong *crops) {
        auto in = input->reshape('c', internal_input_shape);
        auto out = output->reshape('c', internal_output_shape);
        switch (internal_block_dims) {
            case 1:
                _prepare<T, 1, true>(in, out, block_shape, crops);
                break;
            case 2:
                _prepare<T, 2, true>(in, out, block_shape, crops);
                break;
            case 3:
                _prepare<T, 3, true>(in, out, block_shape, crops);
                break;
            case 4:
                _prepare<T, 4, true>(in, out, block_shape, crops);
                break;
            default: {
                return Status::THROW("BatchToSpace: Wrong number of internal_block_dims");
            }
        }

        delete in;
        delete out;

        return Status::OK();
    }
}
}
}

#endif //LIBND4J_S_T_B_H
