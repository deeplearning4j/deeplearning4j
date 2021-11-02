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
// Created by raver119 on 19.01.18.
//

#ifndef LIBND4J_S_T_B_H
#define LIBND4J_S_T_B_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void batchToSpace(sd::LaunchContext* context, const NDArray& input, NDArray& output,
                                const sd::Unsigned cropBottom, const sd::Unsigned cropTop, const sd::Unsigned cropLeft,
                                const sd::Unsigned cropRight, const sd::Unsigned blockSize);

SD_LIB_HIDDEN void spaceToBatch(sd::LaunchContext* context, const NDArray& input, NDArray& output,
                                const sd::Unsigned padBottom, const sd::Unsigned padTop, const sd::Unsigned padLeft,
                                const sd::Unsigned padRight, const sd::Unsigned blockSize);

SD_LIB_HIDDEN void spaceToBatchND(sd::LaunchContext* context, const NDArray& input, const NDArray& blockShape,
                                  const NDArray& padding, NDArray& output);

SD_LIB_HIDDEN void batchToSpaceND(sd::LaunchContext* context, const NDArray& input, const NDArray& blockShape,
                                  const NDArray& crop, NDArray& output);

/*
    // this method MUST be platform-specific

    template <typename T, int NUM_BLOCK_DIMS, bool B2S>
    void _execute(sd::LaunchContext * context, void *ptrSpace, const sd::LongType *space_shape, const sd::LongType
*space_strides, const sd::LongType *block_shape, const sd::LongType *pad_start, const sd::LongType *block_offsets, void
*ptrBatch, const sd::LongType *batch_shape, const sd::LongType *batch_strides);


    template <int NUM_BLOCK_DIMS, bool B2S>
    SD_INLINE void _prepare(sd::LaunchContext * context, NDArray * space, NDArray *batch, const sd::LongType
block_array[NUM_BLOCK_DIMS], const sd::LongType padding_array[NUM_BLOCK_DIMS * 2]) {

        sd::LongType pad_start[NUM_BLOCK_DIMS];
        sd::LongType block_shape[NUM_BLOCK_DIMS];
        sd::LongType space_shape[NUM_BLOCK_DIMS];
        sd::LongType batch_shape[NUM_BLOCK_DIMS];

        const int batchSize = batch->sizeAt(0);
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
        for (int batch_b = 0; batch_b < batchSize; ++batch_b) {
            const sd::LongType space_b = batch_b % space_size;
            sd::LongType block_index = batch_b / space_size;
            sd::LongType block_offsets[NUM_BLOCK_DIMS];
            for (sd::LongType block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
                block_offsets[block_dim] = block_dim > 0 ? block_index % block_shape[block_dim] : block_index;
                block_index /= block_shape[block_dim];
            }

            sd::LongType space_offset = space_b * space_strides[0];
            sd::LongType batch_offset = batch_b * batch_strides[0];

            auto xType = space->dataType();
            //_execute<T, NUM_BLOCK_DIMS, B2S>(space->buffer() + space_offset, space_shape, &space_strides[1],
block_shape, pad_start, block_offsets, batch->buffer() + batch_offset, batch_shape, &batch_strides[1]);
            BUILD_SINGLE_PARTIAL_SELECTOR(xType, _execute<, (NUM_BLOCK_DIMS, B2S>(context,
space->bufferWithOffset(space_offset), space_shape, &space_strides[1], block_shape, pad_start, block_offsets,
batch->bufferWithOffset(batch_offset), batch_shape, &batch_strides[1])), SD_COMMON_TYPES);
        }
    };

    sd::Status _spaceToBatch(sd::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output,
std::vector<sd::LongType> &internal_input_shape, std::vector<sd::LongType> &internal_output_shape, sd::LongType
*block_shape, sd::LongType *paddings);

    sd::Status _batchToSpace(sd::LaunchContext * context, int internal_block_dims, NDArray *input, NDArray *output,
std::vector<sd::LongType> &internal_input_shape, std::vector<sd::LongType> &internal_output_shape, sd::LongType
*block_shape, sd::LongType *crops);
    */
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_S_T_B_H
