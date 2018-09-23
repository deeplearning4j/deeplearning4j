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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//
//  Created by raver119 on 19.01.18.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_batch_to_space)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {
    const int kMaxSpaceToBatchBlockDims = 4;

    CUSTOM_OP_IMPL(batch_to_space, 1, 1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);

        bool order_changed = false;
        if (input->ordering() != 'c' || input->ews() != 1 || input->isView()) {
            order_changed = true;
            input = input->dup('c');
        }

        auto output = OUTPUT_VARIABLE(0);

        const int input_dims = input->rankOf();
        int block_dims = 0;
        std::vector<Nd4jLong> block_shape; // = blocks->template asVectorT<int>();
        std::vector<Nd4jLong> crops_shape; // = crops->template asVectorT<int>();

        if (block.width() >= 3) {
            auto blocks = INPUT_VARIABLE(1);
            auto crops = INPUT_VARIABLE(2);

            block_dims = (int) blocks->lengthOf();

            REQUIRE_TRUE(blocks->isVector(), 0, "BatchToSpace: blocks supposed to be vector, but got %iD instead", blocks->rankOf());
            REQUIRE_TRUE(input->rankOf() >= 1 + blocks->lengthOf() + 1, 0, "BatchToSpace: blocks length + 2 should match input rank at least");
            REQUIRE_TRUE(crops->rankOf() == 2, 0, "BatchToSpace: padding should have rank of 2, but got %i instead", crops->rankOf());
            REQUIRE_TRUE(crops->columns() == 2 && blocks->lengthOf() == crops->rows(), 0, "BatchToSpace: padding should have M rows and 2 columns");

            block_shape = blocks->template asVectorT<Nd4jLong>();
            crops_shape = crops->template asVectorT<Nd4jLong>();

        } else if (block.numI() > 0) {
            int totalArgs = block.numI();

            int M = totalArgs / 3;
            REQUIRE_TRUE(totalArgs % 3 == 0, 0, "BatchToSpace: number of IntArguments should be dividable by 3 without reminder");

            block_dims = M;
            block_shape.resize(block_dims);
            crops_shape.resize(M*2);

            REQUIRE_TRUE(input->rankOf() >= 1 + M + 1, 0, "BatchToSpace: blocks length + 2 should match input rank at least");

            int e = 0;
            for (; e < block_dims; e++)
                block_shape[e] = INT_ARG(e);

            for (; e < block.numI(); e++)
                crops_shape[e - M] = INT_ARG(e);
        } else {
            REQUIRE_TRUE(false, 0, "BatchToSpace: there should be some params :(");
        }



        int removed_prefix_block_dims = 0;
        for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
            const int dim = removed_prefix_block_dims;
            if (crops_shape[2 * dim] != 0 || crops_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int removed_suffix_block_dims = 0;
        for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims; ++removed_suffix_block_dims) {
            const int dim = block_dims - 1 - removed_suffix_block_dims;
            if (crops_shape[2 * dim] != 0 || crops_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int block_shape_product = 1;
        for (int block_dim = 0; block_dim < block_dims; ++block_dim)
            block_shape_product *= block_shape[block_dim];

        REQUIRE_TRUE(block_shape_product > 0, 0, "BatchToSpace: block should contain values >= 1 ONLY");


        const Nd4jLong orig_input_batch_size = input->sizeAt(0);
        const int internal_block_dims = block_dims - removed_prefix_block_dims - removed_suffix_block_dims;

        REQUIRE_TRUE(internal_block_dims <= kMaxSpaceToBatchBlockDims, 0, "BatchToSpace: Maximum number of non-combined block dimensions should be less or equal then %i but got %i instead", kMaxSpaceToBatchBlockDims, internal_block_dims);

        if (internal_block_dims == 0) {
            output->assign(input);
            return Status::OK();
        }

        std::vector<Nd4jLong> internal_input_shape;
        std::vector<Nd4jLong> internal_output_shape;
        std::vector<Nd4jLong> external_output_shape;

        external_output_shape.emplace_back(orig_input_batch_size / block_shape_product);

        int input_batch_size = orig_input_batch_size;
        for (int block_dim = 0; block_dim < removed_prefix_block_dims; ++block_dim) {
            const int size = input->sizeAt(block_dim + 1);
            input_batch_size *= size;
            external_output_shape.emplace_back(size);
        }
        internal_input_shape.emplace_back(input_batch_size);
        internal_output_shape.emplace_back(input_batch_size / block_shape_product);

        for (int block_dim = removed_prefix_block_dims;
             block_dim < block_dims - removed_suffix_block_dims; ++block_dim) {
            const int crop_start = crops_shape[2 * block_dim];
            const int crop_end = crops_shape[2 * block_dim + 1];

            const int input_size = input->sizeAt(block_dim + 1);
            const int block_shape_value = block_shape[block_dim];
            const int cropped_size = input_size * block_shape_value - crop_start - crop_end;

            REQUIRE_TRUE(cropped_size >= 0, 0, "BatchToSpace: cropped_size should have non-negative value");

            internal_input_shape.emplace_back(input_size);
            internal_output_shape.emplace_back(cropped_size);
            external_output_shape.emplace_back(cropped_size);
        }

        int depth = 1;
        for (int dim = block_dims - removed_suffix_block_dims + 1; dim < input_dims; ++dim) {
            const int size = input->sizeAt(dim);
            external_output_shape.emplace_back(size);
            depth *= size;
        }

        internal_input_shape.emplace_back(depth);
        internal_output_shape.emplace_back(depth);

        auto internal_crops = &crops_shape.data()[2 * removed_prefix_block_dims];
        auto internal_block_shape = &block_shape.data()[removed_prefix_block_dims];

        helpers::_batchToSpace(internal_block_dims, output, input, internal_output_shape, internal_input_shape, internal_block_shape, internal_crops);

        if (order_changed)
            delete input;

        return Status::OK();
    }

    DECLARE_SHAPE_FN(batch_to_space) {
        auto in = inputShape->at(0);

        const int input_dims = shape::rank(in);
        int block_dims = 0;

        std::vector<int> block_shape;
        std::vector<int> crops_shape;

        if (block.width() >= 3) {
            auto blocks = INPUT_VARIABLE(1);
            auto crops = INPUT_VARIABLE(2);

            block_dims = (int) blocks->lengthOf();

            block_shape = blocks->template asVectorT<int>();
            crops_shape = crops->template asVectorT<int>();

            //shape::printShapeInfoLinear("STB input shape: ",in);
            //blocks->printBuffer("STB blocks");
            //crops->printBuffer("STB crops");

        } else if (block.numI() > 0) {
            int totalArgs = block.numI();

            int M = totalArgs / 3;

            block_dims = M;
            block_shape.resize(block_dims);
            crops_shape.resize(M*2);

            int e = 0;
            for (; e < block_dims; e++)
                block_shape[e] = INT_ARG(e);

            for (; e < block.numI(); e++)
                crops_shape[e - M] = INT_ARG(e);
        } else {
            // throw something here
        }

        int removed_prefix_block_dims = 0;
        for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
            const int dim = removed_prefix_block_dims;
            if (crops_shape[2 * dim] != 0 || crops_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int removed_suffix_block_dims = 0;
        for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims; ++removed_suffix_block_dims) {
            const int dim = block_dims - 1 - removed_suffix_block_dims;
            if (crops_shape[2 * dim] != 0 || crops_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int block_shape_product = 1;
        for (int block_dim = 0; block_dim < block_dims; ++block_dim)
            block_shape_product *= block_shape[block_dim];


        const int orig_input_batch_size = shape::sizeAt(in, 0);
        const int internal_block_dims = block_dims - removed_prefix_block_dims - removed_suffix_block_dims;

        if (internal_block_dims == 0) {
            // just return input shape here
            Nd4jLong *newShape;
            COPY_SHAPE(in, newShape);
            return SHAPELIST(newShape);
        }

        // go full route otherwise
        std::vector<Nd4jLong> internal_input_shape;
        std::vector<Nd4jLong> internal_output_shape;
        std::vector<Nd4jLong> external_output_shape;

        external_output_shape.emplace_back(orig_input_batch_size / block_shape_product);

        auto input_batch_size = orig_input_batch_size;
        for (int block_dim = 0; block_dim < removed_prefix_block_dims; ++block_dim) {
            const Nd4jLong size = shape::sizeAt(in, block_dim + 1);
            input_batch_size *= size;
            external_output_shape.emplace_back(size);
        }
        internal_input_shape.emplace_back(input_batch_size);
        internal_output_shape.emplace_back(input_batch_size / block_shape_product);

        for (int block_dim = removed_prefix_block_dims;
             block_dim < block_dims - removed_suffix_block_dims; ++block_dim) {
            const int crop_start = crops_shape[2 * block_dim];
            const int crop_end = crops_shape[2 * block_dim + 1];

            const int input_size = shape::sizeAt(in, block_dim + 1);
            const int block_shape_value = block_shape[block_dim];
            const int cropped_size = input_size * block_shape_value - crop_start - crop_end;

            internal_input_shape.emplace_back(input_size);
            internal_output_shape.emplace_back(cropped_size);
            external_output_shape.emplace_back(cropped_size);
        }

        int depth = 1;
        for (int dim = block_dims - removed_suffix_block_dims + 1; dim < input_dims; ++dim) {
            const int size = shape::sizeAt(in, dim);
            external_output_shape.emplace_back(size);
            depth *= size;
        }

        Nd4jLong *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) external_output_shape.size()), Nd4jLong);

        //nd4j_printv("STB shape: ", external_output_shape);

        // we always give out C order here
        shape::shapeBuffer((int) external_output_shape.size(), block.dataType(), external_output_shape.data(), newShape);

        return SHAPELIST(newShape);
    }
}
}

#endif