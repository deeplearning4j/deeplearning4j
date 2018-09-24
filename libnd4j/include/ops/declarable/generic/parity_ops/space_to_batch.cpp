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
#if NOT_EXCLUDED(OP_space_to_batch)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {
    const int kMaxSpaceToBatchBlockDims = 4;

    CUSTOM_OP_IMPL(space_to_batch, 1, 1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);

        std::vector<Nd4jLong> block_shape;
        std::vector<Nd4jLong> padding_shape;

        bool order_changed = false;
        if (input->ordering() != 'c') {
            order_changed = true;
            input = input->dup('c');
        }

        auto output = OUTPUT_VARIABLE(0);

        const int xRank = input->rankOf();
        int block_dims = 0;



        if (block.width() >= 3) {
            auto blocks = INPUT_VARIABLE(1);
            auto padding = INPUT_VARIABLE(2);

            block_dims = (int) blocks->lengthOf();

            REQUIRE_TRUE(blocks->isVector() || blocks->lengthOf() == 1, 0, "SpaceToBatch: blocks supposed to be vector or scalar, but got %iD instead", blocks->rankOf());
            REQUIRE_TRUE(input->rankOf() >= 1 + blocks->lengthOf() + 1, 0, "SpaceToBatch: blocks length + 2 should match input rank at least");
            REQUIRE_TRUE(padding->rankOf() == 2, 0, "SpaceToBatch: padding should have rank of 2, but got %i instead", padding->rankOf());
            REQUIRE_TRUE(padding->columns() == 2 && blocks->lengthOf() == padding->rows(), 0, "SpaceToBatch: padding should have M rows and 2 columns");

            block_shape = blocks->template asVectorT<Nd4jLong>();
            padding_shape = padding->template asVectorT<Nd4jLong>();

        } else if (block.numI() > 0) {
            int totalArgs = block.numI();

            int M = totalArgs / 3;
            REQUIRE_TRUE(totalArgs % 3 == 0, 0, "SpaceToBatch: number of IntArguments should be dividable by 3 without reminder");

            block_dims = M;
            block_shape.resize(block_dims);
            padding_shape.resize(M*2);

            REQUIRE_TRUE(input->rankOf() >= 1 + M + 1, 0, "SpaceToBatch: blocks length + 2 should match input rank at least");

            int e = 0;
            for (; e < block_dims; e++)
                block_shape[e] = INT_ARG(e);

            for (; e < block.numI(); e++)
                padding_shape[e - M] = INT_ARG(e);

        } else {
            REQUIRE_TRUE(false, 0, "SpaceToBatch: there should be some params :(");
        }


        // Determine the length of the prefix of block dims that can be combined
        // into the batch dimension due to having no padding and block_shape=1.
        int removed_prefix_block_dims = 0;
        for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
            const int dim = removed_prefix_block_dims;
            if (padding_shape[2 * dim] != 0 || padding_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;            
        }

        // Determine the length of the suffix of block dims that can be combined
        // into the depth dimension due to having no padding and block_shape=1.
        int removed_suffix_block_dims = 0;
        for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims; ++removed_suffix_block_dims) {
            const int dim = block_dims - 1 - removed_suffix_block_dims;
            if (padding_shape[dim * 2] != 0 || padding_shape[dim * 2 + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int block_shape_product = 1;
        for (int block_dim = 0; block_dim < block_dims; ++block_dim)
            block_shape_product *= block_shape[block_dim];

        REQUIRE_TRUE(block_shape_product > 0, 0, "SpaceToBatch: block should contain values >= 1 ONLY");

        const int internal_block_dims = block_dims - removed_prefix_block_dims - removed_suffix_block_dims;

        REQUIRE_TRUE(internal_block_dims <= kMaxSpaceToBatchBlockDims, 0, "SpaceToBatch: Maximum number of non-combined block dimensions should be less or equal then %i but got %i instead", kMaxSpaceToBatchBlockDims, internal_block_dims);

        if (internal_block_dims == 0) {
            // we return array if there's nothing to move here
            output->assign(input);
            return Status::OK();
        }

        std::vector<Nd4jLong> internal_input_shape;
        std::vector<Nd4jLong> internal_output_shape;
        std::vector<Nd4jLong> external_output_shape;

        external_output_shape.emplace_back(input->sizeAt(0) * block_shape_product);
        int input_batch_size = input->sizeAt(0);
        for (int block_dim = 0; block_dim < removed_prefix_block_dims; block_dim++) {
            const int size = input->sizeAt(block_dim + 1);
            input_batch_size *= size;
            external_output_shape.emplace_back(size);
        }
        internal_input_shape.emplace_back(input_batch_size);
        internal_output_shape.emplace_back(input_batch_size * block_shape_product);

        for (int block_dim = removed_prefix_block_dims; block_dim < block_dims - removed_suffix_block_dims; block_dim++) {
            const int pad_start = padding_shape[2 * block_dim];
            const int pad_end = padding_shape[2 * block_dim + 1];

            const int input_size = input->sizeAt(block_dim + 1);
            const int block_shape_value = block_shape[block_dim];
            const int padded_size = input_size + pad_start + pad_end;
            const int output_size = padded_size / block_shape_value;

            // FIXME: validation required here

            internal_input_shape.emplace_back(input_size);
            internal_output_shape.emplace_back(output_size);
            external_output_shape.emplace_back(output_size);
        }

        int depth = 1;
        for (int dim = block_dims - removed_suffix_block_dims + 1; dim < xRank; dim++) {
            const int size = input->sizeAt(dim);
            external_output_shape.emplace_back(size);
            depth *= size;
        }

        internal_input_shape.emplace_back(depth);
        internal_output_shape.emplace_back(depth);

        Nd4jLong* internal_paddings = &padding_shape.data()[2 * removed_prefix_block_dims];
        Nd4jLong* internal_block_shape = &block_shape.data()[removed_prefix_block_dims];

        helpers::_spaceToBatch(internal_block_dims, input, output, internal_input_shape, internal_output_shape, internal_block_shape, internal_paddings);

        if (order_changed)
            delete input;

        return Status::OK();
    }

    DECLARE_SHAPE_FN(space_to_batch) {
        auto in = inputShape->at(0);

        const int xRank = shape::rank(in);
        int block_dims = 0;

        std::vector<int> block_shape;
        std::vector<int> padding_shape;

        if (block.width() >= 3) {
            auto blocks = INPUT_VARIABLE(1);
            auto padding = INPUT_VARIABLE(2);

            block_dims = (int) blocks->lengthOf();

            block_shape.resize(block_dims);
            padding_shape.resize(padding->lengthOf());

            for (int e = 0; e < block_dims; e++)
                block_shape[e] = blocks->e<int>(e);

            for (int e = 0; e < padding->lengthOf(); e++)
                padding_shape[e] = padding->e<int>(e);
        } else if (block.numI() > 0) {
            int totalArgs = block.numI();

            int M = totalArgs / 3;

            block_dims = M;
            block_shape.resize(block_dims);
            padding_shape.resize(M*2);

            int e = 0;
            for (; e < block_dims; e++)
                block_shape[e] = INT_ARG(e);

            for (; e < block.numI(); e++)
                padding_shape[e - M] = INT_ARG(e);

        } else {
            // throw something here
        }


        int removed_prefix_block_dims = 0;
        for (; removed_prefix_block_dims < block_dims; ++removed_prefix_block_dims) {
            const int dim = removed_prefix_block_dims;
            if (padding_shape[2 * dim] != 0 || padding_shape[2 * dim + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int removed_suffix_block_dims = 0;
        for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims; ++removed_suffix_block_dims) {
            const int dim = block_dims - 1 - removed_suffix_block_dims;
            if (padding_shape[dim * 2] != 0 || padding_shape[dim * 2 + 1] != 0 || block_shape[dim] != 1)
                break;
        }

        int block_shape_product = 1;
        for (int block_dim = 0; block_dim < block_dims; ++block_dim)
            block_shape_product *= block_shape[block_dim];

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

        external_output_shape.emplace_back(shape::sizeAt(in, 0) * block_shape_product);
        Nd4jLong input_batch_size = shape::sizeAt(in, 0);
        for (int block_dim = 0; block_dim < removed_prefix_block_dims; block_dim++) {
            const int size = shape::sizeAt(in, block_dim + 1);
            input_batch_size *= size;
            external_output_shape.emplace_back(size);
        }
        internal_input_shape.emplace_back(input_batch_size);
        internal_output_shape.emplace_back(input_batch_size * block_shape_product);

        for (int block_dim = removed_prefix_block_dims; block_dim < block_dims - removed_suffix_block_dims; block_dim++) {
            const Nd4jLong pad_start = padding_shape[2 * block_dim];
            const Nd4jLong pad_end = padding_shape[2 * block_dim + 1];

            const Nd4jLong input_size = shape::sizeAt(in, block_dim + 1);
            const Nd4jLong block_shape_value = block_shape[block_dim];
            const Nd4jLong padded_size = input_size + pad_start + pad_end;
            const Nd4jLong output_size = padded_size / block_shape_value;

            // FIXME: validation required here

            internal_input_shape.emplace_back(input_size);
            internal_output_shape.emplace_back(output_size);
            external_output_shape.emplace_back(output_size);
        }

        int depth = 1;
        for (int dim = block_dims - removed_suffix_block_dims + 1; dim < xRank; dim++) {
            const Nd4jLong size = shape::sizeAt(in, dim);
            external_output_shape.emplace_back(size);
            depth *= size;
        }

        internal_input_shape.emplace_back(depth);
        internal_output_shape.emplace_back(depth);

        Nd4jLong *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) external_output_shape.size()), Nd4jLong);

        // we always give out C order here
        shape::shapeBuffer((int) external_output_shape.size(), block.dataType(), external_output_shape.data(), newShape);

        return SHAPELIST(newShape);
    }
}
}

#endif