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
//
//

#include <ops/declarable/helpers/d_t_s.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void __depthToSpace(NDArray *input, NDArray *output, int block_size, bool isNHWC) {
        T *input_ptr = reinterpret_cast<T *>(input->buffer());
        T *output_ptr = reinterpret_cast<T *>(output->buffer());

        const int batch_size = input->sizeAt(0);
        const int input_depth = isNHWC ? input->sizeAt(3) : input->sizeAt(1);
        const int input_height = isNHWC ? input->sizeAt(1) : input->sizeAt(2);
        const int input_width = isNHWC ? input->sizeAt(2) : input->sizeAt(3);

        const int output_depth = isNHWC ? output->sizeAt(3) : output->sizeAt(1);
        const int output_height = isNHWC ? output->sizeAt(1) : output->sizeAt(2);
        const int output_width = isNHWC ? output->sizeAt(2) : output->sizeAt(3);

        const int input_area = input_width * input_height;
        const int input_depth_by_input_area = input_depth * input_area;
        const int output_depth_by_input_height = output_depth * input_height;

        if (isNHWC) {
            const int total_count = batch_size * output_height * output_width * output_depth;
            for (int out_idx = 0; out_idx < total_count; out_idx++) {
                const int d = out_idx % output_depth;
                const int out_idx2 = out_idx / output_depth;
                const int w = out_idx2 % output_width;
                const int out_idx3 = out_idx2 / output_width;
                const int h = out_idx3 % output_height;
                const int b = out_idx3 / output_height;

                const int in_h = h / block_size;
                const int offset_h = h % block_size;
                const int in_w = w / block_size;
                const int offset_w = w % block_size;
                const int offset_d = (offset_h * block_size + offset_w) * output_depth;
                const int in_d = d + offset_d;
                const int inp_idx = in_d + input_depth * (in_w + input_width * (in_h + input_height * b));
                (output_ptr + out_idx)[0] = (input_ptr + inp_idx)[0];
            }
        } else {
            const int total_count = batch_size * input_depth_by_input_area;

            for (int input_idx = 0; input_idx < total_count; input_idx++) {
                const int n_bY_bX_oC_iY = input_idx / input_width;
                const int iX = input_idx - n_bY_bX_oC_iY * input_width;

                const int n_bY_bX = n_bY_bX_oC_iY / output_depth_by_input_height;
                const int oC_iY = n_bY_bX_oC_iY - n_bY_bX * output_depth_by_input_height;

                const int n_bY = n_bY_bX / block_size;
                const int bX = n_bY_bX - n_bY * block_size;

                const int n = n_bY / block_size;
                const int bY = n_bY - n * block_size;

                const int output_idx = bX + block_size * (iX + input_width * (bY + block_size * (oC_iY + n * output_depth_by_input_height)));

                (output_ptr + output_idx)[0] = (input_ptr + input_idx)[0];
            }
        }
    }

    void _depthToSpace(NDArray *input, NDArray *output, int block_size, bool isNHWC) {
        auto xType = input->dataType();

        BUILD_SINGLE_SELECTOR(xType, __depthToSpace, (input, output, block_size, isNHWC), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void __depthToSpace, (NDArray *input, NDArray *output, int block_size, bool isNHWC);, LIBND4J_TYPES);

}
}
}