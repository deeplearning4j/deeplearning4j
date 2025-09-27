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
//
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/s_t_d.h>
#if NOT_EXCLUDED(OP_space_to_depth)
namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static void _spaceTodepth_(NDArray&input, NDArray *output, int block_size, bool isNHWC) {
  auto input_ptr = reinterpret_cast<T const *>(input.buffer());
  auto output_ptr = reinterpret_cast<T *>(output->buffer());

  const int batch_size = input.sizeAt(0);
  const int input_depth = isNHWC ? input.sizeAt(3) : input.sizeAt(1);
  const int input_height = isNHWC ? input.sizeAt(1) : input.sizeAt(2);
  const int input_width = isNHWC ? input.sizeAt(2) : input.sizeAt(3);

  const int output_depth = isNHWC ? output->sizeAt(3) : output->sizeAt(1);
  const int output_height = isNHWC ? output->sizeAt(1) : output->sizeAt(2);
  const int output_width = isNHWC ? output->sizeAt(2) : output->sizeAt(3);

  const int input_depth_by_output_height = input_depth * output_height;

  const int output_area = output_width * output_height;
  const int output_depth_by_output_area = output_depth * output_area;

  if (isNHWC) {
    const int total_count = batch_size * input_height * input_width * input_depth;

    auto func = PRAGMA_THREADS_FOR {
      for (auto inp_idx = start; inp_idx < stop; inp_idx++) {
        // inp_idx = d + input_depth * (w + input_width * (h + input_height * b))
        const int d = inp_idx % input_depth;
        const int inp_idx2 = inp_idx / input_depth;
        const int w = inp_idx2 % input_width;
        const int inp_idx3 = inp_idx2 / input_width;
        const int h = inp_idx3 % input_height;
        const int b = inp_idx3 / input_height;

        const int out_h = h / block_size;
        const int offset_h = h % block_size;
        const int out_w = w / block_size;
        const int offset_w = w % block_size;
        const int offset_d = (offset_h * block_size + offset_w) * input_depth;
        const int out_d = d + offset_d;

        const int out_idx = out_d + output_depth * (out_w + output_width * (out_h + output_height * b));
        *(output_ptr + out_idx) = *(input_ptr + inp_idx);
      }
    };

    samediff::Threads::parallel_for(func, 0, total_count);
  } else {
    const int total_count = batch_size * output_depth_by_output_area;

    auto func = PRAGMA_THREADS_FOR {
      for (auto inp_idx = start; inp_idx < stop; inp_idx++) {
        const int n_iC_oY_bY_oX = inp_idx / block_size;
        const int bX = inp_idx - n_iC_oY_bY_oX * block_size;

        const int n_iC_oY_bY = n_iC_oY_bY_oX / output_width;
        const int oX = n_iC_oY_bY_oX - n_iC_oY_bY * output_width;

        const int n_iC_oY = n_iC_oY_bY / block_size;
        const int bY = n_iC_oY_bY - n_iC_oY * block_size;

        const int n = n_iC_oY / input_depth_by_output_height;
        const int iC_oY = n_iC_oY - n * input_depth_by_output_height;

        const int output_idx =
            oX + (((n * block_size + bY) * block_size + bX) * input_depth_by_output_height + iC_oY) * output_width;

        *(output_ptr + output_idx) = *(input_ptr + inp_idx);
      }
    };

    samediff::Threads::parallel_for(func, 0, total_count);
  }
}

void _spaceTodepth(sd::LaunchContext *context, NDArray&input, NDArray *output, int block_size, bool isNHWC) {
  BUILD_SINGLE_SELECTOR(input.dataType(), _spaceTodepth_, (input, output, block_size, isNHWC), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void _spaceTodepth_,
                      (NDArray&input, NDArray *output, int block_size, bool isNHWC), SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif