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
  const auto input_stride = shape::stride(input.shapeInfo());
  const auto output_stride = shape::stride(output->shapeInfo());

  const LongType batch_size = input.sizeAt(0);
  const LongType input_depth = isNHWC ? input.sizeAt(3) : input.sizeAt(1);
  const LongType input_height = isNHWC ? input.sizeAt(1) : input.sizeAt(2);
  const LongType input_width = isNHWC ? input.sizeAt(2) : input.sizeAt(3);

  const LongType output_depth = isNHWC ? output->sizeAt(3) : output->sizeAt(1);
  const LongType output_height = isNHWC ? output->sizeAt(1) : output->sizeAt(2);
  const LongType output_width = isNHWC ? output->sizeAt(2) : output->sizeAt(3);

  const LongType input_depth_by_output_height = input_depth * output_height;

  const LongType output_area = output_width * output_height;
  const LongType output_depth_by_output_area = output_depth * output_area;

  if (isNHWC) {
    const LongType total_count = batch_size * input_height * input_width * input_depth;

    auto func = PRAGMA_THREADS_FOR {
      for (auto inp_idx = start; inp_idx < stop; inp_idx += increment) {
        // inp_idx = d + input_depth * (w + input_width * (h + input_height * b))
        const LongType d = inp_idx % input_depth;
        const LongType inp_idx2 = inp_idx / input_depth;
        const LongType w = inp_idx2 % input_width;
        const LongType inp_idx3 = inp_idx2 / input_width;
        const LongType h = inp_idx3 % input_height;
        const LongType b = inp_idx3 / input_height;

        const LongType out_h = h / block_size;
        const LongType offset_h = h % block_size;
        const LongType out_w = w / block_size;
        const LongType offset_w = w % block_size;
        const LongType offset_d = (offset_h * block_size + offset_w) * input_depth;
        const LongType out_d = d + offset_d;

        const auto input_offset = b * input_stride[0] + h * input_stride[1] + w * input_stride[2] + d * input_stride[3];
        const auto output_offset = b * output_stride[0] + out_h * output_stride[1] +
                                   out_w * output_stride[2] + out_d * output_stride[3];
        output_ptr[output_offset] = input_ptr[input_offset];
      }
    };

    samediff::Threads::parallel_for(func, 0, total_count);
  } else {
    const LongType total_count = batch_size * output_depth_by_output_area;

    auto func = PRAGMA_THREADS_FOR {
      for (auto inp_idx = start; inp_idx < stop; inp_idx += increment) {
        const LongType n_iC_oY_bY_oX = inp_idx / block_size;
        const LongType bX = inp_idx - n_iC_oY_bY_oX * block_size;

        const LongType n_iC_oY_bY = n_iC_oY_bY_oX / output_width;
        const LongType oX = n_iC_oY_bY_oX - n_iC_oY_bY * output_width;

        const LongType n_iC_oY = n_iC_oY_bY / block_size;
        const LongType bY = n_iC_oY_bY - n_iC_oY * block_size;

        const LongType n = n_iC_oY / input_depth_by_output_height;
        const LongType iC_oY = n_iC_oY - n * input_depth_by_output_height;

        const LongType iC = iC_oY / output_height;
        const LongType oY = iC_oY % output_height;
        const LongType oC = (bY * block_size + bX) * input_depth + iC;
        const LongType iY = oY * block_size + bY;
        const LongType iX = oX * block_size + bX;
        const auto input_offset = n * input_stride[0] + iC * input_stride[1] +
                                  iY * input_stride[2] + iX * input_stride[3];
        const auto output_offset = n * output_stride[0] + oC * output_stride[1] +
                                   oY * output_stride[2] + oX * output_stride[3];
        output_ptr[output_offset] = input_ptr[input_offset];
      }
    };

    samediff::Threads::parallel_for(func, 0, total_count);
  }
}

void _spaceTodepth(sd::LaunchContext *context, NDArray&input, NDArray *output, int block_size, bool isNHWC) {
  NDArray::preparePrimaryUse({output}, {&input});
  BUILD_SINGLE_SELECTOR(input.dataType(), _spaceTodepth_, (input, output, block_size, isNHWC), SD_COMMON_TYPES);
  NDArray::registerPrimaryUse({output}, {&input});
}

BUILD_SINGLE_TEMPLATE( void _spaceTodepth_,
                      (NDArray&input, NDArray *output, int block_size, bool isNHWC), SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif