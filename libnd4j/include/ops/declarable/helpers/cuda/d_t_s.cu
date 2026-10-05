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
#include <ops/declarable/helpers/d_t_s.h>
#if NOT_EXCLUDED(OP_depth_to_space)
#include <helpers/DebugHelper.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void depthToSpaceKernel(const void *vx, const LongType *xShapeInfo, void *vz,
                                         const LongType *zShapeInfo, const LongType block_size, const bool isNHWC) {
  auto input_ptr = reinterpret_cast<const T *>(vx);
  auto output_ptr = reinterpret_cast<T *>(vz);
  const auto input_stride = shape::stride(xShapeInfo);
  const auto output_stride = shape::stride(zShapeInfo);

  const LongType batch_size = shape::sizeAt(xShapeInfo, 0);
  const LongType input_depth = isNHWC ? shape::sizeAt(xShapeInfo, 3) : shape::sizeAt(xShapeInfo, 1);
  const LongType input_height = isNHWC ? shape::sizeAt(xShapeInfo, 1) : shape::sizeAt(xShapeInfo, 2);
  const LongType input_width = isNHWC ? shape::sizeAt(xShapeInfo, 2) : shape::sizeAt(xShapeInfo, 3);

  const LongType output_depth = isNHWC ? shape::sizeAt(zShapeInfo, 3) : shape::sizeAt(zShapeInfo, 1);
  const LongType output_height = isNHWC ? shape::sizeAt(zShapeInfo, 1) : shape::sizeAt(zShapeInfo, 2);
  const LongType output_width = isNHWC ? shape::sizeAt(zShapeInfo, 2) : shape::sizeAt(zShapeInfo, 3);

  const LongType input_area = input_width * input_height;
  const LongType input_depth_by_input_area = input_depth * input_area;
  const LongType output_depth_by_input_height = output_depth * input_height;

  const LongType tid = threadIdx.x + static_cast<LongType>(blockIdx.x) * blockDim.x;
  const LongType step = static_cast<LongType>(blockDim.x) * gridDim.x;

  if (isNHWC) {
    const LongType total_count = batch_size * output_height * output_width * output_depth;
    for (LongType out_idx = tid; out_idx < total_count; out_idx += step) {
      const LongType d = out_idx % output_depth;
      const LongType out_idx2 = out_idx / output_depth;
      const LongType w = out_idx2 % output_width;
      const LongType out_idx3 = out_idx2 / output_width;
      const LongType h = out_idx3 % output_height;
      const LongType b = out_idx3 / output_height;

      const LongType in_h = h / block_size;
      const LongType offset_h = h % block_size;
      const LongType in_w = w / block_size;
      const LongType offset_w = w % block_size;
      const LongType offset_d = (offset_h * block_size + offset_w) * output_depth;
      const LongType in_d = d + offset_d;
      const auto inp_idx = b * input_stride[0] + in_h * input_stride[1] +
                           in_w * input_stride[2] + in_d * input_stride[3];
      const auto output_idx = b * output_stride[0] + h * output_stride[1] +
                              w * output_stride[2] + d * output_stride[3];
      output_ptr[output_idx] = input_ptr[inp_idx];
    }
  } else {
    const LongType total_count = batch_size * input_depth_by_input_area;

    for (LongType input_idx = tid; input_idx < total_count; input_idx += step) {
      const LongType n_bY_bX_oC_iY = input_idx / input_width;
      const LongType iX = input_idx - n_bY_bX_oC_iY * input_width;

      const LongType n_bY_bX = n_bY_bX_oC_iY / output_depth_by_input_height;
      const LongType oC_iY = n_bY_bX_oC_iY - n_bY_bX * output_depth_by_input_height;

      const LongType n_bY = n_bY_bX / block_size;
      const LongType bX = n_bY_bX - n_bY * block_size;

      const LongType n = n_bY / block_size;
      const LongType bY = n_bY - n * block_size;

      const LongType oC = oC_iY / input_height;
      const LongType iY = oC_iY % input_height;
      const LongType iC = (bY * block_size + bX) * output_depth + oC;
      const auto inp_idx = n * input_stride[0] + iC * input_stride[1] +
                           iY * input_stride[2] + iX * input_stride[3];
      const auto output_idx = n * output_stride[0] + oC * output_stride[1] +
                              (iY * block_size + bY) * output_stride[2] +
                              (iX * block_size + bX) * output_stride[3];
      output_ptr[output_idx] = input_ptr[inp_idx];
    }
  }
}

template <typename T>
static void __depthToSpace(LaunchContext *context, NDArray&input, NDArray *output, int block_size,
                           bool isNHWC) {
  dim3 launchDims = getLaunchDims("depth_to_space");
  depthToSpaceKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(input.specialBuffer(), input.specialShapeInfo(),
                                                                       output->specialBuffer(),
                                                                       output->specialShapeInfo(), block_size, isNHWC);
  if (!DebugHelper::inGraphCapture(context->getCudaStream())) {
    DebugHelper::checkGlobalErrorCode("depthToSpaceKernel failed");
  }

}

void _depthToSpace(LaunchContext *context, NDArray&input, NDArray *output, int block_size, bool isNHWC) {
  auto xType = input.dataType();

  if (input.lengthOf() == 0) return;
  NDArray::prepareSpecialUse({output}, {&input});

  BUILD_SINGLE_SELECTOR(xType, __depthToSpace, (context, input, output, block_size, isNHWC), SD_COMMON_TYPES);
  NDArray::registerSpecialUse({output}, {&input});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
