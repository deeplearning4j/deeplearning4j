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
//  @author GS <sgazeos@gmail.com>
//
//  mask[..., j] = j < lengths[...]: one thread per element of the mask (grid-stride, 64 bit indices), every element
//  written (true or false). The lengths are read through their own shape and strides and the mask is written through
//  its own strides, so views and F order are correct; a negative length gives an all-false row (the comparison is a
//  signed 64 bit one for the signed types and an unsigned 64 bit one for the unsigned types).
//
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <ops/declarable/helpers/segment_semantics.h>
#include <ops/declarable/helpers/sequence_mask.h>

#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {

template <typename I, typename B>
static SD_KERNEL void sequenceMaskKernel(const I* lengths, const LongType* lengthsShapeInfo, B* mask,
                                         const LongType* maskShapeInfo, LongType width, LongType total) {
  const LongType inRank = shape::rank(lengthsShapeInfo);
  const LongType* inShape = shape::shapeOf(lengthsShapeInfo);
  const LongType* inStride = shape::stride(lengthsShapeInfo);
  const LongType outRank = shape::rank(maskShapeInfo);
  const LongType* outStride = shape::stride(maskShapeInfo);
  const LongType columnStride = outStride[outRank - 1];

  for (LongType g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; g < total;
       g += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType row = g / width;
    const LongType column = g - row * width;
    const I length = lengths[segment_sem::logicalOffset(row, inRank, inShape, inStride)];
    // the row's coordinates over the leading dimensions (the lengths' shape), through the mask's strides
    const LongType rowOffset = segment_sem::logicalOffset(row, inRank, inShape, outStride);
    bool on;
    if constexpr (std::is_signed<I>::value) {
      on = column < static_cast<LongType>(length);
    } else {
      on = static_cast<UnsignedLong>(column) < static_cast<UnsignedLong>(length);
    }
    mask[rowOffset + column * columnStride] = static_cast<B>(on ? 1.0f : 0.0f);
  }
}

template <typename I, typename B>
static void sequenceMask_(LaunchContext* context, NDArray* input, NDArray* output, LongType width) {
  const LongType total = output->lengthOf();
  if (total == 0 || width <= 0) return;
  dim3 launchDims = getSequenceMaskLaunchDims(width > 2147483647 ? 2147483647 : static_cast<int>(width), *input);
  NDArray::prepareSpecialUse({output}, {input});
  auto stream = context->getCudaStream();
  sequenceMaskKernel<I, B><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      reinterpret_cast<const I*>(input->specialBuffer()), input->specialShapeInfo(),
      reinterpret_cast<B*>(output->specialBuffer()), output->specialShapeInfo(), width, total);
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("sequenceMaskKernel failed");
  NDArray::registerSpecialUse({output}, {input});
}

// The width of the mask is the last dimension of the output (the shape function owns the contract); maxIndex is the
// same number.
void sequenceMask(LaunchContext* context, NDArray* input, NDArray* output, int maxIndex) {
  const LongType width = output->sizeAt(output->rankOf() - 1);
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), sequenceMask_, (context, input, output, width),
                        SD_INTEGER_TYPES, SD_COMMON_TYPES_EXTENDED);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
