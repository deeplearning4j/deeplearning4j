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
//  mask[..., j] = j < lengths[...], every element written (true or false). The lengths are read through their own
//  shape and strides and the mask is written through its own strides (views, F order); a negative length gives an
//  all-false row (a signed 64 bit comparison for the signed types, an unsigned 64 bit one for the unsigned types).
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/segment_semantics.h>
#include <ops/declarable/helpers/sequence_mask.h>

#include <type_traits>

#if NOT_EXCLUDED(OP_sequence_mask)
namespace sd {
namespace ops {
namespace helpers {

template <typename I, typename B>
static void sequenceMask_(NDArray* input, NDArray* output, LongType width) {
  const LongType rows = input->lengthOf();
  if (rows == 0 || width <= 0) return;
  const I* lengths = input->bufferAsT<I>();
  B* mask = output->bufferAsT<B>();
  const LongType inRank = input->rankOf();
  const LongType* inShape = shape::shapeOf(input->shapeInfo());
  const LongType* inStride = shape::stride(input->shapeInfo());
  const LongType outRank = output->rankOf();
  const LongType* outStride = shape::stride(output->shapeInfo());
  const LongType columnStride = outStride[outRank - 1];

  auto func = PRAGMA_THREADS_FOR {
    for (LongType row = start; row < stop; ++row) {
      const I length = lengths[segment_sem::logicalOffset(row, inRank, inShape, inStride)];
      // the row's coordinates over the leading dimensions (the lengths' shape), through the mask's strides
      B* maskRow = mask + segment_sem::logicalOffset(row, inRank, inShape, outStride);
      for (LongType column = 0; column < width; ++column) {
        bool on;
        if constexpr (std::is_signed<I>::value) {
          on = column < static_cast<LongType>(length);
        } else {
          on = static_cast<UnsignedLong>(column) < static_cast<UnsignedLong>(length);
        }
        maskRow[column * columnStride] = static_cast<B>(on ? 1.0f : 0.0f);
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, rows);
}

// The width of the mask is the last dimension of the output (the shape function owns the contract); maxIndex is the
// same number.
void sequenceMask(sd::LaunchContext* context, NDArray* input, NDArray* output, int maxIndex) {
  NDArray::preparePrimaryUse({output}, {input});
  const LongType width = output->sizeAt(output->rankOf() - 1);
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), sequenceMask_, (input, output, width), SD_INTEGER_TYPES,
                        SD_COMMON_TYPES_EXTENDED);
  NDArray::registerPrimaryUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
