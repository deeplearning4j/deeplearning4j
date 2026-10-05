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

#ifndef LIBND4J_DENSE_OUTPUT_SHAPE_H
#define LIBND4J_DENSE_OUTPUT_SHAPE_H

#include <helpers/ConstantShapeHelper.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/shape.h>
#include <system/common.h>

namespace sd {

/**
 * The extras bits that make a shape info describe storage that already exists rather than a new
 * array: the view flag, the "needs a copy" flag, the padded-buffer flag and the copy-offset flag of
 * every input. A shape function sets them on an output that is a view of an input (permute,
 * transpose, reshape_no_copy, kv_scatter, ...), or inherits them when it returns an input's shape
 * info for an output of its own.
 */
constexpr LongType ALIAS_EXTRAS_MASK =
    ARRAY_IS_VIEW | ARRAY_NEEDS_COPY | ARRAY_HAS_PADDED_BUFFER | ARRAY_COPY_OFFSET_INPUT_0 |
    ARRAY_COPY_OFFSET_INPUT_1 | ARRAY_COPY_OFFSET_INPUT_2 | ARRAY_COPY_OFFSET_INPUT_3 |
    ARRAY_COPY_OFFSET_INPUT_4 | ARRAY_COPY_OFFSET_INPUT_5 | ARRAY_COPY_OFFSET_INPUT_6 |
    ARRAY_COPY_OFFSET_INPUT_7 | ARRAY_COPY_OFFSET_INPUT_8 | ARRAY_COPY_OFFSET_INPUT_9 |
    ARRAY_COPY_OFFSET_INPUT_10;

/**
 * True when the strides pack the array's elements into exactly length() consecutive offsets in the
 * shape info's order: every dimension longer than 1 has the stride of a packed array of that order.
 * Dimensions of length 1 are ignored, their index is always 0 so their stride never contributes to an
 * address (ND4J stores a [1, K] row with strides [1, 1]). The order follows
 * shape::updateStrides: 'c' is row-major, anything else column-major.
 */
SD_INLINE bool shapeInfoIsPackedInOrder(const LongType* shapeInfo) {
  const LongType rank = shape::rank(shapeInfo);
  const LongType* dims = shape::shapeOf(shapeInfo);
  const LongType* strides = shape::stride(shapeInfo);
  LongType expected = 1;
  if (shape::order(shapeInfo) == 'c') {
    for (LongType d = rank - 1; d >= 0; --d) {
      if (dims[d] != 1 && strides[d] != expected) return false;
      expected *= dims[d];
    }
  } else {
    for (LongType d = 0; d < rank; ++d) {
      if (dims[d] != 1 && strides[d] != expected) return false;
      expected *= dims[d];
    }
  }
  return true;
}

/**
 * The shape info of the array the framework allocates for an op output that a shape function
 * describes with `descriptor`: a new array in the descriptor's shape, data type and order with dense
 * strides, and without the view, copy and copy-offset flags. The empty flag stays.
 *
 * A shape function may return an input's shape info for an output of its own (CONSTANT(inShape),
 * bufferForShapeInfo(inShape), ShapeBuilders::copyShapeInfoAndType with copyStrides, the default
 * shape functions of the legacy transform ops), and a view-producing op describes its view output
 * with the input's strides. Only a new array of exactly length() elements can be addressed by those
 * strides when they are packed: a stepped view such as a [4, 70] slice with strides [140, 2]
 * addresses offset 558 of a 280-element buffer, so an output allocated from the descriptor as it is
 * writes past its buffer. The view flag of the descriptor also stops ~NDArray from freeing the
 * buffer it owns, and the copy-offset flag stops DSP from zeroing an output that needs it.
 *
 * Use this wherever a descriptor becomes an allocation: the result is the descriptor itself when
 * it already describes such an array (nothing is interned, nothing allocated), otherwise a cached
 * constant shape info. An output that is a view of an input is not allocated from a descriptor, so
 * it keeps the descriptor as it is. No array is allocated for an empty descriptor (the flag, or a
 * zero-length dimension): it is returned as it is.
 *
 * The result is owned by the constant shape cache, never freed by the caller.
 */
SD_INLINE LongType* denseOutputShapeInfo(const LongType* descriptor) {
  auto* unchanged = const_cast<LongType*>(descriptor);
  if (descriptor == nullptr) THROW_EXCEPTION("denseOutputShapeInfo: the descriptor is null");

  if (shape::isEmptyConst(descriptor) || shape::length(descriptor) == 0) return unchanged;

  const LongType extra = ArrayOptions::extra(descriptor);
  if ((extra & ALIAS_EXTRAS_MASK) == 0 && shapeInfoIsPackedInOrder(descriptor)) return unchanged;

  // Dense strides in the descriptor's order; every other bit of the extras (data type and the like) stays.
  LongType* dense = ShapeBuilders::copyShapeInfo(descriptor, false, nullptr);
  ArrayOptions::setExtra(dense, extra & ~ALIAS_EXTRAS_MASK);
  LongType* interned = ConstantShapeHelper::getInstance().bufferForShapeInfo(dense)->primary();
  delete[] dense;
  return interned;
}

/**
 * True when every offset the strides address lies inside the array's own length() elements: a dense
 * array, and a view that only permutes the dimensions of one (permute, transpose). A view that skips
 * elements of a larger array (a slice of a wider array, a step) addresses past length(); broadcast
 * (stride 0) and reversed (negative stride) views are not accepted either. Dimensions of length 1
 * are ignored, their index is always 0.
 */
SD_INLINE bool shapeInfoFitsItsLength(const LongType* shapeInfo) {
  const LongType rank = shape::rank(shapeInfo);
  const LongType* dims = shape::shapeOf(shapeInfo);
  const LongType* strides = shape::stride(shapeInfo);
  LongType extent = 1;
  for (LongType d = 0; d < rank; ++d) {
    if (dims[d] == 1) continue;
    if (dims[d] < 1 || strides[d] < 1) return false;
    extent += (dims[d] - 1) * strides[d];
  }
  return extent == shape::length(shapeInfo);
}

/**
 * The shape info of the placeholder the dynamic shape plan publishes, while it infers the shapes of a
 * graph, for the primary output of a view-capable op. At run time that output is a view of the op's
 * input with the strides its shape function describes, and the shape functions of the ops after it
 * read the strides of whatever stands in for it. The placeholder keeps the view's strides when they
 * address only its own buffer (a permutation of dense strides: permute, transpose, a reshape or an
 * expand_dims of a dense array), which is all a buffer of length() elements can hold, so inference
 * sees the layout the slot will have. Strides that address more (a slice of a wider array) cannot be
 * allocated and the placeholder is dense like any other output (denseOutputShapeInfo). The view, copy
 * and copy-offset flags are dropped either way: the placeholder owns its buffer.
 *
 * The result is the descriptor itself when nothing changes, otherwise a cached constant shape info,
 * never freed by the caller.
 */
SD_INLINE LongType* viewStandInShapeInfo(const LongType* descriptor) {
  auto* unchanged = const_cast<LongType*>(descriptor);
  if (descriptor == nullptr) THROW_EXCEPTION("viewStandInShapeInfo: the descriptor is null");

  if (shape::isEmptyConst(descriptor) || shape::length(descriptor) == 0) return unchanged;
  if (!shapeInfoFitsItsLength(descriptor)) return denseOutputShapeInfo(descriptor);

  const LongType extra = ArrayOptions::extra(descriptor);
  if ((extra & ALIAS_EXTRAS_MASK) == 0) return unchanged;

  // The strides stay, every other bit of the extras (data type and the like) stays.
  LongType* standIn = ShapeBuilders::copyShapeInfo(descriptor, true, nullptr);
  ArrayOptions::setExtra(standIn, extra & ~ALIAS_EXTRAS_MASK);
  LongType* interned = ConstantShapeHelper::getInstance().bufferForShapeInfo(standIn)->primary();
  delete[] standIn;
  return interned;
}

}  // namespace sd

#endif  // LIBND4J_DENSE_OUTPUT_SHAPE_H
