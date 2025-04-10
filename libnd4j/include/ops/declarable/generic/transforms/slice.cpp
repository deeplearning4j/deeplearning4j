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
// Created by raver119 on 02.11.2017.
//

#include <system/op_boilerplate.h>
#include <legacy/NativeOpExecutioner.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#if NOT_EXCLUDED(OP_slice)
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(slice, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int x_rank = input->rankOf();



  std::vector<LongType> begin;
  std::vector<LongType> sz;

  if (block.width() == 3) {
    auto b = INPUT_VARIABLE(1);
    auto e = INPUT_VARIABLE(2);

    begin = b->template asVectorT<LongType>();
    sz = e->template asVectorT<LongType>();
  } else {
    REQUIRE_TRUE(block.numI() >= static_cast<size_t>(x_rank * 2), 0, "Number of IArgs should be equal to [%i] but got [%i] instead",
                 x_rank * 2, block.numI());

    ShapeUtils::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
    ShapeUtils::copyVectorPart(sz, *(block.getIArguments()), x_rank, x_rank);
  }

  REQUIRE_TRUE(begin.size() == static_cast<size_t>(x_rank), 0, "begin array should have length of [%i] but got [%i] instead", x_rank,
               begin.size());
  REQUIRE_TRUE(sz.size() == static_cast<size_t>(x_rank), 0, "size array should have length of [%i] but got [%i] instead", x_rank, sz.size());

  std::vector<LongType> indices(2 * x_rank);
  auto empty = false;
  for (int e = 0; e < x_rank; e++) {
    int size = sz[e];
    int start = begin[e];

    REQUIRE_TRUE(start >= 0, 0, "Slice: start index should not be negative");

    REQUIRE_TRUE(start <= input->sizeAt(e), 0, "Index %i is invalid for dimension %i with size %i.", start, e,
                 input->shapeInfo()[e + 1]);
    if (size == -1) {
      size = input->sizeAt(e) - start;
    }
    REQUIRE_TRUE(size >= 0, 0, "Slice: interval for dimension %i is less then 1");
    REQUIRE_TRUE(start + size <= input->sizeAt(e), 0,
                 "Slice: interval [%i, %i] is out of bounds for dimension %i with size %i", start, start + size, e,
                 input->sizeAt(e));

    if (start == input->sizeAt(e) || size == 0) {
      empty = true;
      // Don't break to perform input validation on other dims
    }

    indices[2 * e] = start;
    indices[2 * e + 1] = start + size;
  }

  if (empty) {
    REQUIRE_TRUE(output->isEmpty(), 0, "Slice: empty array indices requested, but output array is not empty");
    return Status::OK;
  }

  LongType* subArrShapeInfo = nullptr;
  ALLOCATE(subArrShapeInfo, block.getWorkspace(), shape::shapeInfoLength(input->rankOf()), sd::LongType);

  LongType offset;

  shape::calcSubArrShapeInfoAndOffset(indices.data(), input->shapeInfo(), subArrShapeInfo, offset, true);

  auto subArrShapeInfoPack = ConstantShapeHelper::getInstance().bufferForShapeInfo(subArrShapeInfo);

  NDArray::prepareSpecialUse({output}, {input});

  NativeOpExecutioner::execTransformAny(block.launchContext(), transform::Assign, input->bufferWithOffset(offset),
                                        subArrShapeInfoPack->primary(), input->specialBufferWithOffset(offset),
                                        subArrShapeInfoPack->special(), output->buffer(), output->shapeInfo(),
                                        output->specialBuffer(), output->specialShapeInfo(), nullptr, true);

  NDArray::registerSpecialUse({output}, {input});

  RELEASE(subArrShapeInfo, block.getWorkspace());


  STORE_RESULT(output);

  return Status::OK;
}

DECLARE_TYPES(slice) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(slice) {
  auto inShape = inputShape->at(0);
  if(shape::isEmptyConst(inShape)) {
    std::vector<LongType> emptyShape = {0};
    return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(ArrayOptions::dataType(inShape), emptyShape));
  }
  auto x_rank = shape::rank(inShape);

  std::vector<LongType> begin;
  std::vector<LongType> sz;

  if (block.width() == 3) {
    auto b = INPUT_VARIABLE(1);
    auto e = INPUT_VARIABLE(2);

    begin = b->template asVectorT<LongType>();
    sz = e->template asVectorT<LongType>();
  } else {
    REQUIRE_TRUE(block.numI() >= static_cast<size_t>(x_rank) * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead",
                 x_rank * 2, block.numI());

    ShapeUtils::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
    ShapeUtils::copyVectorPart(sz, *(block.getIArguments()), x_rank, x_rank);
  }

  REQUIRE_TRUE(begin.size() == static_cast<size_t>(x_rank), 0, "Begin array should have length of [%i] but got [%i] instead", x_rank,
               begin.size());
  REQUIRE_TRUE(sz.size() == static_cast<size_t>(x_rank), 0, "Size array should have length of [%i] but got [%i] instead", x_rank, sz.size());

  std::vector<LongType> shape;
  auto empty = false;
  for (int e = 0; e < x_rank; e++) {
    auto size = sz[e];
    auto start = begin[e];

    if (size == -1) {
      size = inShape[e + 1] - start;
    }

    // Bounds checking. Note that begin[i] == size[i] means empty array
    REQUIRE_TRUE(
        start >= 0 && start <= inShape[e + 1], 0,
        "Invalid begin[%i] value: Begin must satisfy 0 <= begin <= size[i], got begin=%i for dimension size %i", e,
        start, inShape[e + 1]);
    REQUIRE_TRUE(size == -1 || size >= 0, 0,
                 "Invalid size[%i] value: must be positive (or -1 for 'all remaining'), got %i", e, size,
                 inShape[e + 1]);
    REQUIRE_TRUE(
        start >= 0 && start <= inShape[e + 1], 0,
        "Invalid begin[%i] value: Begin must satisfy 0 <= begin <= size[i], got begin=%i for dimension size %i", e,
        start, inShape[e + 1]);
    REQUIRE_TRUE(start + size <= inShape[e + 1], 0,
                 "Slice: interval [%i, %i] is out of bounds for dimension %i with size %i", start, start + size, e,
                 inShape[e + 1]);
    if (start == inShape[e + 1]) {
      size = 0;
    }

    shape.emplace_back(size);
  }

  if(shape.size() == 1 && shape[0] == 0) {
    std::vector<LongType> emptyShape = {0};
    return SHAPELIST(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(ArrayOptions::dataType(inShape), emptyShape));
  }

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), 'c', shape);
  return SHAPELIST(newShape);
}

DECLARE_TYPES(slice_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

CUSTOM_OP_IMPL(slice_bp, 2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto epsNext = block.width() == 4 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);
  double zero = 0.;
  output->assign(zero);
  int x_rank = input->rankOf();

  std::vector<LongType> begin;
  std::vector<LongType> end;

  if (block.width() == 4) {
    auto b = INPUT_VARIABLE(1);
    auto e = INPUT_VARIABLE(2);

    begin = b->template asVectorT<LongType>();
    end = e->template asVectorT<LongType>();
  } else {
    REQUIRE_TRUE(block.numI() >= static_cast<size_t>(x_rank) * 2, 0, "Number of IArgs should be equal to [%i] but got [%i] instead",
                 x_rank * 2, block.numI());

    ShapeUtils::copyVectorPart(begin, *(block.getIArguments()), x_rank, 0);
    ShapeUtils::copyVectorPart(end, *(block.getIArguments()), x_rank, x_rank);
  }

  REQUIRE_TRUE(begin.size() == static_cast<size_t>(x_rank), 0, "begin array should have length of [%i] but got [%i] instead", x_rank,
               begin.size());
  REQUIRE_TRUE(end.size() == static_cast<size_t>(x_rank), 0, "end array should have length of [%i] but got [%i] instead", x_rank,
               end.size());

  std::vector<LongType> indices(2 * x_rank);
  for (int e = 0; e < x_rank; e++) {
    int size = end[e];
    int start = begin[e];

    if (size == -1) {  //-1 means all remaining values
      size = input->sizeAt(e) - start;
    }
    REQUIRE_TRUE(size > 0, 0, "Slice: interval for dimension %i is less then 1", e);

    indices[2 * e] = start;
    indices[2 * e + 1] = start + size;
  }
  auto sub = (*output)(indices, true);
  sub.assign(epsNext);

  return Status::OK;
}

DECLARE_SHAPE_FN(slice_bp) {
  auto inShape = inputShape->at(0);
  return SHAPELIST(CONSTANT(inShape));
}
}  // namespace ops
}  // namespace sd

#endif
