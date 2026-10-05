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
//  @author sgazeos@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_crop_and_resize)

#include <ops/declarable/headers/images.h>
#include <ops/declarable/helpers/crop_and_resize.h>

namespace sd {
namespace ops {
// Shape inference precedes execution/output allocation and must reject malformed descriptors before reading axes.
static void validateCropAndResize(const LongType* imageShape, const LongType* boxesShape,
                                  const LongType* indicesShape, NDArray* cropSize, LongType method,
                                  int numI, int numT) {
  REQUIRE_TRUE(shape::rank(imageShape) == 4, 0,
               "crop_and_resize: the images must be rank 4, [batch, height, width, channels].");
  REQUIRE_TRUE(shape::rank(boxesShape) == 2 && shape::sizeAt(boxesShape, 1) == 4, 0,
               "crop_and_resize: the boxes must be [number of boxes, 4].");
  REQUIRE_TRUE(DataTypeUtils::isR(ArrayOptions::dataType(imageShape)) ||
               DataTypeUtils::isZ(ArrayOptions::dataType(imageShape)), 0,
               "crop_and_resize: images must have a numeric data type.");
  REQUIRE_TRUE(DataTypeUtils::isR(ArrayOptions::dataType(boxesShape)), 0,
               "crop_and_resize: boxes must have a floating-point data type.");
  REQUIRE_TRUE(DataTypeUtils::isZ(ArrayOptions::dataType(indicesShape)) &&
               DataTypeUtils::isZ(cropSize->dataType()), 0,
               "crop_and_resize: box indices and crop size must have integer data types.");
  REQUIRE_TRUE(shape::length(indicesShape) >= shape::sizeAt(boxesShape, 0), 0,
               "crop_and_resize: every box needs an index.");
  REQUIRE_TRUE(shape::sizeAt(imageShape, 1) > 0 && shape::sizeAt(imageShape, 2) > 0, 0,
               "crop_and_resize: image height and width must be positive.");
  REQUIRE_TRUE(numI <= 1 && (method == 0 || method == 1), 0,
               "crop_and_resize: expected one optional method, 0 (bilinear) or 1 (nearest).");
  REQUIRE_TRUE(numT <= 1, 0, "crop_and_resize: expected one optional extrapolation value.");
  REQUIRE_TRUE(cropSize->lengthOf() == 2, 0, "crop_and_resize: crop size must contain height and width.");
  REQUIRE_TRUE(cropSize->e<LongType>(0) > 0 && cropSize->e<LongType>(1) > 0, 0,
               "crop_and_resize: crop height and width must be positive.");
}

CUSTOM_OP_IMPL(crop_and_resize, 4, 1, false, 0, 0) {
  auto image = INPUT_VARIABLE(0);
  auto boxes = INPUT_VARIABLE(1);
  auto boxIndexes = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);
  const LongType method = block.numI() > 0 ? INT_ARG(0) : 0;  // bilinear
#ifdef HAS_DOUBLE
  double extrapolationVal = 0.;
#elif defined(HAS_FLOAT32)
  float extrapolationVal = 0.0f;
#else
#error "No floating-point type available for crop_and_resize operation"
#endif

  auto newImageSize = INPUT_VARIABLE(3);
  REQUIRE_TRUE(output->dataType() == image->dataType(), 0,
               "crop_and_resize: Source images and output should have the same data type.");
  validateCropAndResize(image->shapeInfo(), boxes->shapeInfo(), boxIndexes->shapeInfo(), newImageSize, method,
                      block.numI(), block.numT());

  if (block.numT() == 1) {
#ifdef HAS_DOUBLE
    extrapolationVal = static_cast<double>(T_ARG(0));
#elif defined(HAS_FLOAT32)
    extrapolationVal = static_cast<float>(T_ARG(0));
#else
#error "No floating-point type available for crop_and_resize operation"
#endif
  }

  helpers::cropAndResizeFunctor(block.launchContext(), image, boxes, boxIndexes, newImageSize, static_cast<int>(method), extrapolationVal,
                                output);
  return sd::Status::OK;
}

DECLARE_SHAPE_FN(crop_and_resize) {
  auto in = inputShape->at(0);
  auto boxShape = inputShape->at(1);

  sd::LongType outputShape[4];

  auto newImageSize = INPUT_VARIABLE(3);
  const LongType method = block.numI() > 0 ? INT_ARG(0) : 0;
  validateCropAndResize(in, boxShape, inputShape->at(2), newImageSize, method, block.numI(), block.numT());

  outputShape[0] = shape::sizeAt(boxShape, 0);
  outputShape[1] = newImageSize->e<LongType>(0);
  outputShape[2] = newImageSize->e<LongType>(1);
  outputShape[3] = shape::sizeAt(in, 3);
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(in), shape::order(in), 4, outputShape));
}

DECLARE_TYPES(crop_and_resize) {
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_INTS})
      ->setAllowedInputTypes(3, {ALL_INTS})
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});  // storage preserves the image dtype
}
}  // namespace ops
}  // namespace sd

#endif
