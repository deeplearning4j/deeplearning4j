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
#if NOT_EXCLUDED(OP_image_resize)
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_resize.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(image_resize, 2, 1, false, -2, -2) {
  auto image = INPUT_VARIABLE(0);
  auto size = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  int width;
  int height;
  bool antialias = false;
  REQUIRE_TRUE(size->lengthOf() == 2, 0, "image_resize: Resize params is a pair of values, not %lld.",
               size->lengthOf());
  width = size->e<int>(1);
  height = size->e<int>(0);
  if (block.numB() >= 2) {
    antialias = B_ARG(1);
  }
  bool exclude_outside = true;
  double bicubicCoefficient = helpers::KeysCubicKernelFunc<double>::KEYS_CUBIC_COEF;
  auto method = helpers::ImageResizeMethods::kResizeBilinear;
  helpers::CoordinateTransformationMode coorMode = helpers::CoordinateTransformationMode::HALF_PIXEL;

  if (block.numB() >= 3) {
    exclude_outside = B_ARG(2);
  }
  if (block.numT() > 0) {
    bicubicCoefficient = T_ARG(0);
  }
  if (block.numI() >= 1) {
    method = (helpers::ImageResizeMethods)INT_ARG(0);
  }
  if (block.numI() >= 2) {
    coorMode = static_cast<helpers::CoordinateTransformationMode>(INT_ARG(1));
  } else if (method == helpers::ImageResizeMethods::kResizeNearest) {
    // retain old behavour
    coorMode = helpers::CoordinateTransformationMode::HALF_PIXEL_NN;
  }
  helpers::NearestMode nearestMode = helpers::NearestMode::FLOOR;
  if (method == helpers::ImageResizeMethods::kResizeNearest && block.numI() == 3) {
    nearestMode = static_cast<helpers::NearestMode>(INT_ARG(2));
    REQUIRE_TRUE(nearestMode >= helpers::NearestMode::FLOOR && nearestMode <= helpers::NearestMode::CEIL, 0,
                 "image_resize: nearest Mode should be between %i and %i, but %i was given.",
                 (int)helpers::NearestMode::FLOOR, (int)helpers::NearestMode::CEIL, (int)nearestMode);
  }
  REQUIRE_TRUE(method == helpers::ImageResizeMethods::kResizeNearest || output->dataType() == DataType::FLOAT32, 0,
               "image_resize: Output data type should be FLOAT32 for this method %i", (int)method);
  REQUIRE_TRUE(
      method >= helpers::ImageResizeMethods::kResizeFirst && method <= helpers::ImageResizeMethods::kResizeLast, 0,
      "image_resize: Resize method should be between %i and %i, but %i was given.",
      (int)helpers::ImageResizeMethods::kResizeFirst, (int)helpers::ImageResizeMethods::kResizeLast, (int)method);
  auto inRank = image->rankOf();
  REQUIRE_TRUE(inRank >= 3 && inRank <= 4, 0, "image_resize: Input rank should be 4 or 3, but %i given.",
               image->rankOf());
  std::vector<sd::LongType> imageShape1 = {image->sizeAt(0), image->sizeAt(1), image->sizeAt(2),image->sizeAt(3)};
  std::vector<sd::LongType> imageShape2 = {1, image->sizeAt(0), image->sizeAt(1), image->sizeAt(2)};
  auto source =
      inRank == 4
          ? image->reshape(image->ordering(), imageShape1)
          : image->reshape(image->ordering(),imageShape2);

  std::vector<sd::LongType> outputShape1 = {output->sizeAt(0), output->sizeAt(1), output->sizeAt(2),output->sizeAt(3)};
  std::vector<sd::LongType> outputShape2 = {1, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2)};

  auto target =
      inRank == 4
          ? output->reshape(output->ordering(),
                            outputShape1, false)
          : output->reshape(output->ordering(), outputShape2, false);

  // inform the user about the current state of the implementation
  if (antialias && method != helpers::ImageResizeMethods::kResizeNearest) {
    REQUIRE_TRUE(coorMode == helpers::CoordinateTransformationMode::HALF_PIXEL && exclude_outside, 0,
                 "antialiasing is effective only with HALF_PIXEL and exclude_outside being set true");
  }
  //
  if ((method != helpers::ImageResizeMethods::kResizeBicubic &&
       method != helpers::ImageResizeMethods::kResizeNearest)) {
    REQUIRE_TRUE(coorMode == helpers::CoordinateTransformationMode::HALF_PIXEL && exclude_outside, 0,
                 "this method supports only HALF_PIXEL and exclude_outside being set true");
  }

  auto ret =  resizeFunctor(block.launchContext(), image, width, height, method, coorMode, exclude_outside,
                                nearestMode, bicubicCoefficient, antialias, output);
  delete target;
  return ret;
}

DECLARE_SHAPE_FN(image_resize) {
  auto in = inputShape->at(0);

  LongType* outputShape;
  auto method = helpers::ImageResizeMethods::kResizeBilinear;
  if (block.numI() >= 1) {
    method = (helpers::ImageResizeMethods)INT_ARG(0);
  }

  int width;
  int height;
  double ratio = shape::sizeAt(in, static_cast<LongType>(1)) / (0.0 + shape::sizeAt(in, static_cast<LongType>(2)));
  auto newImageSize = INPUT_VARIABLE(1);
  REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_bilinear: Resize params is a pair of values, not %i.",
               newImageSize->lengthOf());

  width = newImageSize->e<int>(1);
  height = newImageSize->e<int>(0);
  if (block.numB() > 0) {
    if (B_ARG(0)) {
      width = math::sd_ceil<double, int>(height / ratio);
    }
  }
  auto dtype = FLOAT32;
  if (method == helpers::ImageResizeMethods::kResizeNearest) dtype = ArrayOptions::dataType(in);
  auto shape = ConstantShapeHelper::getInstance().createShapeInfo(
      dtype, 'c',
      shape::rank(in) == 4 ? std::vector<LongType>{in[1], height, width, in[4]}
                           : std::vector<LongType>{height, width, in[4]});

  return SHAPELIST(shape);
}
DECLARE_TYPES(image_resize) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
}

}  // namespace ops
}  // namespace sd

#endif
