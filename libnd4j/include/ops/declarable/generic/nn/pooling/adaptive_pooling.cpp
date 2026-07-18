/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_adaptive_avgpool2d)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/headers/convo.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops {

/**
 * adaptive_avgpool2d - Adaptive Average Pooling 2D
 *
 * Automatically computes kernel size and stride to produce output
 * of the specified spatial dimensions. Common in vision models.
 *
 * Inputs:
 *   0: input [batch, channels, height, width] (NCHW) or [batch, height, width, channels] (NHWC)
 *
 * Integer args:
 *   0: output_height - target output height
 *   1: output_width - target output width
 *   2: isNCHW - 1 for NCHW format, 0 for NHWC (default 1)
 *
 * Outputs:
 *   0: output [batch, channels, output_height, output_width] or [batch, output_height, output_width, channels]
 */
CUSTOM_OP_IMPL(adaptive_avgpool2d, 1, 1, false, 0, 3) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto outputH = INT_ARG(0);
  auto outputW = INT_ARG(1);
  auto isNCHW = block.numI() > 2 ? INT_ARG(2) != 0 : true;

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "adaptive_avgpool2d: input must be rank 4, got %i", input->rankOf());

  sd::LongType batchSize, channels, inputH, inputW;

  if (isNCHW) {
    batchSize = input->sizeAt(0);
    channels = input->sizeAt(1);
    inputH = input->sizeAt(2);
    inputW = input->sizeAt(3);
  } else {
    batchSize = input->sizeAt(0);
    inputH = input->sizeAt(1);
    inputW = input->sizeAt(2);
    channels = input->sizeAt(3);
  }

  // Special case: global average pooling
  if (outputH == 1 && outputW == 1) {
    if (isNCHW) {
      // Mean over H and W dimensions (2 and 3)
      std::vector<sd::LongType> dims = {2, 3};
      auto mean = input->reduceAlongDimension(reduce::Mean, &dims, true);
      output->assign(mean);
      delete mean;
    } else {
      // Mean over H and W dimensions (1 and 2)
      std::vector<sd::LongType> dims = {1, 2};
      auto mean = input->reduceAlongDimension(reduce::Mean, &dims, true);
      output->assign(mean);
      delete mean;
    }
    return sd::Status::OK;
  }

  // Calculate adaptive pooling parameters
  // stride = floor(input_size / output_size)
  // kernel = input_size - (output_size - 1) * stride
  sd::LongType strideH = inputH / outputH;
  sd::LongType strideW = inputW / outputW;
  sd::LongType kernelH = inputH - (outputH - 1) * strideH;
  sd::LongType kernelW = inputW - (outputW - 1) * strideW;

  // Use avgpool2d with calculated parameters
  sd::ops::avgpool2d avgPool;
  auto result = avgPool.evaluate({input}, {},
                                  {kernelH, kernelW, strideH, strideW, 0, 0, 1, 1, isNCHW ? 1 : 0, 0, 0});

  if (result.status() != sd::Status::OK) {
    return result.status();
  }

  output->assign(result.at(0));

  return sd::Status::OK;
}

DECLARE_TYPES(adaptive_avgpool2d) {
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(adaptive_avgpool2d) {
  auto inShape = inputShape->at(0);
  auto inputType = sd::ArrayOptions::dataType(inShape);

  auto outputH = INT_ARG(0);
  auto outputW = INT_ARG(1);
  auto isNCHW = block.numI() > 2 ? INT_ARG(2) != 0 : true;

  sd::LongType batchSize = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
  sd::LongType channels;

  std::vector<sd::LongType> outShape;
  if (isNCHW) {
    channels = shape::sizeAt(inShape, static_cast<sd::LongType>(1));
    outShape = {batchSize, channels, outputH, outputW};
  } else {
    channels = shape::sizeAt(inShape, static_cast<sd::LongType>(3));
    outShape = {batchSize, outputH, outputW, channels};
  }

  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', outShape);
  return SHAPELIST(outputShapeInfo);
}

/**
 * adaptive_avgpool2d_bp - Backward pass for adaptive average pooling 2D
 */
CUSTOM_OP_IMPL(adaptive_avgpool2d_bp, 2, 1, false, 0, 3) {
  auto input = INPUT_VARIABLE(0);
  auto gradOut = INPUT_VARIABLE(1);
  auto gradIn = OUTPUT_VARIABLE(0);

  auto outputH = INT_ARG(0);
  auto outputW = INT_ARG(1);
  auto isNCHW = block.numI() > 2 ? INT_ARG(2) != 0 : true;

  sd::LongType inputH, inputW;
  if (isNCHW) {
    inputH = input->sizeAt(2);
    inputW = input->sizeAt(3);
  } else {
    inputH = input->sizeAt(1);
    inputW = input->sizeAt(2);
  }

  // Calculate adaptive pooling parameters
  sd::LongType strideH = inputH / outputH;
  sd::LongType strideW = inputW / outputW;
  sd::LongType kernelH = inputH - (outputH - 1) * strideH;
  sd::LongType kernelW = inputW - (outputW - 1) * strideW;

  // Use avgpool2d_bp with calculated parameters
  sd::ops::avgpool2d_bp avgPoolBp;
  auto result = avgPoolBp.evaluate({input, gradOut}, {},
                                    {kernelH, kernelW, strideH, strideW, 0, 0, 1, 1, isNCHW ? 1 : 0, 0, 0});

  if (result.status() != sd::Status::OK) {
    return result.status();
  }

  gradIn->assign(result.at(0));

  return sd::Status::OK;
}

DECLARE_TYPES(adaptive_avgpool2d_bp) {
  getOpDescriptor()->addTraits((OP_TRAIT_BACKWARD));
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(adaptive_avgpool2d_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)));
}

/**
 * adaptive_maxpool2d - Adaptive Max Pooling 2D
 *
 * Same as adaptive_avgpool2d but uses max pooling instead of average.
 */
CUSTOM_OP_IMPL(adaptive_maxpool2d, 1, 1, false, 0, 3) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto outputH = INT_ARG(0);
  auto outputW = INT_ARG(1);
  auto isNCHW = block.numI() > 2 ? INT_ARG(2) != 0 : true;

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "adaptive_maxpool2d: input must be rank 4, got %i", input->rankOf());

  sd::LongType batchSize, channels, inputH, inputW;

  if (isNCHW) {
    batchSize = input->sizeAt(0);
    channels = input->sizeAt(1);
    inputH = input->sizeAt(2);
    inputW = input->sizeAt(3);
  } else {
    batchSize = input->sizeAt(0);
    inputH = input->sizeAt(1);
    inputW = input->sizeAt(2);
    channels = input->sizeAt(3);
  }

  // Special case: global max pooling
  if (outputH == 1 && outputW == 1) {
    if (isNCHW) {
      std::vector<sd::LongType> dims = {2, 3};
      auto maxVal = input->reduceAlongDimension(reduce::Max, &dims, true);
      output->assign(maxVal);
      delete maxVal;
    } else {
      std::vector<sd::LongType> dims = {1, 2};
      auto maxVal = input->reduceAlongDimension(reduce::Max, &dims, true);
      output->assign(maxVal);
      delete maxVal;
    }
    return sd::Status::OK;
  }

  // Calculate adaptive pooling parameters
  sd::LongType strideH = inputH / outputH;
  sd::LongType strideW = inputW / outputW;
  sd::LongType kernelH = inputH - (outputH - 1) * strideH;
  sd::LongType kernelW = inputW - (outputW - 1) * strideW;

  // Use maxpool2d with calculated parameters
  sd::ops::maxpool2d maxPool;
  auto result = maxPool.evaluate({input}, {},
                                  {kernelH, kernelW, strideH, strideW, 0, 0, 1, 1, isNCHW ? 1 : 0, 0, 0});

  if (result.status() != sd::Status::OK) {
    return result.status();
  }

  output->assign(result.at(0));

  return sd::Status::OK;
}

DECLARE_TYPES(adaptive_maxpool2d) {
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(adaptive_maxpool2d) {
  auto inputShapeInfo = inputShape->at(0);
  auto inputType = sd::ArrayOptions::dataType(inputShapeInfo);

  auto outputH = INT_ARG(0);
  auto outputW = INT_ARG(1);
  auto isNCHW = block.numI() > 2 ? INT_ARG(2) != 0 : true;

  sd::LongType batchSize = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType channels;

  std::vector<sd::LongType> outputShape;
  if (isNCHW) {
    channels = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(1));
    outputShape = {batchSize, channels, outputH, outputW};
  } else {
    channels = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(3));
    outputShape = {batchSize, outputH, outputW, channels};
  }

  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', outputShape);
  return SHAPELIST(outputShapeInfo);
}

/**
 * adaptive_maxpool2d_bp - Backward pass for adaptive max pooling 2D
 */
CUSTOM_OP_IMPL(adaptive_maxpool2d_bp, 2, 1, false, 0, 3) {
  auto input = INPUT_VARIABLE(0);
  auto gradOut = INPUT_VARIABLE(1);
  auto gradIn = OUTPUT_VARIABLE(0);

  auto outputH = INT_ARG(0);
  auto outputW = INT_ARG(1);
  auto isNCHW = block.numI() > 2 ? INT_ARG(2) != 0 : true;

  sd::LongType inputH, inputW;
  if (isNCHW) {
    inputH = input->sizeAt(2);
    inputW = input->sizeAt(3);
  } else {
    inputH = input->sizeAt(1);
    inputW = input->sizeAt(2);
  }

  // Calculate adaptive pooling parameters
  sd::LongType strideH = inputH / outputH;
  sd::LongType strideW = inputW / outputW;
  sd::LongType kernelH = inputH - (outputH - 1) * strideH;
  sd::LongType kernelW = inputW - (outputW - 1) * strideW;

  // Use maxpool2d_bp with calculated parameters
  sd::ops::maxpool2d_bp maxPoolBp;
  auto result = maxPoolBp.evaluate({input, gradOut}, {},
                                    {kernelH, kernelW, strideH, strideW, 0, 0, 1, 1, isNCHW ? 1 : 0, 0, 0});

  if (result.status() != sd::Status::OK) {
    return result.status();
  }

  gradIn->assign(result.at(0));

  return sd::Status::OK;
}

DECLARE_TYPES(adaptive_maxpool2d_bp) {
  getOpDescriptor()->addTraits((OP_TRAIT_BACKWARD));
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(adaptive_maxpool2d_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)));
}

/**
 * adaptive_avgpool3d - Adaptive Average Pooling 3D
 */
CUSTOM_OP_IMPL(adaptive_avgpool3d, 1, 1, false, 0, 4) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto outputD = INT_ARG(0);
  auto outputH = INT_ARG(1);
  auto outputW = INT_ARG(2);
  auto isNCDHW = block.numI() > 3 ? INT_ARG(3) != 0 : true;

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "adaptive_avgpool3d: input must be rank 5, got %i", input->rankOf());

  sd::LongType inputD, inputH, inputW;
  if (isNCDHW) {
    inputD = input->sizeAt(2);
    inputH = input->sizeAt(3);
    inputW = input->sizeAt(4);
  } else {
    inputD = input->sizeAt(1);
    inputH = input->sizeAt(2);
    inputW = input->sizeAt(3);
  }

  // Global pooling case
  if (outputD == 1 && outputH == 1 && outputW == 1) {
    if (isNCDHW) {
      std::vector<sd::LongType> dims = {2, 3, 4};
      auto mean = input->reduceAlongDimension(reduce::Mean, &dims, true);
      output->assign(mean);
      delete mean;
    } else {
      std::vector<sd::LongType> dims = {1, 2, 3};
      auto mean = input->reduceAlongDimension(reduce::Mean, &dims, true);
      output->assign(mean);
      delete mean;
    }
    return sd::Status::OK;
  }

  // Calculate adaptive pooling parameters
  sd::LongType strideD = inputD / outputD;
  sd::LongType strideH = inputH / outputH;
  sd::LongType strideW = inputW / outputW;
  sd::LongType kernelD = inputD - (outputD - 1) * strideD;
  sd::LongType kernelH = inputH - (outputH - 1) * strideH;
  sd::LongType kernelW = inputW - (outputW - 1) * strideW;

  // Use avgpool3d with calculated parameters
  sd::ops::avgpool3dnew avgPool;
  auto result = avgPool.evaluate({input}, {},
                                  {kernelD, kernelH, kernelW, strideD, strideH, strideW,
                                   0, 0, 0, 1, 1, 1, isNCDHW ? 1 : 0});

  if (result.status() != sd::Status::OK) {
    return result.status();
  }

  output->assign(result.at(0));

  return sd::Status::OK;
}

DECLARE_TYPES(adaptive_avgpool3d) {
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(adaptive_avgpool3d) {
  auto inputShapeInfo = inputShape->at(0);
  auto inputType = sd::ArrayOptions::dataType(inputShapeInfo);

  auto outputD = INT_ARG(0);
  auto outputH = INT_ARG(1);
  auto outputW = INT_ARG(2);
  auto isNCDHW = block.numI() > 3 ? INT_ARG(3) != 0 : true;

  sd::LongType batchSize = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType channels;

  std::vector<sd::LongType> outputShape;
  if (isNCDHW) {
    channels = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(1));
    outputShape = {batchSize, channels, outputD, outputH, outputW};
  } else {
    channels = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(4));
    outputShape = {batchSize, outputD, outputH, outputW, channels};
  }

  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(inputType, 'c', outputShape);
  return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
