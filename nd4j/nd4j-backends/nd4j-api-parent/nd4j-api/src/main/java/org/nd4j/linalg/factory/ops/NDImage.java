/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDImage {
  public NDImage() {
  }

  /**
   * Given an input image and some crop boxes, extract out the image subsets and resize them to the specified size.<br>
   *
   * @param image Input image, with shape [batch, height, width, channels] (NUMERIC type)
   * @param cropBoxes Float32 crop, shape [numBoxes, 4] with values in range 0 to 1 (NUMERIC type)
   * @param boxIndices Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes] (NUMERIC type)
   * @param cropOutSize Output size for the images - int32, rank 1 with values [outHeight, outWidth] (INT type)
   * @param extrapolationValue Used for extrapolation, when applicable. 0.0 should be used for the default
   * @return output Cropped and resized images (NUMERIC type)
   */
  public INDArray cropAndResize(INDArray image, INDArray cropBoxes, INDArray boxIndices,
      INDArray cropOutSize, double extrapolationValue) {
    NDValidation.validateNumerical("CropAndResize", "image", image);
    NDValidation.validateNumerical("CropAndResize", "cropBoxes", cropBoxes);
    NDValidation.validateNumerical("CropAndResize", "boxIndices", boxIndices);
    NDValidation.validateInteger("CropAndResize", "cropOutSize", cropOutSize);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.image.CropAndResize(image, cropBoxes, boxIndices, cropOutSize, extrapolationValue))[0];
  }

  /**
   * Given an input image and some crop boxes, extract out the image subsets and resize them to the specified size.<br>
   *
   * @param image Input image, with shape [batch, height, width, channels] (NUMERIC type)
   * @param cropBoxes Float32 crop, shape [numBoxes, 4] with values in range 0 to 1 (NUMERIC type)
   * @param boxIndices Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes] (NUMERIC type)
   * @param cropOutSize Output size for the images - int32, rank 1 with values [outHeight, outWidth] (INT type)
   * @return output Cropped and resized images (NUMERIC type)
   */
  public INDArray cropAndResize(INDArray image, INDArray cropBoxes, INDArray boxIndices,
      INDArray cropOutSize) {
    NDValidation.validateNumerical("CropAndResize", "image", image);
    NDValidation.validateNumerical("CropAndResize", "cropBoxes", cropBoxes);
    NDValidation.validateNumerical("CropAndResize", "boxIndices", boxIndices);
    NDValidation.validateInteger("CropAndResize", "cropOutSize", cropOutSize);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.image.CropAndResize(image, cropBoxes, boxIndices, cropOutSize, 0.0))[0];
  }

  /**
   * Adjusts contrast of RGB or grayscale images.<br>
   *
   * @param in images to adjust. 3D shape or higher (NUMERIC type)
   * @param factor multiplier for adjusting contrast
   * @return output Contrast-adjusted image (NUMERIC type)
   */
  public INDArray adjustContrast(INDArray in, double factor) {
    NDValidation.validateNumerical("adjustContrast", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.AdjustContrast(in, factor))[0];
  }

  /**
   * Adjust hue of RGB image <br>
   *
   * @param in image as 3D array (NUMERIC type)
   * @param delta value to add to hue channel
   * @return output adjusted image (NUMERIC type)
   */
  public INDArray adjustHue(INDArray in, double delta) {
    NDValidation.validateNumerical("adjustHue", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.AdjustHue(in, delta))[0];
  }

  /**
   * Adjust saturation of RGB images<br>
   *
   * @param in RGB image as 3D array (NUMERIC type)
   * @param factor factor for saturation
   * @return output adjusted image (NUMERIC type)
   */
  public INDArray adjustSaturation(INDArray in, double factor) {
    NDValidation.validateNumerical("adjustSaturation", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.AdjustSaturation(in, factor))[0];
  }

  /**
   * Given an input image, extract out image patches (of size kSizes - h x w) and place them in the depth dimension. <br>
   *
   * @param image Input image to extract image patches from - shape [batch, height, width, channels] (NUMERIC type)
   * @param kSizes Kernel size - size of the image patches, [height, width] (Size: Exactly(count=2))
   * @param strides Stride in the input dimension for extracting image patches, [stride_height, stride_width] (Size: Exactly(count=2))
   * @param rates Usually [1,1]. Equivalent to dilation rate in dilated convolutions - how far apart the output pixels
   *                  in the patches should be, in the input. A dilation of [a,b] means every {@code a}th pixel is taken
   *                  along the height/rows dimension, and every {@code b}th pixel is take along the width/columns dimension (Size: AtLeast(min=0))
   * @param sameMode Padding algorithm. If true: use Same padding
   * @return output The extracted image patches (NUMERIC type)
   */
  public INDArray extractImagePatches(INDArray image, int[] kSizes, int[] strides, int[] rates,
      boolean sameMode) {
    NDValidation.validateNumerical("extractImagePatches", "image", image);
    Preconditions.checkArgument(kSizes.length == 2, "kSizes has incorrect size/length. Expected: kSizes.length == 2, got %s", kSizes.length);
    Preconditions.checkArgument(strides.length == 2, "strides has incorrect size/length. Expected: strides.length == 2, got %s", strides.length);
    Preconditions.checkArgument(rates.length >= 0, "rates has incorrect size/length. Expected: rates.length >= 0, got %s", rates.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches(image, kSizes, strides, rates, sameMode))[0];
  }

  /**
   * Converting image from HSV to RGB format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public INDArray hsvToRgb(INDArray input) {
    NDValidation.validateNumerical("hsvToRgb", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.HsvToRgb(input))[0];
  }

  /**
   * Greedily selects a subset of bounding boxes in descending order of score<br>
   *
   * @param boxes Might be null. Name for the output variable (NUMERIC type)
   * @param scores vector of shape [num_boxes] (NUMERIC type)
   * @param maxOutSize scalar representing the maximum number of boxes to be selected
   * @param iouThreshold threshold for deciding whether boxes overlap too much with respect to IOU
   * @param scoreThreshold threshold for deciding when to remove boxes based on score
   * @return output vectort of shape [M] representing the selected indices from the boxes tensor, where M <= max_output_size (NUMERIC type)
   */
  public INDArray nonMaxSuppression(INDArray boxes, INDArray scores, int maxOutSize,
      double iouThreshold, double scoreThreshold) {
    NDValidation.validateNumerical("nonMaxSuppression", "boxes", boxes);
    NDValidation.validateNumerical("nonMaxSuppression", "scores", scores);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.image.NonMaxSuppression(boxes, scores, maxOutSize, iouThreshold, scoreThreshold))[0];
  }

  /**
   * Randomly crops image<br>
   *
   * @param input input array (NUMERIC type)
   * @param shape shape for crop (NUMERIC type)
   * @return output cropped array (NUMERIC type)
   */
  public INDArray randomCrop(INDArray input, INDArray shape) {
    NDValidation.validateNumerical("randomCrop", "input", input);
    NDValidation.validateNumerical("randomCrop", "shape", shape);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.RandomCrop(input, shape))[0];
  }

  /**
   * Converting array from HSV to RGB format<br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public INDArray rgbToHsv(INDArray input) {
    NDValidation.validateNumerical("rgbToHsv", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.RgbToHsv(input))[0];
  }

  /**
   * Converting array from RGB to YIQ format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public INDArray rgbToYiq(INDArray input) {
    NDValidation.validateNumerical("rgbToYiq", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.RgbToYiq(input))[0];
  }

  /**
   * Converting array from RGB to YUV format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public INDArray rgbToYuv(INDArray input) {
    NDValidation.validateNumerical("rgbToYuv", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.RgbToYuv(input))[0];
  }

  /**
   * Converting image from YIQ to RGB format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public INDArray yiqToRgb(INDArray input) {
    NDValidation.validateNumerical("yiqToRgb", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.YiqToRgb(input))[0];
  }

  /**
   * Converting image from YUV to RGB format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public INDArray yuvToRgb(INDArray input) {
    NDValidation.validateNumerical("yuvToRgb", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.custom.YuvToRgb(input))[0];
  }
}
