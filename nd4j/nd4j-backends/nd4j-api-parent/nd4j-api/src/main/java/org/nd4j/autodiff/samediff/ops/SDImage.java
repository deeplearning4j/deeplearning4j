/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.enums.ImageResizeMethod;

public class SDImage extends SDOps {
  public SDImage(SameDiff sameDiff) {
    super(sameDiff);
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
  public SDVariable cropAndResize(SDVariable image, SDVariable cropBoxes, SDVariable boxIndices,
      SDVariable cropOutSize, double extrapolationValue) {
    SDValidation.validateNumerical("CropAndResize", "image", image);
    SDValidation.validateNumerical("CropAndResize", "cropBoxes", cropBoxes);
    SDValidation.validateNumerical("CropAndResize", "boxIndices", boxIndices);
    SDValidation.validateInteger("CropAndResize", "cropOutSize", cropOutSize);
    return new org.nd4j.linalg.api.ops.impl.image.CropAndResize(sd,image, cropBoxes, boxIndices, cropOutSize, extrapolationValue).outputVariable();
  }

  /**
   * Given an input image and some crop boxes, extract out the image subsets and resize them to the specified size.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param image Input image, with shape [batch, height, width, channels] (NUMERIC type)
   * @param cropBoxes Float32 crop, shape [numBoxes, 4] with values in range 0 to 1 (NUMERIC type)
   * @param boxIndices Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes] (NUMERIC type)
   * @param cropOutSize Output size for the images - int32, rank 1 with values [outHeight, outWidth] (INT type)
   * @param extrapolationValue Used for extrapolation, when applicable. 0.0 should be used for the default
   * @return output Cropped and resized images (NUMERIC type)
   */
  public SDVariable cropAndResize(String name, SDVariable image, SDVariable cropBoxes,
      SDVariable boxIndices, SDVariable cropOutSize, double extrapolationValue) {
    SDValidation.validateNumerical("CropAndResize", "image", image);
    SDValidation.validateNumerical("CropAndResize", "cropBoxes", cropBoxes);
    SDValidation.validateNumerical("CropAndResize", "boxIndices", boxIndices);
    SDValidation.validateInteger("CropAndResize", "cropOutSize", cropOutSize);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.image.CropAndResize(sd,image, cropBoxes, boxIndices, cropOutSize, extrapolationValue).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable cropAndResize(SDVariable image, SDVariable cropBoxes, SDVariable boxIndices,
      SDVariable cropOutSize) {
    SDValidation.validateNumerical("CropAndResize", "image", image);
    SDValidation.validateNumerical("CropAndResize", "cropBoxes", cropBoxes);
    SDValidation.validateNumerical("CropAndResize", "boxIndices", boxIndices);
    SDValidation.validateInteger("CropAndResize", "cropOutSize", cropOutSize);
    return new org.nd4j.linalg.api.ops.impl.image.CropAndResize(sd,image, cropBoxes, boxIndices, cropOutSize, 0.0).outputVariable();
  }

  /**
   * Given an input image and some crop boxes, extract out the image subsets and resize them to the specified size.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param image Input image, with shape [batch, height, width, channels] (NUMERIC type)
   * @param cropBoxes Float32 crop, shape [numBoxes, 4] with values in range 0 to 1 (NUMERIC type)
   * @param boxIndices Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes] (NUMERIC type)
   * @param cropOutSize Output size for the images - int32, rank 1 with values [outHeight, outWidth] (INT type)
   * @return output Cropped and resized images (NUMERIC type)
   */
  public SDVariable cropAndResize(String name, SDVariable image, SDVariable cropBoxes,
      SDVariable boxIndices, SDVariable cropOutSize) {
    SDValidation.validateNumerical("CropAndResize", "image", image);
    SDValidation.validateNumerical("CropAndResize", "cropBoxes", cropBoxes);
    SDValidation.validateNumerical("CropAndResize", "boxIndices", boxIndices);
    SDValidation.validateInteger("CropAndResize", "cropOutSize", cropOutSize);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.image.CropAndResize(sd,image, cropBoxes, boxIndices, cropOutSize, 0.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Adjusts contrast of RGB or grayscale images.<br>
   *
   * @param in images to adjust. 3D shape or higher (NUMERIC type)
   * @param factor multiplier for adjusting contrast
   * @return output Contrast-adjusted image (NUMERIC type)
   */
  public SDVariable adjustContrast(SDVariable in, double factor) {
    SDValidation.validateNumerical("adjustContrast", "in", in);
    return new org.nd4j.linalg.api.ops.custom.AdjustContrast(sd,in, factor).outputVariable();
  }

  /**
   * Adjusts contrast of RGB or grayscale images.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in images to adjust. 3D shape or higher (NUMERIC type)
   * @param factor multiplier for adjusting contrast
   * @return output Contrast-adjusted image (NUMERIC type)
   */
  public SDVariable adjustContrast(String name, SDVariable in, double factor) {
    SDValidation.validateNumerical("adjustContrast", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.AdjustContrast(sd,in, factor).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Adjust hue of RGB image <br>
   *
   * @param in image as 3D array (NUMERIC type)
   * @param delta value to add to hue channel
   * @return output adjusted image (NUMERIC type)
   */
  public SDVariable adjustHue(SDVariable in, double delta) {
    SDValidation.validateNumerical("adjustHue", "in", in);
    return new org.nd4j.linalg.api.ops.custom.AdjustHue(sd,in, delta).outputVariable();
  }

  /**
   * Adjust hue of RGB image <br>
   *
   * @param name name May be null. Name for the output variable
   * @param in image as 3D array (NUMERIC type)
   * @param delta value to add to hue channel
   * @return output adjusted image (NUMERIC type)
   */
  public SDVariable adjustHue(String name, SDVariable in, double delta) {
    SDValidation.validateNumerical("adjustHue", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.AdjustHue(sd,in, delta).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Adjust saturation of RGB images<br>
   *
   * @param in RGB image as 3D array (NUMERIC type)
   * @param factor factor for saturation
   * @return output adjusted image (NUMERIC type)
   */
  public SDVariable adjustSaturation(SDVariable in, double factor) {
    SDValidation.validateNumerical("adjustSaturation", "in", in);
    return new org.nd4j.linalg.api.ops.custom.AdjustSaturation(sd,in, factor).outputVariable();
  }

  /**
   * Adjust saturation of RGB images<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in RGB image as 3D array (NUMERIC type)
   * @param factor factor for saturation
   * @return output adjusted image (NUMERIC type)
   */
  public SDVariable adjustSaturation(String name, SDVariable in, double factor) {
    SDValidation.validateNumerical("adjustSaturation", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.AdjustSaturation(sd,in, factor).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable extractImagePatches(SDVariable image, int[] kSizes, int[] strides, int[] rates,
      boolean sameMode) {
    SDValidation.validateNumerical("extractImagePatches", "image", image);
    Preconditions.checkArgument(kSizes.length == 2, "kSizes has incorrect size/length. Expected: kSizes.length == 2, got %s", kSizes.length);
    Preconditions.checkArgument(strides.length == 2, "strides has incorrect size/length. Expected: strides.length == 2, got %s", strides.length);
    Preconditions.checkArgument(rates.length >= 0, "rates has incorrect size/length. Expected: rates.length >= 0, got %s", rates.length);
    return new org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches(sd,image, kSizes, strides, rates, sameMode).outputVariable();
  }

  /**
   * Given an input image, extract out image patches (of size kSizes - h x w) and place them in the depth dimension. <br>
   *
   * @param name name May be null. Name for the output variable
   * @param image Input image to extract image patches from - shape [batch, height, width, channels] (NUMERIC type)
   * @param kSizes Kernel size - size of the image patches, [height, width] (Size: Exactly(count=2))
   * @param strides Stride in the input dimension for extracting image patches, [stride_height, stride_width] (Size: Exactly(count=2))
   * @param rates Usually [1,1]. Equivalent to dilation rate in dilated convolutions - how far apart the output pixels
   *                  in the patches should be, in the input. A dilation of [a,b] means every {@code a}th pixel is taken
   *                  along the height/rows dimension, and every {@code b}th pixel is take along the width/columns dimension (Size: AtLeast(min=0))
   * @param sameMode Padding algorithm. If true: use Same padding
   * @return output The extracted image patches (NUMERIC type)
   */
  public SDVariable extractImagePatches(String name, SDVariable image, int[] kSizes, int[] strides,
      int[] rates, boolean sameMode) {
    SDValidation.validateNumerical("extractImagePatches", "image", image);
    Preconditions.checkArgument(kSizes.length == 2, "kSizes has incorrect size/length. Expected: kSizes.length == 2, got %s", kSizes.length);
    Preconditions.checkArgument(strides.length == 2, "strides has incorrect size/length. Expected: strides.length == 2, got %s", strides.length);
    Preconditions.checkArgument(rates.length >= 0, "rates has incorrect size/length. Expected: rates.length >= 0, got %s", rates.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches(sd,image, kSizes, strides, rates, sameMode).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Converting image from HSV to RGB format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable hsvToRgb(SDVariable input) {
    SDValidation.validateNumerical("hsvToRgb", "input", input);
    return new org.nd4j.linalg.api.ops.custom.HsvToRgb(sd,input).outputVariable();
  }

  /**
   * Converting image from HSV to RGB format <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable hsvToRgb(String name, SDVariable input) {
    SDValidation.validateNumerical("hsvToRgb", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.HsvToRgb(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Resize images to size using the specified method.<br>
   *
   * @param input 4D image [NCHW] (NUMERIC type)
   * @param size new height and width (INT type)
   * @param preserveAspectRatio Whether to preserve the aspect ratio. If this is set, then images will be resized to a size that fits in size while preserving the aspect ratio of the original image. Scales up the image if size is bigger than the current size of the image. Defaults to False.
   * @param antialis Whether to use an anti-aliasing filter when downsampling an image
   * @param ImageResizeMethod ResizeBilinear: Bilinear interpolation. If 'antialias' is true, becomes a hat/tent filter function with radius 1 when downsampling.
   * ResizeLanczos5: Lanczos kernel with radius 5. Very-high-quality filter but may have stronger ringing.
   * ResizeBicubic: Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel, particularly when upsampling.
   * ResizeGaussian: Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
   * ResizeNearest: Nearest neighbor interpolation. 'antialias' has no effect when used with nearest neighbor interpolation.
   * ResizeArea: Anti-aliased resampling with area interpolation. 'antialias' has no effect when used with area interpolation; it always anti-aliases.
   * ResizeMitchelcubic: Mitchell-Netravali Cubic non-interpolating filter. For synthetic images (especially those lacking proper prefiltering), less ringing than Keys cubic kernel but less sharp.
   * @return output Output image (NUMERIC type)
   */
  public SDVariable imageResize(SDVariable input, SDVariable size, boolean preserveAspectRatio,
      boolean antialis, ImageResizeMethod ImageResizeMethod) {
    SDValidation.validateNumerical("imageResize", "input", input);
    SDValidation.validateInteger("imageResize", "size", size);
    return new org.nd4j.linalg.api.ops.impl.image.ImageResize(sd,input, size, preserveAspectRatio, antialis, ImageResizeMethod).outputVariable();
  }

  /**
   * Resize images to size using the specified method.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 4D image [NCHW] (NUMERIC type)
   * @param size new height and width (INT type)
   * @param preserveAspectRatio Whether to preserve the aspect ratio. If this is set, then images will be resized to a size that fits in size while preserving the aspect ratio of the original image. Scales up the image if size is bigger than the current size of the image. Defaults to False.
   * @param antialis Whether to use an anti-aliasing filter when downsampling an image
   * @param ImageResizeMethod ResizeBilinear: Bilinear interpolation. If 'antialias' is true, becomes a hat/tent filter function with radius 1 when downsampling.
   * ResizeLanczos5: Lanczos kernel with radius 5. Very-high-quality filter but may have stronger ringing.
   * ResizeBicubic: Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel, particularly when upsampling.
   * ResizeGaussian: Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
   * ResizeNearest: Nearest neighbor interpolation. 'antialias' has no effect when used with nearest neighbor interpolation.
   * ResizeArea: Anti-aliased resampling with area interpolation. 'antialias' has no effect when used with area interpolation; it always anti-aliases.
   * ResizeMitchelcubic: Mitchell-Netravali Cubic non-interpolating filter. For synthetic images (especially those lacking proper prefiltering), less ringing than Keys cubic kernel but less sharp.
   * @return output Output image (NUMERIC type)
   */
  public SDVariable imageResize(String name, SDVariable input, SDVariable size,
      boolean preserveAspectRatio, boolean antialis, ImageResizeMethod ImageResizeMethod) {
    SDValidation.validateNumerical("imageResize", "input", input);
    SDValidation.validateInteger("imageResize", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.image.ImageResize(sd,input, size, preserveAspectRatio, antialis, ImageResizeMethod).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Resize images to size using the specified method.<br>
   *
   * @param input 4D image [NCHW] (NUMERIC type)
   * @param size new height and width (INT type)
   * @param ImageResizeMethod ResizeBilinear: Bilinear interpolation. If 'antialias' is true, becomes a hat/tent filter function with radius 1 when downsampling.
   * ResizeLanczos5: Lanczos kernel with radius 5. Very-high-quality filter but may have stronger ringing.
   * ResizeBicubic: Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel, particularly when upsampling.
   * ResizeGaussian: Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
   * ResizeNearest: Nearest neighbor interpolation. 'antialias' has no effect when used with nearest neighbor interpolation.
   * ResizeArea: Anti-aliased resampling with area interpolation. 'antialias' has no effect when used with area interpolation; it always anti-aliases.
   * ResizeMitchelcubic: Mitchell-Netravali Cubic non-interpolating filter. For synthetic images (especially those lacking proper prefiltering), less ringing than Keys cubic kernel but less sharp.
   * @return output Output image (NUMERIC type)
   */
  public SDVariable imageResize(SDVariable input, SDVariable size,
      ImageResizeMethod ImageResizeMethod) {
    SDValidation.validateNumerical("imageResize", "input", input);
    SDValidation.validateInteger("imageResize", "size", size);
    return new org.nd4j.linalg.api.ops.impl.image.ImageResize(sd,input, size, false, false, ImageResizeMethod).outputVariable();
  }

  /**
   * Resize images to size using the specified method.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 4D image [NCHW] (NUMERIC type)
   * @param size new height and width (INT type)
   * @param ImageResizeMethod ResizeBilinear: Bilinear interpolation. If 'antialias' is true, becomes a hat/tent filter function with radius 1 when downsampling.
   * ResizeLanczos5: Lanczos kernel with radius 5. Very-high-quality filter but may have stronger ringing.
   * ResizeBicubic: Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel, particularly when upsampling.
   * ResizeGaussian: Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
   * ResizeNearest: Nearest neighbor interpolation. 'antialias' has no effect when used with nearest neighbor interpolation.
   * ResizeArea: Anti-aliased resampling with area interpolation. 'antialias' has no effect when used with area interpolation; it always anti-aliases.
   * ResizeMitchelcubic: Mitchell-Netravali Cubic non-interpolating filter. For synthetic images (especially those lacking proper prefiltering), less ringing than Keys cubic kernel but less sharp.
   * @return output Output image (NUMERIC type)
   */
  public SDVariable imageResize(String name, SDVariable input, SDVariable size,
      ImageResizeMethod ImageResizeMethod) {
    SDValidation.validateNumerical("imageResize", "input", input);
    SDValidation.validateInteger("imageResize", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.image.ImageResize(sd,input, size, false, false, ImageResizeMethod).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable nonMaxSuppression(SDVariable boxes, SDVariable scores, int maxOutSize,
      double iouThreshold, double scoreThreshold) {
    SDValidation.validateNumerical("nonMaxSuppression", "boxes", boxes);
    SDValidation.validateNumerical("nonMaxSuppression", "scores", scores);
    return new org.nd4j.linalg.api.ops.impl.image.NonMaxSuppression(sd,boxes, scores, maxOutSize, iouThreshold, scoreThreshold).outputVariable();
  }

  /**
   * Greedily selects a subset of bounding boxes in descending order of score<br>
   *
   * @param name name May be null. Name for the output variable
   * @param boxes Might be null. Name for the output variable (NUMERIC type)
   * @param scores vector of shape [num_boxes] (NUMERIC type)
   * @param maxOutSize scalar representing the maximum number of boxes to be selected
   * @param iouThreshold threshold for deciding whether boxes overlap too much with respect to IOU
   * @param scoreThreshold threshold for deciding when to remove boxes based on score
   * @return output vectort of shape [M] representing the selected indices from the boxes tensor, where M <= max_output_size (NUMERIC type)
   */
  public SDVariable nonMaxSuppression(String name, SDVariable boxes, SDVariable scores,
      int maxOutSize, double iouThreshold, double scoreThreshold) {
    SDValidation.validateNumerical("nonMaxSuppression", "boxes", boxes);
    SDValidation.validateNumerical("nonMaxSuppression", "scores", scores);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.image.NonMaxSuppression(sd,boxes, scores, maxOutSize, iouThreshold, scoreThreshold).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Randomly crops image<br>
   *
   * @param input input array (NUMERIC type)
   * @param shape shape for crop (INT type)
   * @return output cropped array (NUMERIC type)
   */
  public SDVariable randomCrop(SDVariable input, SDVariable shape) {
    SDValidation.validateNumerical("randomCrop", "input", input);
    SDValidation.validateInteger("randomCrop", "shape", shape);
    return new org.nd4j.linalg.api.ops.custom.RandomCrop(sd,input, shape).outputVariable();
  }

  /**
   * Randomly crops image<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input input array (NUMERIC type)
   * @param shape shape for crop (INT type)
   * @return output cropped array (NUMERIC type)
   */
  public SDVariable randomCrop(String name, SDVariable input, SDVariable shape) {
    SDValidation.validateNumerical("randomCrop", "input", input);
    SDValidation.validateInteger("randomCrop", "shape", shape);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.RandomCrop(sd,input, shape).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Converting array from HSV to RGB format<br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable rgbToHsv(SDVariable input) {
    SDValidation.validateNumerical("rgbToHsv", "input", input);
    return new org.nd4j.linalg.api.ops.custom.RgbToHsv(sd,input).outputVariable();
  }

  /**
   * Converting array from HSV to RGB format<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable rgbToHsv(String name, SDVariable input) {
    SDValidation.validateNumerical("rgbToHsv", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.RgbToHsv(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Converting array from RGB to YIQ format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable rgbToYiq(SDVariable input) {
    SDValidation.validateNumerical("rgbToYiq", "input", input);
    return new org.nd4j.linalg.api.ops.custom.RgbToYiq(sd,input).outputVariable();
  }

  /**
   * Converting array from RGB to YIQ format <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable rgbToYiq(String name, SDVariable input) {
    SDValidation.validateNumerical("rgbToYiq", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.RgbToYiq(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Converting array from RGB to YUV format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable rgbToYuv(SDVariable input) {
    SDValidation.validateNumerical("rgbToYuv", "input", input);
    return new org.nd4j.linalg.api.ops.custom.RgbToYuv(sd,input).outputVariable();
  }

  /**
   * Converting array from RGB to YUV format <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable rgbToYuv(String name, SDVariable input) {
    SDValidation.validateNumerical("rgbToYuv", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.RgbToYuv(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Converting image from YIQ to RGB format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable yiqToRgb(SDVariable input) {
    SDValidation.validateNumerical("yiqToRgb", "input", input);
    return new org.nd4j.linalg.api.ops.custom.YiqToRgb(sd,input).outputVariable();
  }

  /**
   * Converting image from YIQ to RGB format <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable yiqToRgb(String name, SDVariable input) {
    SDValidation.validateNumerical("yiqToRgb", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.YiqToRgb(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Converting image from YUV to RGB format <br>
   *
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable yuvToRgb(SDVariable input) {
    SDValidation.validateNumerical("yuvToRgb", "input", input);
    return new org.nd4j.linalg.api.ops.custom.YuvToRgb(sd,input).outputVariable();
  }

  /**
   * Converting image from YUV to RGB format <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 3D image (NUMERIC type)
   * @return output 3D image (NUMERIC type)
   */
  public SDVariable yuvToRgb(String name, SDVariable input) {
    SDValidation.validateNumerical("yuvToRgb", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.YuvToRgb(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
