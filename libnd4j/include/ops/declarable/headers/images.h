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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
// 
//
// @author AbdelRauf    (rauf@konduit.ai)
//

#ifndef LIBND4J_HEADERS_IMAGES_H
#define LIBND4J_HEADERS_IMAGES_H

#include <ops/declarable/headers/common.h>
#include <ops/declarable/CustomOperations.h>  
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>
#include <ops/declarable/helpers/imagesHelpers.h>

namespace sd {
namespace ops {


/**
 * Rgb To Hsv
 * Input arrays:
 * 0 - input array with rank >= 1, must have at least one dimension equal 3, that is dimension containing channels.
 * Int arguments:
 * 0 - optional argument, corresponds to dimension with 3 channels
 */
#if NOT_EXCLUDED(OP_rgb_to_hsv)
    DECLARE_CONFIGURABLE_OP(rgb_to_hsv, 1, 1, true, 0, 0);
#endif

/**
 * Hsv To Rgb
 * Input arrays:
 * 0 - input array with rank >= 1, must have at least one dimension equal 3, that is dimension containing channels.
 * Int arguments:
 * 0 - optional argument, corresponds to dimension with 3 channels
 */
#if NOT_EXCLUDED(OP_hsv_to_rgb)
    DECLARE_CONFIGURABLE_OP(hsv_to_rgb, 1, 1, true, 0, 0);
#endif

/**
* Rgb To GrayScale
* Input arrays:
* 0 - input array with rank >= 1, the RGB tensor to convert. Last dimension must have size 3 and should contain RGB values.
*/
#if NOT_EXCLUDED(OP_rgb_to_grs)
    DECLARE_CUSTOM_OP(rgb_to_grs, 1, 1, false, 0, 0);
#endif

    /**
     * Rgb To Yuv
     * Input arrays:
     * 0 - input array with rank >= 1, must have at least one dimension equal 3, that is dimension containing channels.
     * Int arguments:
     * 0 - optional argument, corresponds to dimension with 3 channels
     */
#if NOT_EXCLUDED(OP_rgb_to_yuv)
    DECLARE_CONFIGURABLE_OP(rgb_to_yuv, 1, 1, true, 0, 0);
#endif

    /**
     * Yuv To Rgb
     * Input arrays:
     * 0 - input array with rank >= 1, must have at least one dimension equal 3, that is dimension containing channels.
     * Int arguments:
     * 0 - optional argument, corresponds to dimension with 3 channels
     */
#if NOT_EXCLUDED(OP_rgb_to_yuv)
    DECLARE_CONFIGURABLE_OP(yuv_to_rgb, 1, 1, true, 0, 0);
#endif

/**
* Rgb To Yiq
* Input arrays:
* 0 - input array with rank >= 1, must have at least one dimension equal 3, that is dimension containing channels.
* Int arguments:
* 0 - optional argument, corresponds to dimension with 3 channels
*/
#if NOT_EXCLUDED(OP_rgb_to_yiq)
    DECLARE_CONFIGURABLE_OP(rgb_to_yiq, 1, 1, true, 0, 0);
#endif

/**
* Yiq To Rgb
* Input arrays:
* 0 - input array with rank >= 1, must have at least one dimension equal 3, that is dimension containing channels.
* Int arguments:
* 0 - optional argument, corresponds to dimension with 3 channels
*/
#if NOT_EXCLUDED(OP_yiq_to_rgb)
    DECLARE_CONFIGURABLE_OP(yiq_to_rgb, 1, 1, true, 0, 0);
#endif

/**
 * resize_images - resize image with given size and method
 *    there are 4 methods allowed: RESIZE_BILINEAR(0), RESIZE_NEIGHBOR(1), RESIZE_AREA(2) and RESIZE_BICUBIC(3)
 * inputs:
 *      0 - 4D tensor with shape {batch, height, width, channels}
 *      1 - 1D integer tensor with {new_height, new_width} (optional)
 *      2 - 0D integer tensor with method (0 to 3) (optional)
 *
 * int args:
 *      0 - new_height
 *      1 - new_width
 *      2 - method
 *
 * bool args:
 *      0 - align corners (default false) - optional
 *      1 - preserve_aspect_ratio (default false) - optional
 *
 * CAUTION: one of methods can be used to give size and method - as tensors or as int args only
 *
 * output:
 *      0 - 4D float32 tensor with shape {batch, new_height, new_width, channels}
 *
 */
#if NOT_EXCLUDED(OP_resize_images)
    DECLARE_CUSTOM_OP(resize_images, 1,1,false, 0, 0);
#endif

   /**
    * This op make bilinear or nearest neighbor interpolated resize for given tensor
    *
    * input array:
    *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels) numeric type
    *    1 - 2D-Tensor with shape (num_boxes, 4) float type
    *    2 - 1D-Tensor with shape (num_boxes) int type
    *    3 - 1D-Tensor with 2 values (newWidth, newHeight) (optional) int type
    *
    * float arguments (optional)
    *   0 - exprapolation_value (optional) default 0.f
    *
    * int arguments: (optional)
    *   0 - mode (default 0 - bilinear interpolation)
    *
    * output array:
    *   the 4D-Tensor with resized to crop_size images given - float type
    */
    #if NOT_EXCLUDED(OP_crop_and_resize)
    DECLARE_CUSTOM_OP(crop_and_resize, 4, 1, false, -1, -1);
    #endif

   /**
    * This op make bilinear interpolated resize for given tensor
    *
    * input array:
    *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
    *    1 - 1D-Tensor with 2 values (newWidth, newHeight) (optional)
    *
    * int arguments: (optional)
    *   0 - new width
    *   1 - new height
    *
    * output array:
    *   the 4D-Tensor with calculated backproped dots
    *
    * CAUTION: either size tensor or a pair of int params should be provided.
    */

    #if NOT_EXCLUDED(OP_resize_bilinear)
    DECLARE_CUSTOM_OP(resize_bilinear, 1, 1, false, 0, -2);
    #endif

   /**
    * This op make nearest neighbor interpolated resize for given tensor
    *
    * input array:
    *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
    *    1 - 1D-Tensor with 2 values (newWidth, newHeight) (optional)
    *
    * int arguments: (optional)
    *   0 - new width
    *   1 - new height
    *
    * output array:
    *   the 4D-Tensor with resized image (shape is {batch, newWidth, newHeight, channels})
    *
    * CAUTION: either size tensor or a pair of int params should be provided.
    */

    #if NOT_EXCLUDED(OP_resize_nearest_neighbor)
    DECLARE_CUSTOM_OP(resize_nearest_neighbor, 1, 1, false, 0, -2);
    #endif

   /**
    * This op make bicubic interpolated resize for given tensor
    *
    * input array:
    *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
    *    1 - 1D-Tensor with 2 values (newWidth, newHeight)
    *
    * output array:
    *   the 4D-Tensor with resized image (shape is {batch, newWidth, newHeight, channels})
    *
    */
    #if NOT_EXCLUDED(OP_resize_bicubic)
    DECLARE_CUSTOM_OP(resize_bicubic, 1, 1, false, 0, -2);
    #endif

   /**
    * This op make area interpolated resize (as OpenCV INTER_AREA algorithm) for given tensor
    *
    * input array:
    *    0 - images - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
    *    1 - size -   1D-Tensor with 2 values (newWidth, newHeight) (if missing a pair of integer args should be provided).
    *
    * int args: - proveded only when size tensor is missing
    *    0 - new height
    *    1 - new width
    * boolean args:
    *    0 - align_corners - optional (default is false)
    *
    * output array:
    *   the 4D-Tensor with resized image (shape is {batch, newWidth, newHeight, channels})
    *
    */
    #if NOT_EXCLUDED(OP_resize_area)
    DECLARE_CUSTOM_OP(resize_area, 1, 1, false, 0, -2);
    #endif

   /**
    * This op make interpolated resize for given tensor with given algorithm.
    * Supported algorithms are bilinear, bicubic, nearest_neighbor, lanczos5, gaussian, area and mitchellcubic.
    *
    * input array:
    *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
    *    1 - 1D-Tensor with 2 values (newWidth, newHeight)
    *
    * optional int args:
    *    0 - algorithm - bilinear by default
    * optional bool args:
    *    0 - preserve_aspect_ratio - default False
    *    1 - antialias - default False
    *
    * output array:
    *   the 4D-Tensor with resized by given algorithm image (shape is {batch, newWidth, newHeight, channels})
    *
    */

    #if NOT_EXCLUDED(OP_image_resize)
    DECLARE_CUSTOM_OP(image_resize, 2, 1, false, 0, 0);
    #endif

}
}
#endif
