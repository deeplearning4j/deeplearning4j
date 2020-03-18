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

}
}

#endif
#endif
