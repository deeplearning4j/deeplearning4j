/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.enums;

/**
 * ResizeBilinear: Bilinear interpolation. If 'antialias' is true, becomes a hat/tent filter function with radius 1 when downsampling.
 * ResizeLanczos5: Lanczos kernel with radius 5. Very-high-quality filter but may have stronger ringing.
 * ResizeBicubic: Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel, particularly when upsampling.
 * ResizeGaussian: Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
 * ResizeNearest: Nearest neighbor interpolation. 'antialias' has no effect when used with nearest neighbor interpolation.
 * ResizeArea: Anti-aliased resampling with area interpolation. 'antialias' has no effect when used with area interpolation; it always anti-aliases.
 * ResizeMitchelcubic: Mitchell-Netravali Cubic non-interpolating filter. For synthetic images (especially those lacking proper prefiltering), less ringing than Keys cubic kernel but less sharp. */
public enum ImageResizeMethod {
  ResizeBilinear, // as java require
  ResizeNearest,
  ResizeBicubic,
  ResizeArea,
  ResizeGaussian,
  ResizeLanczos3,
  ResizeLanczos5,
  ResizeMitchellcubic;
}
