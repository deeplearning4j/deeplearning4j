/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.datavec.image.util;

public class ImageUtils {

    /**
     * Calculate coordinates in an image, assuming the image has been scaled from (oH x oW) pixels to (nH x nW) pixels
     *
     * @param x          X position (pixels) to translate
     * @param y          Y position (pixels) to translate
     * @param origImageW Original image width (pixels)
     * @param origImageH Original image height (pixels)
     * @param newImageW  New image width (pixels)
     * @param newImageH  New image height (pixels)
     * @return  New X and Y coordinates (pixels, in new image)
     */
    public static double[] translateCoordsScaleImage(double x, double y, double origImageW, double origImageH, double newImageW, double newImageH){

        double newX = x * newImageW / origImageW;
        double newY = y * newImageH / origImageH;

        return new double[]{newX, newY};
    }

}
