/*
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

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
