/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.util;


import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Convolutional shape utilities
 *
 * @author Adam Gibson
 */
public class ConvolutionUtils {


    /**
     * Get the height and width
     * from the configuration
     * @param conf the configuration to get height and width from
     * @return the configuration to get height and width from
     */
    public static int[] getHeightAndWidth(NeuralNetConfiguration conf) {
        return getHeightAndWidth(conf.getFilterSize());
    }


    /**
     * Retrieve the number of feature maps from the configuration
     * @param conf the configuration to get
     *             the feature maps from
     * @return the number of feature maps from
     * the filter size of the configuration
     */
    public static int numFeatureMap(NeuralNetConfiguration conf) {
        return numFeatureMap(conf.getFilterSize());
    }

    /**
     * Get the height and width
     * for an image
     * @param shape the shape of the image
     * @return the height and width for the image
     */
    public static int[] getHeightAndWidth(int[] shape) {
        if(shape.length < 2)
            throw new IllegalArgumentException("No width and height able to be found: array must be at least length 2");
        return new int[] {shape[shape.length - 1],shape[shape.length - 2]};
    }


    /**
     * Returns the number of
     * feature maps for a given shape (must be at least 3 dimensions
     * @param shape the shape to get the
     *              number of feature maps for
     * @return the number of feature maps
     * for a particular shape
     */
    public static int numFeatureMap(int[] shape) {
        return shape[0];
    }

}
