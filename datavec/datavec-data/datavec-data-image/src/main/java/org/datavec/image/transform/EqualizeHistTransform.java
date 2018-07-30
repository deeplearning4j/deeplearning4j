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

package org.datavec.image.transform;

import lombok.Data;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;

/**
 * "<a href="https://opencv-srf.blogspot.com/2013/08/histogram-equalization.html">Histogram Equalization</a> equalizes the intensity distribution of an image or flattens the intensity distribution curve.
 * Used to improve the contrast of an image."
 *
 */
@JsonIgnoreProperties({"splitChannels", "converter"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class EqualizeHistTransform extends BaseImageTransform {

    /**
     * Color Conversion code
     * {@link org.bytedeco.javacpp.opencv_imgproc}
     */
    private int conversionCode;

    private MatVector splitChannels = new MatVector();

    /**
     * Default transforms histogram equalization for CV_BGR2GRAY (grayscale)
     */
    public EqualizeHistTransform() {
        this(new Random(1234), CV_BGR2GRAY);
    }

    /**
     * Return contrast normalized object
     *
     * @param conversionCode  to transform,
     */
    public EqualizeHistTransform(int conversionCode) {
        this(null, conversionCode);
    }

    /**
     * Return contrast normalized object
     *
     * @param random Random
     * @param conversionCode  to transform,
     */
    public EqualizeHistTransform(Random random, int conversionCode) {
        super(random);
        this.conversionCode = conversionCode;
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    /**
     * Takes an image and returns a transformed image.
     * Uses the random object in the case of random transformations.
     *
     * @param image  to transform, null == end of stream
     * @param random object to use (or null for deterministic)
     * @return transformed image
     */

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = (Mat) converter.convert(image.getFrame());
        Mat result = new Mat();
        try {
            if (mat.channels() == 1) {
                equalizeHist(mat, result);
            } else {
                split(mat, splitChannels);
                equalizeHist(splitChannels.get(0), splitChannels.get(0)); //equalize histogram on the 1st channel (Y)
                merge(splitChannels, result);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        return coordinates;
    }
}
