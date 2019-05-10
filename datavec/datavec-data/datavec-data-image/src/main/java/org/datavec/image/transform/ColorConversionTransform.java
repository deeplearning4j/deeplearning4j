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
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.util.Random;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Color conversion transform using CVT (cvtcolor):
 * <a href="https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html">CVT Color</a>.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class ColorConversionTransform extends BaseImageTransform {

    /**
     * Color Conversion code
     * {@link org.bytedeco.opencv.global.opencv_imgproc}
     */
    private int conversionCode;

    /**
     * Default conversion BGR to Luv (chroma) color.
     */
    public ColorConversionTransform() {
        this(new Random(1234), COLOR_BGR2Luv);
    }

    /**
     * Return new ColorConversion object
     *
     * @param conversionCode  to transform,
     */
    public ColorConversionTransform(int conversionCode) {
        this(null, conversionCode);
    }

    /**
     * Return new ColorConversion object
     *
     * @param random Random
     * @param conversionCode  to transform,
     */
    public ColorConversionTransform(Random random, int conversionCode) {
        super(random);
        this.conversionCode = conversionCode;
        converter = new OpenCVFrameConverter.ToMat();
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
            cvtColor(mat, result, conversionCode);
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
