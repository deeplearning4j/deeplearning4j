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

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2Luv;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

/**
 * Color conversion transform using CVT (cvtcolor):
 * <a href="http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor">CVT Color</a>.
 * <a href="http://bytedeco.org/javacpp-presets/opencv/apidocs/org/bytedeco/javacpp/opencv_imgproc.html#cvtColor-org.bytedeco.javacpp.opencv_core.Mat-org.bytedeco.javacpp.opencv_core.Mat-int-int-">More CVT Color</a>.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class ColorConversionTransform extends BaseImageTransform {

    /**
     * Color Conversion code
     * {@link org.bytedeco.javacpp.opencv_imgproc}
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
