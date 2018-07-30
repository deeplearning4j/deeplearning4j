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
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * ResizeImageTransform is suited to force the <b>same image size</b> for whole pipeline
 * and it doesn't use any random factor for width and height.
 *
 * If you need to use random scales to scale or crop the images,
 * these links might be helpful {@link ScaleImageTransform} or {@link ScaleImageTransform}
 *
 * @author raver119@gmail.com
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class ResizeImageTransform extends BaseImageTransform<Mat> {

    private int newHeight;
    private int newWidth;

    private int srch;
    private int srcw;

    /**
     * Returns new ResizeImageTransform object
     *
     * @param newWidth new Width for the outcome images
     * @param newHeight new Height for outcome images
     */
    public ResizeImageTransform(@JsonProperty("newWidth") int newWidth, @JsonProperty("newHeight") int newHeight) {
        this(null, newWidth, newHeight);
    }

    /**
     * Returns new ResizeImageTransform object
     *
     * @param random Random
     * @param newWidth new Width for the outcome images
     * @param newHeight new Height for outcome images
     */
    public ResizeImageTransform(Random random, int newWidth, int newHeight) {
        super(random);

        this.newWidth = newWidth;
        this.newHeight = newHeight;
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
        Mat mat = converter.convert(image.getFrame());
        Mat result = new Mat();
        srch = mat.rows();
        srcw = mat.cols();
        resize(mat, result, new Size(newWidth, newHeight));
        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        float[] transformed = new float[coordinates.length];
        for (int i = 0; i < coordinates.length; i += 2) {
            transformed[i    ] = newWidth * coordinates[i    ] / srcw;
            transformed[i + 1] = newHeight * coordinates[i + 1] / srch;
        }
        return transformed;
    }
}
