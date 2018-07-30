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

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Rect;

/**
 * Crops images deterministically or randomly.
 *
 * @author saudet
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class CropImageTransform extends BaseImageTransform<Mat> {

    private int cropTop;
    private int cropLeft;
    private int cropBottom;
    private int cropRight;

    private int x;
    private int y;

    /** Calls {@code this(null, crop, crop, crop, crop)}. */
    public CropImageTransform(int crop) {
        this(null, crop, crop, crop, crop);
    }

    /** Calls {@code this(random, crop, crop, crop, crop)}. */
    public CropImageTransform(Random random, int crop) {
        this(random, crop, crop, crop, crop);
    }

    /** Calls {@code this(random, cropTop, cropLeft, cropBottom, cropRight)}. */
    public CropImageTransform(@JsonProperty("cropTop") int cropTop, @JsonProperty("cropLeft") int cropLeft,
                    @JsonProperty("cropBottom") int cropBottom, @JsonProperty("cropRight") int cropRight) {
        this(null, cropTop, cropLeft, cropBottom, cropRight);
    }

    /**
     * Constructs an instance of the ImageTransform.
     *
     * @param random     object to use (or null for deterministic)
     * @param cropTop    maximum cropping of the top of the image (pixels)
     * @param cropLeft   maximum cropping of the left of the image (pixels)
     * @param cropBottom maximum cropping of the bottom of the image (pixels)
     * @param cropRight  maximum cropping of the right of the image (pixels)
     */
    public CropImageTransform(Random random, int cropTop, int cropLeft, int cropBottom, int cropRight) {
        super(random);
        this.cropTop = cropTop;
        this.cropLeft = cropLeft;
        this.cropBottom = cropBottom;
        this.cropRight = cropRight;
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
        int top = random != null ? random.nextInt(cropTop + 1) : cropTop;
        int left = random != null ? random.nextInt(cropLeft + 1) : cropLeft;
        int bottom = random != null ? random.nextInt(cropBottom + 1) : cropBottom;
        int right = random != null ? random.nextInt(cropRight + 1) : cropRight;

        y = Math.min(top, mat.rows() - 1);
        x = Math.min(left, mat.cols() - 1);
        int h = Math.max(1, mat.rows() - bottom - y);
        int w = Math.max(1, mat.cols() - right - x);
        Mat result = mat.apply(new Rect(x, y, w, h));

        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        float[] transformed = new float[coordinates.length];
        for (int i = 0; i < coordinates.length; i += 2) {
            transformed[i    ] = coordinates[i    ] - x;
            transformed[i + 1] = coordinates[i + 1] - y;
        }
        return transformed;
    }
}
