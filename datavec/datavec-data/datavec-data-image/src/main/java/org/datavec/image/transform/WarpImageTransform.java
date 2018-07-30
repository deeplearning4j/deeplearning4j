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
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.nio.FloatBuffer;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Warps the perspective of images deterministically or randomly. Calls
 * {@link org.bytedeco.javacpp.opencv_imgproc#warpPerspective(Mat, Mat, Mat, Size, int, int, Scalar)}
 * with given properties (interMode, borderMode, and borderValue).
 *
 * @author saudet
 */
@Accessors(fluent = true)
@JsonIgnoreProperties({"interMode", "borderMode", "borderValue", "converter"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class WarpImageTransform extends BaseImageTransform<Mat> {

    private float[] deltas;

    @Getter
    @Setter
    int interMode = INTER_LINEAR;
    @Getter
    @Setter
    int borderMode = BORDER_CONSTANT;
    @Getter
    @Setter
    Scalar borderValue = Scalar.ZERO;

    private Mat M;

    /** Calls {@code this(null, delta, delta, delta, delta, delta, delta, delta, delta)}. */
    public WarpImageTransform(float delta) {
        this(null, delta, delta, delta, delta, delta, delta, delta, delta);
    }

    /** Calls {@code this(random, delta, delta, delta, delta, delta, delta, delta, delta)}. */
    public WarpImageTransform(Random random, float delta) {
        this(random, delta, delta, delta, delta, delta, delta, delta, delta);
    }

    /** Calls {@code this(null, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4)}. */
    public WarpImageTransform(@JsonProperty("deltas[0]") float dx1, @JsonProperty("deltas[1]") float dy1,
                    @JsonProperty("deltas[2]") float dx2, @JsonProperty("deltas[3]") float dy2,
                    @JsonProperty("deltas[4]") float dx3, @JsonProperty("deltas[5]") float dy3,
                    @JsonProperty("deltas[6]") float dx4, @JsonProperty("deltas[7]") float dy4) {
        this(null, dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4);
    }

    /**
     * Constructs an instance of the ImageTransform.
     *
     * @param random object to use (or null for deterministic)
     * @param dx1    maximum warping in x for the top-left corner (pixels)
     * @param dy1    maximum warping in y for the top-left corner (pixels)
     * @param dx2    maximum warping in x for the top-right corner (pixels)
     * @param dy2    maximum warping in y for the top-right corner (pixels)
     * @param dx3    maximum warping in x for the bottom-right corner (pixels)
     * @param dy3    maximum warping in y for the bottom-right corner (pixels)
     * @param dx4    maximum warping in x for the bottom-left corner (pixels)
     * @param dy4    maximum warping in y for the bottom-left corner (pixels)
     */
    public WarpImageTransform(Random random, float dx1, float dy1, float dx2, float dy2, float dx3, float dy3,
                    float dx4, float dy4) {
        super(random);
        deltas = new float[8];
        deltas[0] = dx1;
        deltas[1] = dy1;
        deltas[2] = dx2;
        deltas[3] = dy2;
        deltas[4] = dx3;
        deltas[5] = dy3;
        deltas[6] = dx4;
        deltas[7] = dy4;
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
        Point2f src = new Point2f(4);
        Point2f dst = new Point2f(4);
        src.put(0, 0, mat.cols(), 0, mat.cols(), mat.rows(), 0, mat.rows());

        for (int i = 0; i < 8; i++) {
            dst.put(i, src.get(i) + deltas[i] * (random != null ? 2 * random.nextFloat() - 1 : 1));
        }
        Mat result = new Mat();
        M = getPerspectiveTransform(src, dst);
        warpPerspective(mat, result, M, mat.size(), interMode, borderMode, borderValue);

        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        Mat src = new Mat(1, coordinates.length / 2, CV_32FC2, new FloatPointer(coordinates));
        Mat dst = new Mat();
        perspectiveTransform(src, dst, M);
        FloatBuffer buf = dst.createBuffer();
        float[] transformed = new float[coordinates.length];
        buf.get(transformed);
        return transformed;
    }
}
