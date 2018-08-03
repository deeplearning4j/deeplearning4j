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
import static org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * ScaleImageTransform is aim to scale by a certain <b>random factor</b>,
 * <b>not</b> the <b>final size</b> of the image.
 *
 * @author saudet
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class ScaleImageTransform extends BaseImageTransform<Mat> {

    private float dx;
    private float dy;

    private int srch, h;
    private int srcw, w;

    /** Calls {@code this(null, delta, delta)}. */
    public ScaleImageTransform(float delta) {
        this(null, delta, delta);
    }

    /** Calls {@code this(random, delta, delta)}. */
    public ScaleImageTransform(Random random, float delta) {
        this(random, delta, delta);
    }

    /** Calls {@code this(null, dx, dy)}. */
    public ScaleImageTransform(@JsonProperty("dx") float dx, @JsonProperty("dy") float dy) {
        this(null, dx, dy);
    }

    /**
     * Constructs an instance of the ImageTransform.
     *
     * @param random object to use (or null for deterministic)
     * @param dx     maximum scaling in width of the image (pixels)
     * @param dy     maximum scaling in height of the image (pixels)
     */
    public ScaleImageTransform(Random random, float dx, float dy) {
        super(random);
        this.dx = dx;
        this.dy = dy;
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());
        srch = mat.rows();
        srcw = mat.cols();
        h = Math.round(mat.rows() + dy * (random != null ? 2 * random.nextFloat() - 1 : 1));
        w = Math.round(mat.cols() + dx * (random != null ? 2 * random.nextFloat() - 1 : 1));

        Mat result = new Mat();
        resize(mat, result, new Size(w, h));
        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        float[] transformed = new float[coordinates.length];
        for (int i = 0; i < coordinates.length; i += 2) {
            transformed[i    ] = w * coordinates[i    ] / srcw;
            transformed[i + 1] = h * coordinates[i + 1] / srch;
        }
        return transformed;
    }
}
