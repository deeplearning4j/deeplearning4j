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
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.*;

/**
 * Boxes images to a given width and height without changing their aspect ratios,
 * or the size of the objects, by either padding or cropping them from the center.
 * When the width/height of an image is less than the given width/height, it gets
 * padded in that dimension, otherwise it gets cropped.
 *
 * @author saudet
 */
@Accessors(fluent = true)
@JsonIgnoreProperties({"borderValue"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class BoxImageTransform extends BaseImageTransform<Mat> {

    private int width;
    private int height;

    private int x;
    private int y;

    @Getter
    @Setter
    Scalar borderValue = Scalar.ZERO;

    /** Calls {@code this(null, width, height)}. */
    public BoxImageTransform(@JsonProperty("width") int width, @JsonProperty("height") int height) {
        this(null, width, height);
    }

    /**
     * Constructs an instance of the ImageTransform.
     *
     * @param random object to use (or null for deterministic)
     * @param width  of the boxed image (pixels)
     * @param height of the boxed image (pixels)
     */
    public BoxImageTransform(Random random, int width, int height) {
        super(random);
        this.width = width;
        this.height = height;
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    /**
     * Takes an image and returns a boxed version of the image.
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
        Mat box = new Mat(height, width, mat.type());
        box.put(borderValue);
        x = (mat.cols() - width) / 2;
        y = (mat.rows() - height) / 2;
        int w = Math.min(mat.cols(), width);
        int h = Math.min(mat.rows(), height);
        Rect matRect = new Rect(x, y, w, h);
        Rect boxRect = new Rect(x, y, w, h);

        if (x <= 0) {
            matRect.x(0);
            boxRect.x(-x);
        } else {
            matRect.x(x);
            boxRect.x(0);
        }

        if (y <= 0) {
            matRect.y(0);
            boxRect.y(-y);
        } else {
            matRect.y(y);
            boxRect.y(0);
        }
        mat.apply(matRect).copyTo(box.apply(boxRect));
        return new ImageWritable(converter.convert(box));
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
