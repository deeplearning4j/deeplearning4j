/*
 *
 *  *
 *  *  * Copyright 2016 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */
package org.canova.image.transform;

import java.util.Random;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.canova.image.data.ImageWritable;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Scales images deterministically or randomly.
 *
 * @author saudet
 */
public class ScaleImageTransform extends BaseImageTransform<Mat> {

    float dx, dy;

    /** Calls {@code this(null, delta, delta)}. */
    public ScaleImageTransform(float delta) {
        this(null, delta, delta);
    }

    /** Calls {@code this(random, delta, delta)}. */
    public ScaleImageTransform(Random random, float delta) {
        this(random, delta, delta);
    }

    /** Calls {@code this(null, dx, dy)}. */
    public ScaleImageTransform(float dx, float dy) {
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

        converter = new OpenCVFrameConverter.ToMat();
    }

    @Override
    public ImageWritable transform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());

        int h = Math.round(mat.rows() + dy * (random != null ? 2 * random.nextFloat() - 1 : 1));
        int w = Math.round(mat.cols() + dx * (random != null ? 2 * random.nextFloat() - 1 : 1));

        Mat result = new Mat();
        resize(mat, result, new Size(w, h));

        return new ImageWritable(converter.convert(result));
    }
}
