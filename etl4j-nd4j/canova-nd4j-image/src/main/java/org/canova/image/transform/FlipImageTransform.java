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

/**
 * Flips images deterministically or randomly.
 *
 * @author saudet
 */
public class FlipImageTransform extends BaseImageTransform<Mat> {

    int flipMode;

    /** Calls {@code this(null)}. */
    public FlipImageTransform() {
        this(null);
    }

    /**
     * Calls {@code this(null)} and sets the flip mode.
     *
     * @param flipMode the deterministic flip mode
     *                 {@code  0} Flips around x-axis.
     *                 {@code >0} Flips around y-axis.
     *                 {@code <0} Flips around both axes.
     */
    public FlipImageTransform(int flipMode) {
        this(null);
        this.flipMode = flipMode;
    }

    /**
     * Constructs an instance of the ImageTransform. Randomly does not flip,
     * or flips horizontally or vertically, or both.
     *
     * @param random object to use (or null for deterministic)
     */
    public FlipImageTransform(Random random) {
        super(random);
        converter = new OpenCVFrameConverter.ToMat();
    }

    @Override
    public ImageWritable transform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());

        int mode = random != null ? random.nextInt(4) - 2 : flipMode;

        Mat result = new Mat();
        if (mode < -1) {
            // no flip
            mat.copyTo(result);
        } else {
            flip(mat, result, mode);
        }

        return new ImageWritable(converter.convert(result));
    }
}
