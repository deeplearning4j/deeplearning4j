/*
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */
package org.datavec.image.transform;

import java.util.Random;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Rotates and scales images deterministically or randomly. Calls
 * {@link org.bytedeco.javacpp.opencv_imgproc#warpAffine(Mat, Mat, Mat, Size, int, int, Scalar)}
 * with given properties (interMode, borderMode, and borderValue).
 *
 * @author saudet
 */
@Accessors(fluent = true)
public class RotateImageTransform extends BaseImageTransform<Mat> {

    float centerx, centery, angle, scale;
    @Getter @Setter int interMode = INTER_LINEAR;
    @Getter @Setter int borderMode = BORDER_CONSTANT;
    @Getter @Setter Scalar borderValue = Scalar.ZERO;

    /** Calls {@code this(null, 0, 0, angle, 0)}. */
    public RotateImageTransform(float angle) {
        this(null, 0, 0, angle, 0);
    }

    /** Calls {@code this(random, 0, 0, angle, 0)}. */
    public RotateImageTransform(Random random, float angle) {
        this(random, 0, 0, angle, 0);
    }

    /**
     * Constructs an instance of the ImageTransform.
     *
     * @param random  object to use (or null for deterministic)
     * @param centerx maximum deviation in x of center of rotation (relative to image center)
     * @param centery maximum deviation in y of center of rotation (relative to image center)
     * @param angle   maximum rotation (degrees)
     * @param scale   maximum scaling (relative to 1)
     */
    public RotateImageTransform(Random random, float centerx, float centery, float angle, float scale) {
        super(random);
        this.centerx = centerx;
        this.centery = centery;
        this.angle = angle;
        this.scale = scale;

        converter = new OpenCVFrameConverter.ToMat();
    }

    @Override
    public ImageWritable transform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        Mat mat = converter.convert(image.getFrame());
        float cy = mat.rows() / 2 + centery * (random != null ? 2 * random.nextFloat() - 1 : 1);
        float cx = mat.cols() / 2 + centerx * (random != null ? 2 * random.nextFloat() - 1 : 1);
        float a  =     angle * (random != null ? 2 * random.nextFloat() - 1 : 1);
        float s  = 1 + scale * (random != null ? 2 * random.nextFloat() - 1 : 1);

        Mat result = new Mat();
        Mat M = getRotationMatrix2D(new Point2f(cx, cy), angle, scale);
        warpAffine(mat, result, M, mat.size(), interMode, borderMode, borderValue);
        return new ImageWritable(converter.convert(result));
    }

}
