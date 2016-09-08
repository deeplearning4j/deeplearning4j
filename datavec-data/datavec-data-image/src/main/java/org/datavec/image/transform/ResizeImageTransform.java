/*
 *  * Copyright 2016 Skymind, Inc.
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

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Resize image transform is suited to force the same image size for whole pipeline.
 *
 * @author raver119@gmail.com
 */
public class ResizeImageTransform extends BaseImageTransform<opencv_core.Mat>  {

    int newHeight, newWidth;

    /**
     * Returns new ResizeImageTransform object
     *
     * @param newWidth new Width for the outcome images
     * @param newHeight new Height for outcome images
     */
    public ResizeImageTransform(int newWidth, int newHeight) {
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
    public ImageWritable transform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        opencv_core.Mat mat = converter.convert(image.getFrame());
        opencv_core.Mat result = new opencv_core.Mat();
        resize(mat, result, new opencv_core.Size(newWidth, newHeight));
        return new ImageWritable(converter.convert(result));
    }


}
