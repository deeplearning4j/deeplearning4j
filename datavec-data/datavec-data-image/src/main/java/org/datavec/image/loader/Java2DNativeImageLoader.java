/*-
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
package org.datavec.image.loader;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.FrameConverter;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Segregates functionality specific to Java 2D that is not available on Android.
 *
 * @author saudet
 */
public class Java2DNativeImageLoader extends NativeImageLoader {

    protected Map<Long, Java2DFrameConverter> java2dConverter = new HashMap<>();

    public Java2DNativeImageLoader() {}

    public Java2DNativeImageLoader(int height, int width) {
        super(height, width);
    }

    public Java2DNativeImageLoader(int height, int width, int channels) {
        super(height, width, channels);
    }

    public Java2DNativeImageLoader(int height, int width, int channels, boolean centerCropIfNeeded) {
        super(height, width, channels, centerCropIfNeeded);
    }

    public Java2DNativeImageLoader(int height, int width, int channels, ImageTransform imageTransform) {
        super(height, width, channels, imageTransform);
    }

    protected Java2DNativeImageLoader(NativeImageLoader other) {
        super(other);
    }

    public INDArray asRowVector(BufferedImage image) throws IOException {
        return asMatrix(image).ravel();
    }

    public INDArray asMatrix(BufferedImage image) throws IOException {
        FrameConverter<opencv_core.Mat> converter = getSafeConverter(Thread.currentThread().getId());
        Java2DFrameConverter converter2 = getSafeJava2DFrameConverter(Thread.currentThread().getId());

        return asMatrix(converter.convert(converter2.convert(image)));
    }

    @Override
    public INDArray asRowVector(Object image) throws IOException {
        return image instanceof BufferedImage ? asRowVector((BufferedImage)image) : null;
    }

    @Override
    public INDArray asMatrix(Object image) throws IOException {
        return image instanceof BufferedImage ? asMatrix((BufferedImage)image) : null;
    }

    /**
     * Returns thread-safe Java2DFrameConverter
     *
     * @param threadId
     * @return
     */
    protected Java2DFrameConverter getSafeJava2DFrameConverter(long threadId) {
        if (java2dConverter.containsKey(threadId)) {
            return java2dConverter.get(Thread.currentThread().getId());
        }else {
            Java2DFrameConverter converter = new Java2DFrameConverter();
            java2dConverter.put(threadId, converter);
            return converter;
        }
    }

}
