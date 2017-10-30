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

import android.graphics.Bitmap;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

/**
 * Segregates functionality specific to Android that is not available on Java SE.
 *
 * @author saudet
 */
public class AndroidNativeImageLoader extends NativeImageLoader {

    AndroidFrameConverter converter2 = new AndroidFrameConverter();

    public AndroidNativeImageLoader() {}

    public AndroidNativeImageLoader(int height, int width) {
        super(height, width);
    }

    public AndroidNativeImageLoader(int height, int width, int channels) {
        super(height, width, channels);
    }

    public AndroidNativeImageLoader(int height, int width, int channels, boolean centerCropIfNeeded) {
        super(height, width, channels, centerCropIfNeeded);
    }

    public AndroidNativeImageLoader(int height, int width, int channels, ImageTransform imageTransform) {
        super(height, width, channels, imageTransform);
    }

    protected AndroidNativeImageLoader(NativeImageLoader other) {
        super(other);
    }

    public INDArray asRowVector(Bitmap image) throws IOException {
        return asMatrix(image).ravel();
    }

    public INDArray asMatrix(Bitmap image) throws IOException {
        if (converter == null) {
            converter = new OpenCVFrameConverter.ToMat();
        }
        return asMatrix(converter.convert(converter2.convert(image)));
    }

    @Override
    public INDArray asRowVector(Object image) throws IOException {
        return image instanceof Bitmap ? asRowVector((Bitmap) image) : null;
    }

    @Override
    public INDArray asMatrix(Object image) throws IOException {
        return image instanceof Bitmap ? asMatrix((Bitmap) image) : null;
    }

    /** Returns {@code asBitmap(array, Frame.DEPTH_UBYTE)}. */
    public Bitmap asBitmap(INDArray array) {
        return asBitmap(array, Frame.DEPTH_UBYTE);
    }

    /**
     * Converts an INDArray to a Bitmap. Only intended for images with rank 3.
     *
     * @param array to convert
     * @param dataType from JavaCV (DEPTH_FLOAT, DEPTH_UBYTE, etc), or -1 to use same type as the INDArray
     * @return data copied to a Frame
     */
    public Bitmap asBitmap(INDArray array, int dataType) {
        return converter2.convert(asFrame(array, dataType));
    }
}
