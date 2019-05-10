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

package org.datavec.image.loader;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Segregates functionality specific to Java 2D that is not available on Android.
 *
 * @author saudet
 */
public class Java2DNativeImageLoader extends NativeImageLoader {

    Java2DFrameConverter converter2 = new Java2DFrameConverter();

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

    /**
     * Returns {@code asMatrix(image, false).ravel()}.
     */
    public INDArray asRowVector(BufferedImage image) throws IOException {
        return asMatrix(image, false).ravel();
    }

    /**
     * Returns {@code asMatrix(image, false)}.
     */
    public INDArray asMatrix(BufferedImage image) throws IOException {
        return asMatrix(image, false);
    }

    /**
     * Returns {@code asMatrix(image, flipChannels).ravel()}.
     */
    public INDArray asRowVector(BufferedImage image, boolean flipChannels) throws IOException {
        return asMatrix(image, flipChannels).ravel();
    }

    /**
     * Loads a {@link INDArray} from a {@link BufferedImage}.
     *
     * @param image as a BufferedImage
     * @param flipChannels to have a format like TYPE_INT_RGB (ARGB) output as BGRA, etc
     * @return the loaded matrix
     * @throws IOException
     */
    public INDArray asMatrix(BufferedImage image, boolean flipChannels) throws IOException {
        if (converter == null) {
            converter = new OpenCVFrameConverter.ToMat();
        }
        return asMatrix(converter.convert(converter2.getFrame(image, 1.0, flipChannels)));
    }

    @Override
    public INDArray asRowVector(Object image) throws IOException {
        return image instanceof BufferedImage ? asRowVector((BufferedImage) image) : null;
    }

    @Override
    public INDArray asMatrix(Object image) throws IOException {
        return image instanceof BufferedImage ? asMatrix((BufferedImage) image) : null;
    }

    /** Returns {@code asBufferedImage(array, Frame.DEPTH_UBYTE)}. */
    public BufferedImage asBufferedImage(INDArray array) {
        return asBufferedImage(array, Frame.DEPTH_UBYTE);
    }

    /**
     * Converts an INDArray to a BufferedImage. Only intended for images with rank 3.
     *
     * @param array to convert
     * @param dataType from JavaCV (DEPTH_FLOAT, DEPTH_UBYTE, etc), or -1 to use same type as the INDArray
     * @return data copied to a Frame
     */
    public BufferedImage asBufferedImage(INDArray array, int dataType) {
        return converter2.convert(asFrame(array, dataType));
    }
}
