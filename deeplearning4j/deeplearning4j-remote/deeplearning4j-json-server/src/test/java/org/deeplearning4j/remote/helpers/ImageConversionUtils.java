/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.remote.helpers;

import lombok.val;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC;

public class ImageConversionUtils {

    public static Mat makeRandomImage(int height, int width, int channels) {
        if (height <= 0) {

            height = new Random().nextInt() % 100 + 100;
        }
        if (width <= 0) {
            width = new Random().nextInt() % 100 + 100;
        }

        Mat img = new Mat(height, width, CV_8UC(channels));
        UByteIndexer idx = img.createIndexer();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < channels; k++) {
                    idx.put(i, j, k, new Random().nextInt());
                }
            }
        }
        return img;
    }

    public static BufferedImage makeRandomBufferedImage(int height, int width, int channels) {
        Mat img = makeRandomImage(height, width, channels);

        OpenCVFrameConverter.ToMat c = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter c2 = new Java2DFrameConverter();

        return c2.convert(c.convert(img));
    }

    public static INDArray convert(BufferedImage image) {
        INDArray retVal = null;
        try {
            retVal = new Java2DNativeImageLoader(image.getHeight(), image.getWidth(), image.getRaster().getNumBands()).
                    asRowVector(image);
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
        return retVal;
    }

    public static INDArray convert(Mat image) {
        INDArray retVal = null;
        try {
            new NativeImageLoader().asRowVector(image);
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
        return retVal;
    }

    public static BufferedImage convert(INDArray input) {
        return new Java2DNativeImageLoader(input.rows(),input.columns()).asBufferedImage(input);
    }

    public static INDArray makeRandomImageAsINDArray(int height, int width, int channels) {
        val image = makeRandomBufferedImage(height, width, channels);
        INDArray retVal = convert(image);
        return retVal;
    }
}
