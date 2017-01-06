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
package org.datavec.image.loader;

import java.awt.image.BufferedImage;
import java.util.Random;

import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.bytedeco.javacpp.opencv_core.*;

/**
 *
 * @author saudet
 */
public class TestNativeImageLoader {
    static final long seed = 10;
    static final Random rng = new Random(seed);

    @Test
    public void testAsRowVector() throws Exception {
        BufferedImage img1 = makeRandomBufferedImage(0, 0, 1);
        Mat img2 = makeRandomImage(0, 0, 3);

        int w1 = 35, h1 = 79, ch1 = 3;
        NativeImageLoader loader1 = new NativeImageLoader(h1, w1, ch1);

        INDArray array1 = loader1.asRowVector(img1);
        assertEquals(2, array1.rank());
        assertEquals(1, array1.rows());
        assertEquals(h1 * w1 * ch1, array1.columns());

        INDArray array2 = loader1.asRowVector(img2);
        assertEquals(2, array2.rank());
        assertEquals(1, array2.rows());
        assertEquals(h1 * w1 * ch1, array2.columns());

        int w2 = 103, h2 = 68, ch2 = 4;
        NativeImageLoader loader2 = new NativeImageLoader(h2, w2, ch2);

        INDArray array3 = loader2.asRowVector(img1);
        assertEquals(2, array3.rank());
        assertEquals(1, array3.rows());
        assertEquals(h2 * w2 * ch2, array3.columns());

        INDArray array4 = loader2.asRowVector(img2);
        assertEquals(2, array4.rank());
        assertEquals(1, array4.rows());
        assertEquals(h2 * w2 * ch2, array4.columns());
    }

    @Test
    public void testAsMatrix() throws Exception {
        BufferedImage img1 = makeRandomBufferedImage(0, 0, 3);
        Mat img2 = makeRandomImage(0, 0, 4);

        int w1 = 33, h1 = 77, ch1 = 1;
        NativeImageLoader loader1 = new NativeImageLoader(h1, w1, ch1);

        INDArray array1 = loader1.asMatrix(img1);
        assertEquals(4, array1.rank());
        assertEquals(1, array1.size(0));
        assertEquals(1, array1.size(1));
        assertEquals(h1, array1.size(2));
        assertEquals(w1, array1.size(3));

        INDArray array2 = loader1.asMatrix(img2);
        assertEquals(4, array2.rank());
        assertEquals(1, array2.size(0));
        assertEquals(1, array2.size(1));
        assertEquals(h1, array2.size(2));
        assertEquals(w1, array2.size(3));

        int w2 = 111, h2 = 66, ch2 = 3;
        NativeImageLoader loader2 = new NativeImageLoader(h2, w2, ch2);

        INDArray array3 = loader2.asMatrix(img1);
        assertEquals(4, array3.rank());
        assertEquals(1, array3.size(0));
        assertEquals(3, array3.size(1));
        assertEquals(h2, array3.size(2));
        assertEquals(w2, array3.size(3));

        INDArray array4 = loader2.asMatrix(img2);
        assertEquals(4, array4.rank());
        assertEquals(1, array4.size(0));
        assertEquals(3, array4.size(1));
        assertEquals(h2, array4.size(2));
        assertEquals(w2, array4.size(3));
    }

    @Test
    public void testScalingIfNeed() throws Exception {
        Mat img1 = makeRandomImage(0, 0, 1);
        Mat img2 = makeRandomImage(0, 0, 3);

        int w1 = 60, h1 = 110, ch1 = 1;
        NativeImageLoader loader1 = new NativeImageLoader(h1, w1, ch1);

        Mat scaled1 = loader1.scalingIfNeed(img1);
        assertEquals(h1, scaled1.rows());
        assertEquals(w1, scaled1.cols());
        assertEquals(img1.channels(), scaled1.channels());

        Mat scaled2 = loader1.scalingIfNeed(img2);
        assertEquals(h1, scaled2.rows());
        assertEquals(w1, scaled2.cols());
        assertEquals(img2.channels(), scaled2.channels());

        int w2 = 70, h2 = 120, ch2 = 3;
        NativeImageLoader loader2 = new NativeImageLoader(h2, w2, ch2);

        Mat scaled3 = loader2.scalingIfNeed(img1);
        assertEquals(h2, scaled3.rows());
        assertEquals(w2, scaled3.cols());
        assertEquals(img1.channels(), scaled3.channels());

        Mat scaled4 = loader2.scalingIfNeed(img2);
        assertEquals(h2, scaled4.rows());
        assertEquals(w2, scaled4.cols());
        assertEquals(img2.channels(), scaled4.channels());
    }

    @Test
    public void testCenterCropIfNeeded() throws Exception {
        int w1 = 60, h1 = 110, ch1 = 1;
        int w2 = 120, h2 = 70, ch2 = 3;

        Mat img1 = makeRandomImage(h1, w1, ch1);
        Mat img2 = makeRandomImage(h2, w2, ch2);

        NativeImageLoader loader = new NativeImageLoader(h1, w1, ch1, true);

        Mat cropped1 = loader.centerCropIfNeeded(img1);
        assertEquals(85, cropped1.rows());
        assertEquals(60, cropped1.cols());
        assertEquals(img1.channels(), cropped1.channels());

        Mat cropped2 = loader.centerCropIfNeeded(img2);
        assertEquals(70, cropped2.rows());
        assertEquals(95, cropped2.cols());
        assertEquals(img2.channels(), cropped2.channels());
    }


    BufferedImage makeRandomBufferedImage(int height, int width, int channels) {
        Mat img = makeRandomImage(height, width, channels);

        OpenCVFrameConverter.ToMat c = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter c2 = new Java2DFrameConverter();

        return c2.convert(c.convert(img));
    }

    Mat makeRandomImage(int height, int width, int channels) {
        if (height <= 0) {
            height = rng.nextInt() % 100 + 100;
        }
        if (width <= 0) {
            width = rng.nextInt() % 100 + 100;
        }

        Mat img = new Mat(height, width, CV_8UC(channels));
        UByteIndexer idx = img.createIndexer();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < channels; k++) {
                    idx.put(i, j, k, rng.nextInt());
                }
            }
        }
        return img;
    }
}
