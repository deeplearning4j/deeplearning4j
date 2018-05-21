/*-
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

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.util.Random;

import static org.junit.Assert.assertEquals;


public class TestImageLoader {

    private static long seed = 10;
    private static Random rng = new Random(seed);

    @Test
    public void testToIntArrayArray() throws Exception {
        BufferedImage img = makeRandomBufferedImage(true);

        int w = img.getWidth();
        int h = img.getHeight();
        int ch = 4;
        ImageLoader loader = new ImageLoader(0, 0, ch);
        int[][] arr = loader.toIntArrayArray(img);

        assertEquals(h, arr.length);
        assertEquals(w, arr[0].length);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                assertEquals(img.getRGB(j, i), arr[i][j]);
            }
        }
    }

    @Test
    public void testToINDArrayBGR() throws Exception {
        BufferedImage img = makeRandomBufferedImage(false);
        int w = img.getWidth();
        int h = img.getHeight();
        int ch = 3;

        ImageLoader loader = new ImageLoader(0, 0, ch);
        INDArray arr = loader.toINDArrayBGR(img);

        long[] shape = arr.shape();
        assertEquals(3, shape.length);
        assertEquals(ch, shape[0]);
        assertEquals(h, shape[1]);
        assertEquals(w, shape[2]);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int srcColor = img.getRGB(j, i);
                int a = 0xff << 24;
                int r = arr.getInt(2, i, j) << 16;
                int g = arr.getInt(1, i, j) << 8;
                int b = arr.getInt(0, i, j) & 0xff;
                int dstColor = a | r | g | b;
                assertEquals(srcColor, dstColor);
            }
        }
    }

    @Test
    public void testScalingIfNeed() throws Exception {
        BufferedImage img1 = makeRandomBufferedImage(true);
        BufferedImage img2 = makeRandomBufferedImage(false);

        int w1 = 60, h1 = 110, ch1 = 6;
        ImageLoader loader1 = new ImageLoader(h1, w1, ch1);

        BufferedImage scaled1 = loader1.scalingIfNeed(img1, true);
        assertEquals(w1, scaled1.getWidth());
        assertEquals(h1, scaled1.getHeight());
        assertEquals(BufferedImage.TYPE_4BYTE_ABGR, scaled1.getType());
        assertEquals(4, scaled1.getSampleModel().getNumBands());

        BufferedImage scaled2 = loader1.scalingIfNeed(img1, false);
        assertEquals(w1, scaled2.getWidth());
        assertEquals(h1, scaled2.getHeight());
        assertEquals(BufferedImage.TYPE_3BYTE_BGR, scaled2.getType());
        assertEquals(3, scaled2.getSampleModel().getNumBands());

        BufferedImage scaled3 = loader1.scalingIfNeed(img2, true);
        assertEquals(w1, scaled3.getWidth());
        assertEquals(h1, scaled3.getHeight());
        assertEquals(BufferedImage.TYPE_3BYTE_BGR, scaled3.getType());
        assertEquals(3, scaled3.getSampleModel().getNumBands());

        BufferedImage scaled4 = loader1.scalingIfNeed(img2, false);
        assertEquals(w1, scaled4.getWidth());
        assertEquals(h1, scaled4.getHeight());
        assertEquals(BufferedImage.TYPE_3BYTE_BGR, scaled4.getType());
        assertEquals(3, scaled4.getSampleModel().getNumBands());

        int w2 = 70, h2 = 120, ch2 = 6;
        ImageLoader loader2 = new ImageLoader(h2, w2, ch2);

        BufferedImage scaled5 = loader2.scalingIfNeed(img1, true);
        assertEquals(w2, scaled5.getWidth());
        assertEquals(h2, scaled5.getHeight(), h2);
        assertEquals(BufferedImage.TYPE_4BYTE_ABGR, scaled5.getType());
        assertEquals(4, scaled5.getSampleModel().getNumBands());

        BufferedImage scaled6 = loader2.scalingIfNeed(img1, false);
        assertEquals(w2, scaled6.getWidth());
        assertEquals(h2, scaled6.getHeight());
        assertEquals(BufferedImage.TYPE_3BYTE_BGR, scaled6.getType());
        assertEquals(3, scaled6.getSampleModel().getNumBands());

    }

    @Test
    public void testToBufferedImageRGB() {
        BufferedImage img = makeRandomBufferedImage(false);
        int w = img.getWidth();
        int h = img.getHeight();
        int ch = 3;

        ImageLoader loader = new ImageLoader(0, 0, ch);
        INDArray arr = loader.toINDArrayBGR(img);
        BufferedImage img2 = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
        loader.toBufferedImageRGB(arr, img2);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int srcColor = img.getRGB(j, i);
                int restoredColor = img2.getRGB(j, i);
                assertEquals(srcColor, restoredColor);
            }
        }

    }

    private BufferedImage makeRandomBufferedImage(boolean alpha) {
        int w = rng.nextInt() % 100 + 100;
        int h = rng.nextInt() % 100 + 100;
        int type = alpha ? BufferedImage.TYPE_4BYTE_ABGR : BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage img = new BufferedImage(w, h, type);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int a = (alpha ? rng.nextInt() : 1) & 0xff;
                int r = rng.nextInt() & 0xff;
                int g = rng.nextInt() & 0xff;
                int b = rng.nextInt() & 0xff;
                int v = (a << 24) | (r << 16) | (g << 8) | b;
                img.setRGB(j, i, v);
            }
        }
        return img;
    }
}
