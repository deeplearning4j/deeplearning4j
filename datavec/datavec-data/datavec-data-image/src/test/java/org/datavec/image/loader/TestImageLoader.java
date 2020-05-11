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

import org.datavec.image.data.Image;
import org.junit.Test;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
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
    public void testScalingIfNeedWhenSuitableSizeButDiffChannel() {
        int width1 = 60;
        int height1 = 110;
        int channel1 = BufferedImage.TYPE_BYTE_GRAY;
        BufferedImage img1 = makeRandomBufferedImage(true, width1, height1);
        ImageLoader loader1 = new ImageLoader(height1, width1, channel1);
        BufferedImage scaled1 = loader1.scalingIfNeed(img1, false);
        assertEquals(width1, scaled1.getWidth());
        assertEquals(height1, scaled1.getHeight());
        assertEquals(channel1, scaled1.getType());
        assertEquals(1, scaled1.getSampleModel().getNumBands());

        int width2 = 70;
        int height2 = 120;
        int channel2 = BufferedImage.TYPE_BYTE_GRAY;
        BufferedImage img2 = makeRandomBufferedImage(false, width2, height2);
        ImageLoader loader2 = new ImageLoader(height2, width2, channel2);
        BufferedImage scaled2 = loader2.scalingIfNeed(img2, false);
        assertEquals(width2, scaled2.getWidth());
        assertEquals(height2, scaled2.getHeight());
        assertEquals(channel2, scaled2.getType());
        assertEquals(1, scaled2.getSampleModel().getNumBands());
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

    /**
     * Generate a Random BufferedImage with specified width and height
     *
     * @param alpha  Is image alpha
     * @param width  Proposed width
     * @param height Proposed height
     * @return Generated BufferedImage
     */
    private BufferedImage makeRandomBufferedImage(boolean alpha, int width, int height) {
        int type = alpha ? BufferedImage.TYPE_4BYTE_ABGR : BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage img = new BufferedImage(width, height, type);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
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

    /**
     * Generate a Random BufferedImage with random width and height
     *
     * @param alpha Is image alpha
     * @return Generated BufferedImage
     */
    private BufferedImage makeRandomBufferedImage(boolean alpha) {
        return makeRandomBufferedImage(alpha, rng.nextInt() % 100 + 100, rng.nextInt() % 100 + 100);
    }


    @Test
    public void testNCHW_NHWC() throws Exception {
        File f = Resources.asFile("datavec-data-image/voc/2007/JPEGImages/000005.jpg");

        ImageLoader il = new ImageLoader(32, 32, 3);

        //asMatrix(File, boolean)
        INDArray a_nchw = il.asMatrix(f);
        INDArray a_nchw2 = il.asMatrix(f, true);
        INDArray a_nhwc = il.asMatrix(f, false);

        assertEquals(a_nchw, a_nchw2);
        assertEquals(a_nchw, a_nhwc.permute(0,3,1,2));


        //asMatrix(InputStream, boolean)
        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            a_nchw = il.asMatrix(is);
        }
        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            a_nchw2 = il.asMatrix(is, true);
        }
        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            a_nhwc = il.asMatrix(is, false);
        }
        assertEquals(a_nchw, a_nchw2);
        assertEquals(a_nchw, a_nhwc.permute(0,3,1,2));


        //asImageMatrix(File, boolean)
        Image i_nchw = il.asImageMatrix(f);
        Image i_nchw2 = il.asImageMatrix(f, true);
        Image i_nhwc = il.asImageMatrix(f, false);

        assertEquals(i_nchw.getImage(), i_nchw2.getImage());
        assertEquals(i_nchw.getImage(), i_nhwc.getImage().permute(0,3,1,2));        //NHWC to NCHW


        //asImageMatrix(InputStream, boolean)
        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            i_nchw = il.asImageMatrix(is);
        }
        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            i_nchw2 = il.asImageMatrix(is, true);
        }
        try(InputStream is = new BufferedInputStream(new FileInputStream(f))){
            i_nhwc = il.asImageMatrix(is, false);
        }
        assertEquals(i_nchw.getImage(), i_nchw2.getImage());
        assertEquals(i_nchw.getImage(), i_nhwc.getImage().permute(0,3,1,2));        //NHWC to NCHW
    }
}
