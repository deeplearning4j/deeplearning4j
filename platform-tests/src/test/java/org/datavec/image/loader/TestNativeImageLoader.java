/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.image.loader;

import lombok.extern.slf4j.Slf4j;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.data.Image;
import org.datavec.image.data.ImageWritable;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Path;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 *
 * @author saudet
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
@Tag(TagNames.LARGE_RESOURCES)
@Tag(TagNames.LONG_TEST)
public class TestNativeImageLoader {
    static final long seed = 10;
    static final Random rng = new Random(seed);


    @Test
    public void testAsRowVector() throws Exception {
        org.opencv.core.Mat img1 = makeRandomOrgOpenCvCoreMatImage(0, 0, 1);
        Mat img2 = makeRandomImage(0, 0, 3);

        int w1 = 35, h1 = 79, ch1 = 3;
        NativeImageLoader loader1 = new NativeImageLoader(h1, w1, ch1);

        INDArray array1 = loader1.asRowVector(img1);
        assertEquals(2, array1.rank());
        assertEquals(1, array1.rows());
        assertEquals(h1 * w1 * ch1, array1.columns());
        assertNotEquals(0.0, array1.sum().getDouble(0), 0.0);

        INDArray array2 = loader1.asRowVector(img2);
        assertEquals(2, array2.rank());
        assertEquals(1, array2.rows());
        assertEquals(h1 * w1 * ch1, array2.columns());
        assertNotEquals(0.0, array2.sum().getDouble(0), 0.0);

        int w2 = 103, h2 = 68, ch2 = 4;
        NativeImageLoader loader2 = new NativeImageLoader(h2, w2, ch2);
        loader2.direct = false; // simulate conditions under Android

        INDArray array3 = loader2.asRowVector(img1);
        assertEquals(2, array3.rank());
        assertEquals(1, array3.rows());
        assertEquals(h2 * w2 * ch2, array3.columns());
        assertNotEquals(0.0, array3.sum().getDouble(0), 0.0);

        INDArray array4 = loader2.asRowVector(img2);
        assertEquals(2, array4.rank());
        assertEquals(1, array4.rows());
        assertEquals(h2 * w2 * ch2, array4.columns());
        assertNotEquals(0.0, array4.sum().getDouble(0), 0.0);
    }

    @Test
    public void testDataTypes_1() throws Exception {
        var dtypes = new DataType[]{DataType.FLOAT, DataType.HALF, DataType.SHORT, DataType.INT};

        var dt = Nd4j.dataType();

        for (var dtype: dtypes) {
            Nd4j.setDataType(dtype);
            int w3 = 123, h3 = 77, ch3 = 3;
            var loader = new NativeImageLoader(h3, w3, ch3);
            File f3 = new ClassPathResource("datavec-data-image/testimages/class0/2.jpg").getFile();
            ImageWritable iw3 = loader.asWritable(f3);

            var array = loader.asMatrix(iw3);

            assertEquals(dtype, array.dataType());
        }

        Nd4j.setDataType(dt);
    }

    @Test
    public void testDataTypes_2() throws Exception {
        var dtypes = new DataType[]{DataType.FLOAT, DataType.HALF, DataType.SHORT, DataType.INT};

        var dt = Nd4j.dataType();

        for (var dtype: dtypes) {
            Nd4j.setDataType(dtype);
            int w3 = 123, h3 = 77, ch3 = 3;
            var loader = new NativeImageLoader(h3, w3, 1);
            File f3 = new ClassPathResource("datavec-data-image/testimages/class0/2.jpg").getFile();
            var array = loader.asMatrix(f3);

            assertEquals(dtype, array.dataType());
        }

        Nd4j.setDataType(dt);
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
        assertNotEquals(0.0, array1.sum().getDouble(0), 0.0);

        INDArray array2 = loader1.asMatrix(img2);
        assertEquals(4, array2.rank());
        assertEquals(1, array2.size(0));
        assertEquals(1, array2.size(1));
        assertEquals(h1, array2.size(2));
        assertEquals(w1, array2.size(3));
        assertNotEquals(0.0, array2.sum().getDouble(0), 0.0);

        int w2 = 111, h2 = 66, ch2 = 3;
        NativeImageLoader loader2 = new NativeImageLoader(h2, w2, ch2);
        loader2.direct = false; // simulate conditions under Android

        INDArray array3 = loader2.asMatrix(img1);
        assertEquals(4, array3.rank());
        assertEquals(1, array3.size(0));
        assertEquals(3, array3.size(1));
        assertEquals(h2, array3.size(2));
        assertEquals(w2, array3.size(3));
        assertNotEquals(0.0, array3.sum().getDouble(0), 0.0);

        INDArray array4 = loader2.asMatrix(img2);
        assertEquals(4, array4.rank());
        assertEquals(1, array4.size(0));
        assertEquals(3, array4.size(1));
        assertEquals(h2, array4.size(2));
        assertEquals(w2, array4.size(3));
        assertNotEquals(0.0, array4.sum().getDouble(0), 0.0);

        int w3 = 123, h3 = 77, ch3 = 3;
        NativeImageLoader loader3 = new NativeImageLoader(h3, w3, ch3);
        File f3 = new ClassPathResource("datavec-data-image/testimages/class0/2.jpg").getFile();
        ImageWritable iw3 = loader3.asWritable(f3);

        INDArray array5 = loader3.asMatrix(iw3);
        assertEquals(4, array5.rank());
        assertEquals(1, array5.size(0));
        assertEquals(3, array5.size(1));
        assertEquals(h3, array5.size(2));
        assertEquals(w3, array5.size(3));
        assertNotEquals(0.0, array5.sum().getDouble(0), 0.0);

        Mat mat = loader3.asMat(array5);
        assertEquals(w3, mat.cols());
        assertEquals(h3, mat.rows());
        assertEquals(ch3, mat.channels());
        assertTrue(mat.type() == CV_32FC(ch3) || mat.type() == CV_64FC(ch3));
        assertNotEquals(0.0, sumElems(mat).get(), 0.0);

        Frame frame = loader3.asFrame(array5, Frame.DEPTH_UBYTE);
        assertEquals(w3, frame.imageWidth);
        assertEquals(h3, frame.imageHeight);
        assertEquals(ch3, frame.imageChannels);
        assertEquals(Frame.DEPTH_UBYTE, frame.imageDepth);

        Java2DNativeImageLoader loader4 = new Java2DNativeImageLoader();
        BufferedImage img12 = loader4.asBufferedImage(array1);
        assertEquals(array1, loader4.asMatrix(img12));

        NativeImageLoader loader5 = new NativeImageLoader(0, 0, 0);
        loader5.direct = false; // simulate conditions under Android
        INDArray array7 = loader5.asMatrix(f3);
        assertEquals(4, array7.rank());
        assertEquals(1, array7.size(0));
        assertEquals(3, array7.size(1));
        assertEquals(32, array7.size(2));
        assertEquals(32, array7.size(3));
        assertNotEquals(0.0, array7.sum().getDouble(0), 0.0);
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
        assertNotEquals(0.0, sumElems(scaled1).get(), 0.0);

        Mat scaled2 = loader1.scalingIfNeed(img2);
        assertEquals(h1, scaled2.rows());
        assertEquals(w1, scaled2.cols());
        assertEquals(img2.channels(), scaled2.channels());
        assertNotEquals(0.0, sumElems(scaled2).get(), 0.0);

        int w2 = 70, h2 = 120, ch2 = 3;
        NativeImageLoader loader2 = new NativeImageLoader(h2, w2, ch2);
        loader2.direct = false; // simulate conditions under Android

        Mat scaled3 = loader2.scalingIfNeed(img1);
        assertEquals(h2, scaled3.rows());
        assertEquals(w2, scaled3.cols());
        assertEquals(img1.channels(), scaled3.channels());
        assertNotEquals(0.0, sumElems(scaled3).get(), 0.0);

        Mat scaled4 = loader2.scalingIfNeed(img2);
        assertEquals(h2, scaled4.rows());
        assertEquals(w2, scaled4.cols());
        assertEquals(img2.channels(), scaled4.channels());
        assertNotEquals(0.0, sumElems(scaled4).get(), 0.0);
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
        assertNotEquals(0.0, sumElems(cropped1).get(), 0.0);

        Mat cropped2 = loader.centerCropIfNeeded(img2);
        assertEquals(70, cropped2.rows());
        assertEquals(95, cropped2.cols());
        assertEquals(img2.channels(), cropped2.channels());
        assertNotEquals(0.0, sumElems(cropped2).get(), 0.0);
    }


    BufferedImage makeRandomBufferedImage(int height, int width, int channels) {
        Mat img = makeRandomImage(height, width, channels);

        OpenCVFrameConverter.ToMat c = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter c2 = new Java2DFrameConverter();

        return c2.convert(c.convert(img));
    }

    org.opencv.core.Mat makeRandomOrgOpenCvCoreMatImage(int height, int width, int channels) {
        Mat img = makeRandomImage(height, width, channels);

        Loader.load(org.bytedeco.opencv.opencv_java.class);
        OpenCVFrameConverter.ToOrgOpenCvCoreMat c = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();

        return c.convert(c.convert(img));
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

    @Test
    public void testAsWritable() throws Exception {
        String f0 = new ClassPathResource("datavec-data-image/testimages/class0/0.jpg").getFile().getAbsolutePath();

        NativeImageLoader imageLoader = new NativeImageLoader();
        ImageWritable img = imageLoader.asWritable(f0);

        assertEquals(32, img.getFrame().imageHeight);
        assertEquals(32, img.getFrame().imageWidth);
        assertEquals(3, img.getFrame().imageChannels);

        BufferedImage img1 = makeRandomBufferedImage(0, 0, 3);
        Mat img2 = makeRandomImage(0, 0, 4);

        int w1 = 33, h1 = 77, ch1 = 1;
        NativeImageLoader loader1 = new NativeImageLoader(h1, w1, ch1);

        INDArray array1 = loader1.asMatrix(f0);
        assertEquals(4, array1.rank());
        assertEquals(1, array1.size(0));
        assertEquals(1, array1.size(1));
        assertEquals(h1, array1.size(2));
        assertEquals(w1, array1.size(3));
        assertNotEquals(0.0, array1.sum().getDouble(0), 0.0);
    }

    @Test
    public void testNativeImageLoaderEmptyStreams(@TempDir Path testDir) throws Exception {
        File dir = testDir.toFile();
        File f = new File(dir, "myFile.jpg");
        f.createNewFile();

        NativeImageLoader nil = new NativeImageLoader(32, 32, 3);

        try(InputStream is = new FileInputStream(f)){
            nil.asMatrix(is);
            fail("Expected exception");
        } catch (IOException e){
            String msg = e.getMessage();
            assertTrue(msg.contains("decode image"),msg);
        }

        try(InputStream is = new FileInputStream(f)){
            nil.asImageMatrix(is);
            fail("Expected exception");
        } catch (IOException e){
            String msg = e.getMessage();
            assertTrue(msg.contains("decode image"),msg);
        }

        try(InputStream is = new FileInputStream(f)){
            nil.asRowVector(is);
            fail("Expected exception");
        } catch (IOException e){
            String msg = e.getMessage();
            assertTrue(msg.contains("decode image"),msg);
        }

        try(InputStream is = new FileInputStream(f)){
            INDArray arr = Nd4j.create(DataType.FLOAT, 1, 3, 32, 32);
            nil.asMatrixView(is, arr);
            fail("Expected exception");
        } catch (IOException e){
            String msg = e.getMessage();
            assertTrue( msg.contains("decode image"),msg);
        }
    }

    @Test
    public void testNCHW_NHWC() throws Exception {
        File f = Resources.asFile("datavec-data-image/voc/2007/JPEGImages/000005.jpg");

        NativeImageLoader il = new NativeImageLoader(32, 32, 3);

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
