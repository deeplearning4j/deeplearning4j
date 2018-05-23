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
package org.datavec.image.transform;

import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.junit.Ignore;
import org.junit.Test;

import java.awt.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.junit.Assert.*;

/**
 *
 * @author saudet
 */
public class TestImageTransform {
    static final long seed = 10;
    static final Random rng = new Random(seed);
    static final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

    @Test
    public void testBoxImageTransform() throws Exception {
        ImageTransform transform = new BoxImageTransform(rng, 237, 242).borderValue(Scalar.GRAY);

        for (int i = 0; i < 100; i++) {
            ImageWritable writable = makeRandomImage(0, 0, i % 4 + 1);
            Frame frame = writable.getFrame();

            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertEquals(237, f.imageWidth);
            assertEquals(242, f.imageHeight);
            assertEquals(frame.imageChannels, f.imageChannels);

            float[] coordinates = {1, 2, 3, 4, 0, 0};
            float[] transformed = transform.query(coordinates);
            int x = (frame.imageWidth - f.imageWidth) / 2;
            int y = (frame.imageHeight - f.imageHeight) / 2;

            assertEquals(1 - x, transformed[0], 0);
            assertEquals(2 - y, transformed[1], 0);
            assertEquals(3 - x, transformed[2], 0);
            assertEquals(4 - y, transformed[3], 0);
            assertEquals(  - x, transformed[4], 0);
            assertEquals(  - y, transformed[5], 0);
        }
        assertEquals(null, transform.transform(null));
    }

    @Test
    public void testCropImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 1);
        Frame frame = writable.getFrame();
        ImageTransform transform = new CropImageTransform(rng, frame.imageHeight / 2, frame.imageWidth / 2,
                        frame.imageHeight / 2, frame.imageWidth / 2);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertTrue(f.imageHeight <= frame.imageHeight);
            assertTrue(f.imageWidth <= frame.imageWidth);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));

        transform = new CropImageTransform(1, 2, 3, 4);
        writable = transform.transform(writable);
        float[] coordinates = {1, 2, 3, 4};
        float[] transformed = transform.query(coordinates);
        assertEquals(1 - 2, transformed[0], 0);
        assertEquals(2 - 1, transformed[1], 0);
        assertEquals(3 - 2, transformed[2], 0);
        assertEquals(4 - 1, transformed[3], 0);
    }

    @Test
    public void testFlipImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 3);
        Frame frame = writable.getFrame();
        ImageTransform transform = new FlipImageTransform(rng);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertEquals(f.imageHeight, frame.imageHeight);
            assertEquals(f.imageWidth, frame.imageWidth);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));

        transform = new FlipImageTransform(-2);
        writable = transform.transform(writable);
        float[] transformed = transform.query(new float[] {10, 20});
        assertEquals(10, transformed[0], 0);
        assertEquals(20, transformed[1], 0);

        transform = new FlipImageTransform(0);
        writable = transform.transform(writable);
        transformed = transform.query(new float[] {30, 40});
        assertEquals(30,                         transformed[0], 0);
        assertEquals(frame.imageHeight - 40 - 1, transformed[1], 0);

        transform = new FlipImageTransform(1);
        writable = transform.transform(writable);
        transformed = transform.query(new float[] {50, 60});
        assertEquals(frame.imageWidth - 50 - 1, transformed[0], 0);
        assertEquals(60,                        transformed[1], 0);

        transform = new FlipImageTransform(-1);
        writable = transform.transform(writable);
        transformed = transform.query(new float[] {70, 80});
        assertEquals(frame.imageWidth  - 70 - 1, transformed[0], 0);
        assertEquals(frame.imageHeight - 80 - 1, transformed[1], 0);
    }

    @Test
    public void testScaleImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 4);
        Frame frame = writable.getFrame();
        ImageTransform transform = new ScaleImageTransform(rng, frame.imageWidth / 2, frame.imageHeight / 2);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertTrue(f.imageHeight >= frame.imageHeight / 2);
            assertTrue(f.imageHeight <= 3 * frame.imageHeight / 2);
            assertTrue(f.imageWidth >= frame.imageWidth / 2);
            assertTrue(f.imageWidth <= 3 * frame.imageWidth / 2);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));

        transform = new ScaleImageTransform(frame.imageWidth, 2 * frame.imageHeight);
        writable = transform.transform(writable);
        float[] coordinates = {5, 7, 11, 13};
        float[] transformed = transform.query(coordinates);
        assertEquals(5 * 2,  transformed[0], 0);
        assertEquals(7 * 3,  transformed[1], 0);
        assertEquals(11 * 2, transformed[2], 0);
        assertEquals(13 * 3, transformed[3], 0);
    }

    @Test
    public void testRotateImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 1);
        Frame frame = writable.getFrame();
        ImageTransform transform =
                        new RotateImageTransform(rng, 180).interMode(INTER_NEAREST).borderMode(BORDER_REFLECT);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertEquals(f.imageHeight, frame.imageHeight);
            assertEquals(f.imageWidth, frame.imageWidth);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));

        transform = new RotateImageTransform(0, 0, -90, 0);
        writable = transform.transform(writable);
        float[] coordinates = {frame.imageWidth / 2, frame.imageHeight / 2, 0, 0};
        float[] transformed = transform.query(coordinates);
        assertEquals(frame.imageWidth  / 2, transformed[0], 0);
        assertEquals(frame.imageHeight / 2, transformed[1], 0);
        assertEquals((frame.imageHeight + frame.imageWidth) / 2, transformed[2], 1);
        assertEquals((frame.imageHeight - frame.imageWidth) / 2, transformed[3], 1);
    }

    @Test
    public void testWarpImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 1);
        Frame frame = writable.getFrame();
        ImageTransform transform = new WarpImageTransform(rng, frame.imageWidth / 10).interMode(INTER_CUBIC)
                        .borderMode(BORDER_REPLICATE);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertEquals(f.imageHeight, frame.imageHeight);
            assertEquals(f.imageWidth, frame.imageWidth);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));

        transform = new WarpImageTransform(1, 2, 3, 4, 5, 6, 7, 8);
        writable = transform.transform(writable);
        float[] coordinates = { 0, 0,                                frame.imageWidth, 0,
                                frame.imageWidth, frame.imageHeight, 0, frame.imageHeight};
        float[] transformed = transform.query(coordinates);
        assertEquals(1,                     transformed[0], 0);
        assertEquals(2,                     transformed[1], 0);
        assertEquals(3 + frame.imageWidth,  transformed[2], 0);
        assertEquals(4,                     transformed[3], 0);
        assertEquals(5 + frame.imageWidth,  transformed[4], 0);
        assertEquals(6 + frame.imageHeight, transformed[5], 0);
        assertEquals(7,                     transformed[6], 0);
        assertEquals(8 + frame.imageHeight, transformed[7], 0);
    }

    @Test
    public void testMultiImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 3);
        Frame frame = writable.getFrame();
        ImageTransform transform = new MultiImageTransform(rng, new CropImageTransform(10),
                        new FlipImageTransform(), new ScaleImageTransform(10), new WarpImageTransform(10));

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertTrue(f.imageHeight >= frame.imageHeight - 30);
            assertTrue(f.imageHeight <= frame.imageHeight + 20);
            assertTrue(f.imageWidth >= frame.imageWidth - 30);
            assertTrue(f.imageWidth <= frame.imageWidth + 20);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));

        transform = new MultiImageTransform(new ColorConversionTransform(COLOR_BGR2RGB));
        writable = transform.transform(writable);
        float[] transformed = transform.query(new float[] {11, 22});
        assertEquals(11, transformed[0], 0);
        assertEquals(22, transformed[1], 0);
    }

    @Ignore
    @Test
    public void testFilterImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 4);
        Frame frame = writable.getFrame();
        ImageTransform transform = new FilterImageTransform("noise=alls=20:allf=t+u,format=rgba", frame.imageWidth,
                        frame.imageHeight, frame.imageChannels);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertEquals(f.imageHeight, frame.imageHeight);
            assertEquals(f.imageWidth, frame.imageWidth);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));
    }

    @Test
    public void testShowImageTransform() throws Exception {
        if (GraphicsEnvironment.isHeadless()) { return; }

        ImageWritable writable = makeRandomImage(0, 0, 3);
        ImageTransform transform = new ShowImageTransform("testShowImageTransform", 100);

        for (int i = 0; i < 10; i++) {
            ImageWritable w = transform.transform(writable);
            assertEquals(w, writable);
        }

        assertEquals(null, transform.transform(null));

        float[] transformed = transform.query(new float[] {33, 44});
        assertEquals(33, transformed[0], 0);
        assertEquals(44, transformed[1], 0);
    }

    @Test
    public void testConvertColorTransform() throws Exception {
        if (GraphicsEnvironment.isHeadless()) { return; }

        //        Mat origImage = new Mat();
        //        Mat transImage = new Mat();
        //        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        ImageWritable writable = makeRandomImage(32, 32, 3);
        Frame frame = writable.getFrame();
        ImageTransform showOrig = new ShowImageTransform("Original Image", 50);
        showOrig.transform(writable);
        //        origImage = converter.convert(writable.getFrame());

        ImageTransform transform = new ColorConversionTransform(new Random(42), COLOR_BGR2YCrCb);
        ImageWritable w = transform.transform(writable);
        ImageTransform showTrans = new ShowImageTransform("LUV Image", 50);
        showTrans.transform(writable);
        //        transImage = converter.convert(writable.getFrame());

        Frame newframe = w.getFrame();
        assertNotEquals(frame, newframe);
        assertEquals(null, transform.transform(null));

        float[] transformed = transform.query(new float[] {55, 66});
        assertEquals(55, transformed[0], 0);
        assertEquals(66, transformed[1], 0);
    }

    @Test
    public void testHistEqualization() throws CanvasFrame.Exception {
        if (GraphicsEnvironment.isHeadless()) { return; }

        // TODO pull out historgram to confirm equalization...
        ImageWritable writable = makeRandomImage(32, 32, 3);
        Frame frame = writable.getFrame();
        ImageTransform showOrig = new ShowImageTransform("Original Image", 50);
        showOrig.transform(writable);

        ImageTransform transform = new EqualizeHistTransform(new Random(42), COLOR_BGR2YCrCb);
        ImageWritable w = transform.transform(writable);

        ImageTransform showTrans = new ShowImageTransform("LUV Image", 50);
        showTrans.transform(writable);
        Frame newframe = w.getFrame();
        assertNotEquals(frame, newframe);
        assertEquals(null, transform.transform(null));

        float[] transformed = transform.query(new float[] {66, 77});
        assertEquals(66, transformed[0], 0);
        assertEquals(77, transformed[1], 0);
    }

    @Test
    public void testRandomCropTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 1);
        Frame frame = writable.getFrame();
        ImageTransform transform = new RandomCropTransform(frame.imageHeight / 2, frame.imageWidth / 2);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertTrue(f.imageHeight == frame.imageHeight / 2);
            assertTrue(f.imageWidth == frame.imageWidth / 2);
        }
        assertEquals(null, transform.transform(null));

        transform = new RandomCropTransform(frame.imageHeight, frame.imageWidth);
        writable = transform.transform(writable);
        float[] coordinates = {2, 4, 6, 8};
        float[] transformed = transform.query(coordinates);
        assertEquals(2, transformed[0], 0);
        assertEquals(4, transformed[1], 0);
        assertEquals(6, transformed[2], 0);
        assertEquals(8, transformed[3], 0);
    }

    @Test
    public void testProbabilisticPipelineTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 3);
        Frame frame = writable.getFrame();

        ImageTransform randCrop = new RandomCropTransform(frame.imageHeight / 2, frame.imageWidth / 2);
        ImageTransform flip = new FlipImageTransform();
        List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        pipeline.add(new Pair<>(randCrop, 1.0));
        pipeline.add(new Pair<>(flip, 0.5));
        ImageTransform transform = new PipelineImageTransform(pipeline, true);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertTrue(f.imageHeight == frame.imageHeight / 2);
            assertTrue(f.imageWidth == frame.imageWidth / 2);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));

        transform = new PipelineImageTransform(new EqualizeHistTransform());
        writable = transform.transform(writable);
        float[] transformed = transform.query(new float[] {88, 99});
        assertEquals(88, transformed[0], 0);
        assertEquals(99, transformed[1], 0);
    }

    /**
     * This test code is kind of a manual test using specific image(largestblobtest.jpg)
     * with particular thresholds(blur size, thresholds for edge detector)
     * The cropped largest blob size should be 74x61
     * because we use a specific image and thresholds
     *
     * @throws Exception
     */
    @Test
    public void testLargestBlobCropTransform() throws Exception {
        if (GraphicsEnvironment.isHeadless()) { return; }


        java.io.File f1 = new ClassPathResource("datavec-data-image/testimages2/largestblobtest.jpg").getFile();
        NativeImageLoader loader = new NativeImageLoader();
        ImageWritable writable = loader.asWritable(f1);

        ImageTransform showOrig = new ShowImageTransform("Original Image", 50);
        showOrig.transform(writable);

        ImageTransform transform =
                        new LargestBlobCropTransform(null, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, 3, 3, 100, 300, true);
        ImageWritable w = transform.transform(writable);

        ImageTransform showTrans = new ShowImageTransform("Largest Blob", 50);
        showTrans.transform(w);
        Frame newFrame = w.getFrame();

        assertEquals(newFrame.imageHeight, 74);
        assertEquals(newFrame.imageWidth, 61);

        float[] transformed = transform.query(new float[] {88, 32});
        assertEquals(0, transformed[0], 0);
        assertEquals(0, transformed[1], 0);
    }

    public static ImageWritable makeRandomImage(int height, int width, int channels) {
        if (height <= 0) {
            height = rng.nextInt() % 100 + 200;
        }
        if (width <= 0) {
            width = rng.nextInt() % 100 + 200;
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
        Frame frame = converter.convert(img);
        return new ImageWritable(frame);
    }
}
