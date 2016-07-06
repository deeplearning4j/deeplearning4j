/*
 *
 *  *
 *  *  * Copyright 2016 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */
package org.canova.image.transform;

import java.util.Random;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.junit.Test;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.canova.image.data.ImageWritable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import static org.bytedeco.javacpp.opencv_core.*;

/**
 *
 * @author saudet
 */
public class TestImageTransform {
    static final long seed = 10;
    static final Random rng = new Random(seed);
    static final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

    @Test
    public void testCropImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 1);
        Frame frame = writable.getFrame();
        ImageTransform transform = new CropImageTransform(rng,
                frame.imageHeight / 2, frame.imageWidth / 2,
                frame.imageHeight / 2, frame.imageWidth / 2);

        for (int i = 0; i < 100; i++) {
            ImageWritable w = transform.transform(writable);
            Frame f = w.getFrame();
            assertTrue(f.imageHeight <= frame.imageHeight);
            assertTrue(f.imageWidth <= frame.imageWidth);
            assertEquals(f.imageChannels, frame.imageChannels);
        }
        assertEquals(null, transform.transform(null));
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
    }

    @Test
    public void testScaleImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 4);
        Frame frame = writable.getFrame();
        ImageTransform transform = new ScaleImageTransform(rng,
                frame.imageWidth / 2, frame.imageHeight / 2);

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
    }

    @Test
    public void testWarpImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 1);
        Frame frame = writable.getFrame();
        ImageTransform transform = new WarpImageTransform(rng, frame.imageWidth / 10);

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
    public void testMultiImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 3);
        Frame frame = writable.getFrame();
        ImageTransform transform = new MultiImageTransform(rng,
                new CropImageTransform(10), new FlipImageTransform(),
                new ScaleImageTransform(10), new WarpImageTransform(10));

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
    }

    @Test
    public void testFilterImageTransform() throws Exception {
        ImageWritable writable = makeRandomImage(0, 0, 4);
        Frame frame = writable.getFrame();
        ImageTransform transform = new FilterImageTransform("noise=alls=20:allf=t+u,format=rgba",
                frame.imageWidth, frame.imageHeight, frame.imageChannels);

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
        ImageWritable writable = makeRandomImage(0, 0, 3);
        ImageTransform transform = new ShowImageTransform("testShowImageTransform", 100);

        for (int i = 0; i < 10; i++) {
            ImageWritable w = transform.transform(writable);
            assertEquals(w, writable);
        }

        assertEquals(null, transform.transform(null));
    }

    ImageWritable makeRandomImage(int height, int width, int channels) {
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
        Frame frame = converter.convert(img);
        return new ImageWritable(frame);
    }
}
