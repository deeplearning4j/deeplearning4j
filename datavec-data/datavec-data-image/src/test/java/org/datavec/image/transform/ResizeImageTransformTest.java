package org.datavec.image.transform;

import org.bytedeco.javacv.Frame;
import org.datavec.image.data.ImageWritable;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Tests for ResizeImage
 *
 * @author raver119@gmail.com
 */
public class ResizeImageTransformTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testResizeUpscale1() throws Exception {
        ImageWritable srcImg = TestImageTransform.makeRandomImage(32, 32, 3);

        ResizeImageTransform transform = new ResizeImageTransform(200, 200);

        ImageWritable dstImg = transform.transform(srcImg);

        Frame f = dstImg.getFrame();
        assertEquals(f.imageWidth, 200);
        assertEquals(f.imageHeight, 200);

        float[] coordinates = {100, 200};
        float[] transformed = transform.query(coordinates);
        assertEquals(200f * 100 / 32, transformed[0], 0);
        assertEquals(200f * 200 / 32, transformed[1], 0);
    }

    @Test
    public void testResizeDownscale() throws Exception {
        ImageWritable srcImg = TestImageTransform.makeRandomImage(571, 443, 3);

        ResizeImageTransform transform = new ResizeImageTransform(200, 200);

        ImageWritable dstImg = transform.transform(srcImg);

        Frame f = dstImg.getFrame();
        assertEquals(f.imageWidth, 200);
        assertEquals(f.imageHeight, 200);

        float[] coordinates = {300, 400};
        float[] transformed = transform.query(coordinates);
        assertEquals(200f * 300 / 443, transformed[0], 0);
        assertEquals(200f * 400 / 571, transformed[1], 0);
    }

}
