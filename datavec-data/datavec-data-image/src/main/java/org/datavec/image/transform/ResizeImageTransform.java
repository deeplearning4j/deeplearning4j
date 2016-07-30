package org.datavec.image.transform;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Resize image transform is suited to force the same image size for whole pipeline.
 *
 * @author raver119@gmail.com
 */
public class ResizeImageTransform extends BaseImageTransform<opencv_core.Mat>  {

    int newHeight, newWidth;

    /**
     * Returns new ResizeImageTransform object
     *
     * @param newWidth new Width for the outcome images
     * @param newHeight new Height for outcome images
     */
    public ResizeImageTransform(int newWidth, int newHeight) {
        this(null, newWidth, newHeight);
    }

    /**
     * Returns new ResizeImageTransform object
     *
     * @param random Random
     * @param newWidth new Width for the outcome images
     * @param newHeight new Height for outcome images
     */
    public ResizeImageTransform(Random random, int newWidth, int newHeight) {
        super(random);

        this.newWidth = newWidth;
        this.newHeight = newHeight;

        converter = new OpenCVFrameConverter.ToMat();
    }

    /**
     * Takes an image and returns a transformed image.
     * Uses the random object in the case of random transformations.
     *
     * @param image  to transform, null == end of stream
     * @param random object to use (or null for deterministic)
     * @return transformed image
     */
    @Override
    public ImageWritable transform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        opencv_core.Mat mat = converter.convert(image.getFrame());

        opencv_core.Mat result = new opencv_core.Mat();
        resize(mat, result, new opencv_core.Size(newWidth, newHeight));

        return new ImageWritable(converter.convert(result));
    }
}
