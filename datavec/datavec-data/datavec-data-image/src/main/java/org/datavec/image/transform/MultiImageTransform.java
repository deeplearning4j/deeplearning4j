package org.datavec.image.transform;

import lombok.Data;
import org.datavec.image.data.ImageWritable;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Transforms images deterministically or randomly with the help of an array of ImageTransform
 *
 * @author saudet
 */
@Data
public class MultiImageTransform extends BaseImageTransform<Mat> {
    private PipelineImageTransform transform;

    public MultiImageTransform(ImageTransform... transforms) {
        this(null, transforms);
    }

    public MultiImageTransform(Random random, ImageTransform... transforms) {
        super(random);
        transform = new PipelineImageTransform(transforms);
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        return random == null ? transform.transform(image) : transform.transform(image, random);
    }

    @Override
    public float[] query(float... coordinates) {
        return transform.query(coordinates);
    }
}
