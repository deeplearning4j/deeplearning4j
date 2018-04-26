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

import lombok.Data;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.Rect;

/**
 * Randomly crops an image to a desired output size. Will determine if
 * output size is valid, otherwise will throw an error.
 *
 * @author Justin Long (@crockpotveggies)
 */
@JsonIgnoreProperties({"rng", "converter"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class RandomCropTransform extends BaseImageTransform<Mat> {

    protected int outputHeight;
    protected int outputWidth;
    protected org.nd4j.linalg.api.rng.Random rng;

    private int x;
    private int y;

    public RandomCropTransform(@JsonProperty("outputHeight") int height, @JsonProperty("outputWidth") int width) {
        this(1234, height, width);
    }

    public RandomCropTransform(long seed, int height, int width) {
        this(null, seed, height, width);
    }

    public RandomCropTransform(Random random, long seed, int height, int width) {
        super(random);
        this.outputHeight = height;
        this.outputWidth = width;
        this.rng = Nd4j.getRandom();
        this.rng.setSeed(seed);
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    /**
     * Takes an image and returns a randomly cropped image.
     *
     * @param image  to transform, null == end of stream
     * @param random object to use (or null for deterministic)
     * @return transformed image
     */
    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        // ensure that transform is valid
        if (image.getFrame().imageHeight < outputHeight || image.getFrame().imageWidth < outputWidth)
            throw new UnsupportedOperationException(
                            "Output height/width cannot be more than the input image. Requested: " + outputHeight + "+x"
                                            + outputWidth + ", got " + image.getFrame().imageHeight + "+x"
                                            + image.getFrame().imageWidth);

        // determine boundary to place random offset
        int cropTop = image.getFrame().imageHeight - outputHeight;
        int cropLeft = image.getFrame().imageWidth - outputWidth;

        Mat mat = converter.convert(image.getFrame());
        int top = rng.nextInt(cropTop + 1);
        int left = rng.nextInt(cropLeft + 1);

        y = Math.min(top, mat.rows() - 1);
        x = Math.min(left, mat.cols() - 1);
        Mat result = mat.apply(new Rect(x, y, outputWidth, outputHeight));


        return new ImageWritable(converter.convert(result));
    }

    @Override
    public float[] query(float... coordinates) {
        float[] transformed = new float[coordinates.length];
        for (int i = 0; i < coordinates.length; i += 2) {
            transformed[i    ] = coordinates[i    ] - x;
            transformed[i + 1] = coordinates[i + 1] - y;
        }
        return transformed;
    }
}
