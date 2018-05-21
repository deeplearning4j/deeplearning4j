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

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * crop images based on it's largest blob. Calls internally
 * {@link org.bytedeco.javacpp.opencv_imgproc#blur(Mat, Mat, Size)}
 * {@link org.bytedeco.javacpp.opencv_imgproc#Canny(Mat ,Mat, double, double)}
 * {@link org.bytedeco.javacpp.opencv_imgproc#threshold(Mat, Mat, double, double, int)}
 * {@link org.bytedeco.javacpp.opencv_imgproc#findContours(Mat, MatVector, Mat, int, int)}
 * {@link org.bytedeco.javacpp.opencv_imgproc#contourArea(Mat, boolean)}
 *
 * @author antdood
 */
@Data
public class LargestBlobCropTransform extends BaseImageTransform<Mat> {

    protected org.nd4j.linalg.api.rng.Random rng;

    protected int mode, method, blurWidth, blurHeight, upperThresh, lowerThresh;
    protected boolean isCanny;

    private int x;
    private int y;

    /** Calls {@code this(null}*/
    public LargestBlobCropTransform() {
        this(null);
    }

    /** Calls {@code this(random, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, 3, 3, 100, 200, true)}*/
    public LargestBlobCropTransform(Random random) {
        this(random, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, 3, 3, 100, 200, true);
    }

    /**
     *
     * @param random        Object to use (or null for deterministic)
     * @param mode          Contour retrieval mode
     * @param method        Contour approximation method
     * @param blurWidth     Width of blurring kernel size
     * @param blurHeight    Height of blurring kernel size
     * @param lowerThresh   Lower threshold for either Canny or Threshold
     * @param upperThresh   Upper threshold for either Canny or Threshold
     * @param isCanny       Whether the edge detector is Canny or Threshold
     */
    public LargestBlobCropTransform(Random random, int mode, int method, int blurWidth, int blurHeight, int lowerThresh,
                    int upperThresh, boolean isCanny) {
        super(random);
        this.rng = Nd4j.getRandom();
        this.mode = mode;
        this.method = method;
        this.blurWidth = blurWidth;
        this.blurHeight = blurHeight;
        this.lowerThresh = lowerThresh;
        this.upperThresh = upperThresh;
        this.isCanny = isCanny;
        this.converter = new OpenCVFrameConverter.ToMat();
    }

    /**
     * Takes an image and returns a cropped image based on it's largest blob.
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

        //Convert image to gray and blur
        Mat original = converter.convert(image.getFrame());
        Mat grayed = new Mat();
        cvtColor(original, grayed, CV_BGR2GRAY);
        if (blurWidth > 0 && blurHeight > 0)
            blur(grayed, grayed, new Size(blurWidth, blurHeight));

        //Get edges from Canny edge detector
        Mat edgeOut = new Mat();
        if (isCanny)
            Canny(grayed, edgeOut, lowerThresh, upperThresh);
        else
            threshold(grayed, edgeOut, lowerThresh, upperThresh, 0);

        double largestArea = 0;
        Rect boundingRect = new Rect();
        MatVector contours = new MatVector();
        Mat hierarchy = new Mat();

        findContours(edgeOut, contours, hierarchy, this.mode, this.method);

        for (int i = 0; i < contours.size(); i++) {
            //  Find the area of contour
            double area = contourArea(contours.get(i), false);

            if (area > largestArea) {
                // Find the bounding rectangle for biggest contour
                boundingRect = boundingRect(contours.get(i));
            }
        }

        //Apply crop and return result
        x = boundingRect.x();
        y = boundingRect.y();
        Mat result = original.apply(boundingRect);

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
