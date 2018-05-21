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

package org.datavec.image.mnist;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


/**
 * <p>
 * Utility class for working with the MNIST database.
 * <p>
 * Provides methods for traversing the images and labels data files separately,
 * as well as simultaneously.
 * <p>
 * Provides also method for exporting an image by writing it as a PPM file.
 * <p> 
 * Example usage:
 * <pre>
 *  MnistManager m = new MnistManager("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
 *  m.setCurrent(10); //index of the image that we are interested in
 *  int[][] image = m.readImage();
 *  System.out.println("Label:" + m.readLabel());
 *  MnistManager.writeImageToPpm(image, "10.ppm");
 * </pre>
 */
public class MnistManager {
    private MnistImageFile images;
    private MnistLabelFile labels;

    /**
     * Writes the given image in the given file using the PPM data format.
     * 
     * @param image
     * @param ppmFileName
     * @throws java.io.IOException
     */
    public static void writeImageToPpm(int[][] image, String ppmFileName) throws IOException {
        try (BufferedWriter ppmOut = new BufferedWriter(new FileWriter(ppmFileName))) {
            int rows = image.length;
            int cols = image[0].length;
            ppmOut.write("P3\n");
            ppmOut.write("" + rows + " " + cols + " 255\n");
            for (int[] anImage : image) {
                StringBuilder s = new StringBuilder();
                for (int j = 0; j < cols; j++) {
                    s.append(anImage[j] + " " + anImage[j] + " " + anImage[j] + "  ");
                }
                ppmOut.write(s.toString());
            }
        }

    }

    /**
     * Constructs an instance managing the two given data files. Supports
     * <code>NULL</code> value for one of the arguments in case reading only one
     * of the files (images and labels) is required.
     *
     * @param imagesFile
     *            Can be <code>NULL</code>. In that case all future operations
     *            using that file will fail.
     * @param labelsFile
     *            Can be <code>NULL</code>. In that case all future operations
     *            using that file will fail.
     * @throws java.io.IOException
     */
    public MnistManager(String imagesFile, String labelsFile) throws IOException {
        if (imagesFile != null) {
            images = new MnistImageFile(imagesFile, "r");
        }
        if (labelsFile != null) {
            labels = new MnistLabelFile(labelsFile, "r");
        }
    }

    /**
     * Reads the current image.
     *
     * @return matrix
     * @throws java.io.IOException
     */
    public int[][] readImage() throws IOException {
        if (images == null) {
            throw new IllegalStateException("Images file not initialized.");
        }
        return images.readImage();
    }

    /**
     * Set the position to be read.
     *
     * @param index
     */
    public void setCurrent(int index) {
        images.setCurrentIndex(index);
        labels.setCurrentIndex(index);
    }

    /**
     * Reads the current label.
     *
     * @return int
     * @throws java.io.IOException
     */
    public int readLabel() throws IOException {
        if (labels == null) {
            throw new IllegalStateException("labels file not initialized.");
        }
        return labels.readLabel();
    }

    /**
     * Get the underlying images file as {@link MnistImageFile}.
     * 
     * @return {@link MnistImageFile}.
     */
    public MnistImageFile getImages() {
        return images;
    }

    /**
     * Get the underlying labels file as {@link MnistLabelFile}.
     *
     * @return {@link MnistLabelFile}.
     */
    public MnistLabelFile getLabels() {
        return labels;
    }
}
