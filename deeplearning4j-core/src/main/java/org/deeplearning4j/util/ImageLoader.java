/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * Image loader for taking images and converting them to matrices
 * @author Adam Gibson
 *
 */
@Deprecated
public class ImageLoader {

    private int width = -1;
    private int height = -1;


    public ImageLoader() {
        super();
    }

    public ImageLoader(int width, int height) {
        super();
        this.width = width;
        this.height = height;
    }

    public INDArray asRowVector(File f) throws Exception {
        return NDArrayUtil.toNDArray(flattenedImageFromFile(f));
    }


    /**
     * Slices up an image in to a mini batch.
     *
     * @param f the file to load from
     * @param numMiniBatches the number of images in a mini batch
     * @param numRowsPerSlice the number of rows for each image
     * @return a tensor representing one image as a mini batch
     */
    public INDArray asImageMiniBatches(File f,int numMiniBatches,int numRowsPerSlice) {
        try {
            INDArray d = asMatrix(f);
            INDArray f2 = Nd4j.create(numMiniBatches, numRowsPerSlice, d.columns());
            return f2;
        }catch(Exception e) {
            throw new RuntimeException(e);
        }

    }

    public INDArray asMatrix(File f) throws IOException {
        return NDArrayUtil.toNDArray(fromFile(f));
    }

    public int[] flattenedImageFromFile(File f) throws Exception {
        return ArrayUtil.flatten(fromFile(f));
    }

    public int[][] fromFile(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        if (height > 0 && width > 0)
            image = toBufferedImage(image.getScaledInstance(height, width, Image.SCALE_SMOOTH));
        Raster raster = image.getData();
        int w = raster.getWidth(), h = raster.getHeight();
        int[][] ret = new int[w][h];
        for (int i = 0; i < w; i++)
            for (int j = 0; j < h; j++)
                ret[i][j] = raster.getSample(i, j, 0);

        return ret;
    }


    /**
     * Convert the given image to an rgb image
     * @param arr the array to use
     */
    public static  BufferedImage toBufferedImageRGB(INDArray arr) {
        if(arr.rank() < 3)
            throw new IllegalArgumentException("Arr must be 3d");

        int height = arr.size(-2);
        int width = arr.size(-1);

        BufferedImage image = new BufferedImage(width,height ,BufferedImage.TYPE_INT_ARGB);
        int[] data = new int[arr.size(-1) * arr.size(-2)];
        int i = 0;
        for (int y = 0; y < height; y++) {
            int red = (y * 255) / (height - 1);
            for (int x = 0; x < width; x++) {
                int green = (x * 255) / (width - 1);
                int blue = 128;
                data[i++] = (red << 16) | (green << 8) | blue;
            }
        }

        image.setRGB(0, 0, width, height, data, 0, width);

        return image;

    }

    /**
     *
     * @param matrix
     * @return
     */
    public static BufferedImage toImage(INDArray matrix) {
        if(matrix.isMatrix()) {
            BufferedImage img = new BufferedImage(matrix.size(-1), matrix.size(-2), BufferedImage.TYPE_INT_RGB);
            INDArray toRound = matrix;
            WritableRaster r = img.getRaster();
            int[] equiv = new int[matrix.length()];
            for(int i = 0; i < equiv.length; i++) {
                equiv[i] = (int) toRound.linearView().getDouble(i) * 256;
            }


            r.setDataElements(0,0,matrix.columns(),matrix.rows(),equiv);
            return img;
        }

        else {
            BufferedImage img = new BufferedImage(matrix.size(-1), matrix.size(-2), BufferedImage.TYPE_INT_RGB);
            INDArray toRound = Transforms.sigmoid(matrix);
            WritableRaster r = img.getRaster();
            int[] equiv = new int[matrix.length()];
            for(int i = 0; i < equiv.length; i++) {
                equiv[i] = (int) toRound.linearView().getDouble(i);
            }


            r.setDataElements(0,0,matrix.size(-1),matrix.size(-2),equiv);
            return img;
        }

    }


    /**
     * Converts a given Image into a BufferedImage
     *
     * @param img The Image to be converted
     * @return The converted BufferedImage
     */
    public static BufferedImage toBufferedImage(Image img)
    {
        if (img instanceof BufferedImage)
        {
            return (BufferedImage) img;
        }

        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_RGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

}
