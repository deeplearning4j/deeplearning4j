/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.image.loader;

import com.github.jaiimageio.impl.plugins.tiff.TIFFImageReaderSpi;
import com.github.jaiimageio.impl.plugins.tiff.TIFFImageWriterSpi;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

import javax.imageio.ImageIO;
import javax.imageio.spi.IIORegistry;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.*;
import java.util.Arrays;

/**
 * Image loader for taking images
 * and converting them to matrices
 * @author Adam Gibson
 *
 */
public class ImageLoader extends BaseImageLoader {

    static {
        ImageIO.scanForPlugins();
        IIORegistry registry = IIORegistry.getDefaultInstance();
        registry.registerServiceProvider(new TIFFImageWriterSpi());
        registry.registerServiceProvider(new TIFFImageReaderSpi());
        registry.registerServiceProvider(new com.twelvemonkeys.imageio.plugins.jpeg.JPEGImageReaderSpi());
        registry.registerServiceProvider(new com.twelvemonkeys.imageio.plugins.jpeg.JPEGImageWriterSpi());
        registry.registerServiceProvider(new com.twelvemonkeys.imageio.plugins.psd.PSDImageReaderSpi());
        registry.registerServiceProvider(Arrays.asList(new com.twelvemonkeys.imageio.plugins.bmp.BMPImageReaderSpi(),
                        new com.twelvemonkeys.imageio.plugins.bmp.CURImageReaderSpi(),
                        new com.twelvemonkeys.imageio.plugins.bmp.ICOImageReaderSpi()));
    }

    public ImageLoader() {
        super();
    }

    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load*
     * @param width  the width to load
    
     */
    public ImageLoader(long height, long width) {
        super();
        this.height = height;
        this.width = width;
    }


    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     */
    public ImageLoader(long height, long width, long channels) {
        super();
        this.height = height;
        this.width = width;
        this.channels = channels;
    }

    /**
     * Instantiate an image with the given
     * height and width
     * @param height the height to load
     * @param width  the width to load
     * @param channels the number of channels for the image*
     * @param centerCropIfNeeded to crop before rescaling and converting
     */
    public ImageLoader(long height, long width, long channels, boolean centerCropIfNeeded) {
        this(height, width, channels);
        this.centerCropIfNeeded = centerCropIfNeeded;
    }

    /**
     * Convert a file to a row vector
     *
     * @param f the image to convert
     * @return the flattened image
     * @throws IOException
     */
    public INDArray asRowVector(File f) throws IOException {
        return asRowVector(ImageIO.read(f));
        //        if(channels == 3) {
        //            return toRaveledTensor(f);
        //        }
        //        return NDArrayUtil.toNDArray(flattenedImageFromFile(f));
    }

    public INDArray asRowVector(InputStream inputStream) throws IOException {
        return asRowVector(ImageIO.read(inputStream));
        //        return asMatrix(inputStream).ravel();
    }

    /**
     * Convert an image in to a row vector
     * @param image the image to convert
     * @return the row vector based on a rastered
     * representation of the image
     */
    public INDArray asRowVector(BufferedImage image) {
        if (centerCropIfNeeded) {
            image = centerCropIfNeeded(image);
        }
        image = scalingIfNeed(image, true);
        if (channels == 3) {
            return toINDArrayBGR(image).ravel();
        }
        int[][] ret = toIntArrayArray(image);
        return NDArrayUtil.toNDArray(ArrayUtil.flatten(ret));
    }

    /**
     * Changes the input stream in to an
     * bgr based raveled(flattened) vector
     * @param file the input stream to convert
     * @return  the raveled bgr values for this input stream
     */
    public INDArray toRaveledTensor(File file) {
        try {
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file));
            INDArray ret = toRaveledTensor(bis);
            bis.close();
            return ret.ravel();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Changes the input stream in to an
     * bgr based raveled(flattened) vector
     * @param is the input stream to convert
     * @return  the raveled bgr values for this input stream
     */
    public INDArray toRaveledTensor(InputStream is) {
        return toBgr(is).ravel();
    }

    /**
     * Convert an image in to a raveled tensor of
     * the bgr values of the image
     * @param image the image to parse
     * @return the raveled tensor of bgr values
     */
    public INDArray toRaveledTensor(BufferedImage image) {
        try {
            image = scalingIfNeed(image, false);
            return toINDArrayBGR(image).ravel();
        } catch (Exception e) {
            throw new RuntimeException("Unable to load image", e);
        }
    }

    /**
     * Convert an input stream to an bgr spectrum image
     *
     * @param file the file to convert
     * @return the input stream to convert
     */
    public INDArray toBgr(File file) {
        try {
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file));
            INDArray ret = toBgr(bis);
            bis.close();
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Convert an input stream to an bgr spectrum image
     *
     * @param inputStream the input stream to convert
     * @return the input stream to convert
     */
    public INDArray toBgr(InputStream inputStream) {
        try {
            BufferedImage image = ImageIO.read(inputStream);
            return toBgr(image);
        } catch (IOException e) {
            throw new RuntimeException("Unable to load image", e);
        }
    }

    private org.datavec.image.data.Image toBgrImage(InputStream inputStream){
        try {
            BufferedImage image = ImageIO.read(inputStream);
            INDArray img = toBgr(image);
            return new org.datavec.image.data.Image(img, image.getData().getNumBands(), image.getHeight(), image.getWidth());
        } catch (IOException e) {
            throw new RuntimeException("Unable to load image", e);
        }
    }

    /**
     * Convert an BufferedImage to an bgr spectrum image
     *
     * @param image the BufferedImage to convert
     * @return the input stream to convert
     */
    public INDArray toBgr(BufferedImage image) {
        if (image == null)
            throw new IllegalStateException("Unable to load image");
        image = scalingIfNeed(image, false);
        return toINDArrayBGR(image);
    }

    /**
     * Convert an image file
     * in to a matrix
     * @param f the file to convert
     * @return a 2d matrix of a rastered version of the image
     * @throws IOException
     */
    public INDArray asMatrix(File f) throws IOException {
        return NDArrayUtil.toNDArray(fromFile(f));
    }

    /**
     * Convert an input stream to a matrix
     * @param inputStream the input stream to convert
     * @return the input stream to convert
     */
    public INDArray asMatrix(InputStream inputStream) throws IOException {
        if (channels == 3)
            return toBgr(inputStream);
        try {
            BufferedImage image = ImageIO.read(inputStream);
            return asMatrix(image);
        } catch (IOException e) {
            throw new IOException("Unable to load image", e);
        }
    }

    @Override
    public org.datavec.image.data.Image asImageMatrix(File f) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f))) {
            return asImageMatrix(bis);
        }
    }

    @Override
    public org.datavec.image.data.Image asImageMatrix(InputStream inputStream) throws IOException {
        if (channels == 3)
            return toBgrImage(inputStream);
        try {
            BufferedImage image = ImageIO.read(inputStream);
            INDArray asMatrix = asMatrix(image);
            return new org.datavec.image.data.Image(asMatrix, image.getData().getNumBands(), image.getHeight(), image.getWidth());
        } catch (IOException e) {
            throw new IOException("Unable to load image", e);
        }
    }

    /**
     * Convert an BufferedImage to a matrix
     * @param image the BufferedImage to convert
     * @return the input stream to convert
     */
    public INDArray asMatrix(BufferedImage image) {
        if (channels == 3) {
            return toBgr(image);
        } else {
            image = scalingIfNeed(image, true);
            int w = image.getWidth();
            int h = image.getHeight();
            INDArray ret = Nd4j.create(h, w);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    ret.putScalar(new int[] {i, j}, image.getRGB(j, i));
                }
            }
            return ret;
        }
    }

    /**
     * Slices up an image in to a mini batch.
     *
     * @param f the file to load from
     * @param numMiniBatches the number of images in a mini batch
     * @param numRowsPerSlice the number of rows for each image
     * @return a tensor representing one image as a mini batch
     */
    public INDArray asImageMiniBatches(File f, int numMiniBatches, int numRowsPerSlice) {
        try {
            INDArray d = asMatrix(f);
            return Nd4j.create(numMiniBatches, numRowsPerSlice, d.columns());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public int[] flattenedImageFromFile(File f) throws IOException {
        return ArrayUtil.flatten(fromFile(f));
    }

    /**
     * Load a rastered image from file
     * @param file the file to load
     * @return the rastered image
     * @throws IOException
     */
    public int[][] fromFile(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        image = scalingIfNeed(image, true);
        return toIntArrayArray(image);
    }

    /**
     * Load a rastered image from file
     * @param file the file to load
     * @return the rastered image
     * @throws IOException
     */
    public int[][][] fromFileMultipleChannels(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        image = scalingIfNeed(image, channels > 3);

        int w = image.getWidth(), h = image.getHeight();
        int bands = image.getSampleModel().getNumBands();
        int[][][] ret = new int[(int)Math.min(channels, Integer.MAX_VALUE)]
                               [(int)Math.min(h, Integer.MAX_VALUE)]
                               [(int)Math.min(w, Integer.MAX_VALUE)];
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < channels; k++) {
                    if (k >= bands)
                        break;
                    ret[k][i][j] = pixels[(int)Math.min(channels * w * i + channels * j + k, Integer.MAX_VALUE)];
                }
            }
        }
        return ret;
    }

    /**
     * Convert a matrix in to a buffereed image
     * @param matrix the
     * @return {@link java.awt.image.BufferedImage}
     */
    public static BufferedImage toImage(INDArray matrix) {
        BufferedImage img = new BufferedImage(matrix.rows(), matrix.columns(), BufferedImage.TYPE_INT_ARGB);
        WritableRaster r = img.getRaster();
        int[] equiv = new int[(int) matrix.length()];
        for (int i = 0; i < equiv.length; i++) {
            equiv[i] = (int) matrix.getDouble(i);
        }

        r.setDataElements(0, 0, matrix.rows(), matrix.columns(), equiv);
        return img;
    }


    private static int[] rasterData(INDArray matrix) {
        int[] ret = new int[(int) matrix.length()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = (int) Math.round((double) matrix.getScalar(i).element());
        return ret;
    }

    /**
     * Convert the given image to an rgb image
     * @param arr the array to use
     * @param image the image to set
     */
    public void toBufferedImageRGB(INDArray arr, BufferedImage image) {
        if (arr.rank() < 3)
            throw new IllegalArgumentException("Arr must be 3d");

        image = scalingIfNeed(image, arr.size(-2), arr.size(-1), true);
        for (int i = 0; i < image.getHeight(); i++) {
            for (int j = 0; j < image.getWidth(); j++) {
                int r = arr.slice(2).getInt(i, j);
                int g = arr.slice(1).getInt(i, j);
                int b = arr.slice(0).getInt(i, j);
                int a = 1;
                int col = (a << 24) | (r << 16) | (g << 8) | b;
                image.setRGB(j, i, col);
            }
        }
    }

    /**
     * Converts a given Image into a BufferedImage
     *
     * @param img The Image to be converted
     * @param type The color model of BufferedImage
     * @return The converted BufferedImage
     */
    public static BufferedImage toBufferedImage(Image img, int type) {
        if (img instanceof BufferedImage) {
            return (BufferedImage) img;
        }

        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), type);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

    protected int[][] toIntArrayArray(BufferedImage image) {
        int w = image.getWidth(), h = image.getHeight();
        int[][] ret = new int[h][w];
        if (image.getRaster().getNumDataElements() == 1) {
            Raster raster = image.getRaster();
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    ret[i][j] = raster.getSample(j, i, 0);
                }
            }
        } else {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    ret[i][j] = image.getRGB(j, i);
                }
            }
        }
        return ret;
    }

    protected INDArray toINDArrayBGR(BufferedImage image) {
        int height = image.getHeight();
        int width = image.getWidth();
        int bands = image.getSampleModel().getNumBands();

        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        int[] shape = new int[] {height, width, bands};

        INDArray ret2 = Nd4j.create(1, pixels.length);
        for (int i = 0; i < ret2.length(); i++) {
            ret2.putScalar(i, ((int) pixels[i]) & 0xFF);
        }
        return ret2.reshape(shape).permute(2, 0, 1);
    }

    // TODO build flexibility on where to crop the image
    public BufferedImage centerCropIfNeeded(BufferedImage img) {
        int x = 0;
        int y = 0;
        int height = img.getHeight();
        int width = img.getWidth();
        int diff = Math.abs(width - height) / 2;

        if (width > height) {
            x = diff;
            width = width - diff;
        } else if (height > width) {
            y = diff;
            height = height - diff;
        }
        return img.getSubimage(x, y, width, height);
    }

    protected BufferedImage scalingIfNeed(BufferedImage image, boolean needAlpha) {
        return scalingIfNeed(image, height, width, needAlpha);
    }

    protected BufferedImage scalingIfNeed(BufferedImage image, long dstHeight, long dstWidth, boolean needAlpha) {
        if (dstHeight > 0 && dstWidth > 0 && (image.getHeight() != dstHeight || image.getWidth() != dstWidth)) {
            Image scaled = image.getScaledInstance((int) dstWidth, (int) dstHeight, Image.SCALE_SMOOTH);

            if (needAlpha && image.getColorModel().hasAlpha() && channels == BufferedImage.TYPE_4BYTE_ABGR) {
                return toBufferedImage(scaled, BufferedImage.TYPE_4BYTE_ABGR);
            } else {
                if (channels == BufferedImage.TYPE_BYTE_GRAY)
                    return toBufferedImage(scaled, BufferedImage.TYPE_BYTE_GRAY);
                else
                    return toBufferedImage(scaled, BufferedImage.TYPE_3BYTE_BGR);
            }
        } else {
            if (image.getType() == BufferedImage.TYPE_4BYTE_ABGR || image.getType() == BufferedImage.TYPE_3BYTE_BGR) {
                return image;
            } else if (needAlpha && image.getColorModel().hasAlpha() && channels == BufferedImage.TYPE_4BYTE_ABGR) {
                return toBufferedImage(image, BufferedImage.TYPE_4BYTE_ABGR);
            } else {
                if (channels == BufferedImage.TYPE_BYTE_GRAY)
                    return toBufferedImage(image, BufferedImage.TYPE_BYTE_GRAY);
                else
                    return toBufferedImage(image, BufferedImage.TYPE_3BYTE_BGR);
            }
        }
    }

}
