package org.deeplearning4j.util;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.deeplearning4j.nn.FourDTensor;
import org.deeplearning4j.nn.Tensor;
import org.jblas.DoubleMatrix;
/**
 * Image loader for taking images and converting them to matrices
 * @author Adam Gibson
 *
 */
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

    public DoubleMatrix asRowVector(File f) throws Exception {
        return MatrixUtil.toMatrix(flattenedImageFromFile(f));
    }


    /**
     * Slices up an image in to a mini batch.
     *
     * @param f the file to load from
     * @param numMiniBatches the number of images in a mini batch
     * @param numRowsPerSlice the number of rows for each image
     * @return a tensor representing one image as a mini batch
     */
    public Tensor asImageMiniBatches(File f,int numMiniBatches,int numRowsPerSlice) {
        try {
            DoubleMatrix d = asMatrix(f);
            Tensor f2 = new Tensor(d,numRowsPerSlice,d.columns,numMiniBatches);
            return f2;
        }catch(Exception e) {
            throw new RuntimeException(e);
        }

    }

    public DoubleMatrix asMatrix(File f) throws IOException {
        return MatrixUtil.toMatrix(fromFile(f));
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


    public static BufferedImage toImage(DoubleMatrix matrix) {
        BufferedImage img = new BufferedImage(matrix.rows, matrix.columns, BufferedImage.TYPE_INT_ARGB);
        WritableRaster r = img.getRaster();
        int[] equiv = new int[matrix.length];
        for(int i = 0; i < equiv.length; i++) {
            equiv[i] = (int) matrix.get(i);
        }


        r.setDataElements(0,0,matrix.rows,matrix.columns,equiv);
        return img;
    }


    private static int[] rasterData(DoubleMatrix matrix) {
        int[] ret = new int[matrix.length];
        for(int i = 0; i < ret.length; i++)
            ret[i] = (int) Math.round(matrix.get(i));
        return ret;
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
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

}
