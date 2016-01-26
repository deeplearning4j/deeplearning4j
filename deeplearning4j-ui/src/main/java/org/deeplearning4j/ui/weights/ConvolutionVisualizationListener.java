package org.deeplearning4j.ui.weights;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class ConvolutionVisualizationListener implements IterationListener {
    private int freq = 10;
    private static final Logger log = LoggerFactory.getLogger(ConvolutionVisualizationListener.class);
    private int minibatchNum = 0;

    public ConvolutionVisualizationListener() {

    }

    public ConvolutionVisualizationListener(int visualizationFrequency) {
        this.freq = visualizationFrequency;
    }

    /**
     * Get if listener invoked
     */
    @Override
    public boolean invoked() {
        return false;
    }

    /**
     * Change invoke to true
     */
    @Override
    public void invoke() {

    }

    /**
     * Event listener for each iteration
     *
     * @param model     the model iterating
     * @param iteration the iteration number
     */
    @Override
    public void iterationDone(Model model, int iteration) {
        if (iteration == 0) {

            List<INDArray> tensors = new ArrayList<>();
            int cnt = 0;
            MultiLayerNetwork l = (MultiLayerNetwork) model;
            for (Layer layer: l.getLayers()) {
                if (layer.type() == Layer.Type.CONVOLUTIONAL) {
                    INDArray output = layer.activate();

                    log.info("Layer output shape: " + Arrays.toString(output.shape()));

                    INDArray tad = output.tensorAlongDimension(21, 3, 2, 1);
                    log.info("TAD(3,2,1) shape: " + Arrays.toString(tad.shape()));

                    tensors.add(tad);

                    cnt++;
                }
            }
            BufferedImage render = rasterizeConvoLayers(tensors);
            try {
                ImageIO.write(render, "png", new File("render_" + minibatchNum + ".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }
            minibatchNum++;

        }
    }

    /**
     * We visualize set of tensors as vertically aligned set of patches
     *
     * @param tensors3D list of tensors retrieved from convolution
     */
    private BufferedImage rasterizeConvoLayers(@NonNull List<INDArray> tensors3D) {
        int border = 1;
        int padding_row = 2;
        int padding_col = 40;
        int width = 0;
        int height = 0;

        /*
            We determine height of joint output image. We assume that first position holds maximum dimensionality
         */
        int[] shape = tensors3D.get(0).shape();
        height = (shape[1]) + (border * 2) + padding_row;
        width = (shape[2]) + (border * 2) + padding_col;
        log.info("Output image dimensions: {height: " + height + ", width: " + width + "}");

        BufferedImage output = new BufferedImage(width * tensors3D.size() + ((tensors3D.size() - 1) * width), height * shape[0], BufferedImage.TYPE_BYTE_GRAY);

        Graphics2D graphics2D = output.createGraphics();

        for (int layer = 0; layer < tensors3D.size(); layer++) {
            INDArray tad = tensors3D.get(layer);
            int zoomed = 0;
            for (int z = 0; z < tad.shape()[0]; z++) {

                INDArray tad2D = tad.tensorAlongDimension(z, 2, 1);
                int loc_height = (tad2D.shape()[0]) + (border * 2) + padding_row;
                int loc_width = (tad2D.shape()[1]) + (border * 2) + padding_col;

                BufferedImage currentImage = renderImageGrayscale(tad2D);

                /*
                    now we should place this image into output image
                */

                try {
                    ImageIO.write(currentImage, "png", new File("tmp/image_l" + layer + "_z" + z +".png"));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                graphics2D.drawImage(currentImage, layer * width + 1, z * loc_height + 1, null);

                /*
                    draw borders around each image
                */

                graphics2D.drawRect(layer* width, z* loc_height, tad2D.shape()[1], tad2D.shape()[0]);

                /*
                    draw one of 3 zoomed images if we're not on first level
                */
                if (z % 5 == 0 && // zoom each 5th element
                    loc_height != height && loc_width != width && // that's not already equal to size of first convolutional layer
                    z != 0 && // do not zoom 0 element
                    z < tad.shape()[0] - 10 // do not zoom anything too close to bottom of output image
                    ) {
                    log.info("Vis layer: " + z);
                    graphics2D.drawImage(currentImage, layer * width + padding_col, z * loc_height + 1, shape[2], shape[1], null);
                    graphics2D.drawRect(layer * width + padding_col, z * loc_height, shape[2], shape[1]);

                    // draw line to connect this zoomed pic with its original
                    graphics2D.drawLine(layer * width + tad2D.shape()[1], z * loc_height + tad2D.shape()[0] , layer * width + padding_col, z * loc_height + shape[1] );
                }
            }
        }

        return output;
    }


    /**
     * Renders 2D INDArray into BufferedImage
     *
     * @param array
     */
    private BufferedImage renderImageGrayscale(INDArray array) {
        BufferedImage imageToRender = new BufferedImage(array.columns(),array.rows(),BufferedImage.TYPE_BYTE_GRAY);
        for( int x = 0; x < array.columns(); x++ ){
            for (int y = 0; y < array.rows(); y++ ) {
                imageToRender.getRaster().setSample(x, y, 0, (int) (255 * array.getRow(y).getDouble(x)));
            }
        }

        return imageToRender;
    }

    private void writeImageGrayscale(INDArray array, File file) {
        try {
            ImageIO.write(renderImageGrayscale(array), "png", file);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void writeImage(INDArray array, File file) {
        BufferedImage image = ImageLoader.toImage(array);
        try {
            ImageIO.write(image, "png", file);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void writeRows(INDArray array, File file) {
        try {
            PrintWriter writer = new PrintWriter(file);
            for (int x = 0; x < array.rows(); x++) {
                writer.println("Row [" + x + "]: " + array.getRow(x));
            }
            writer.flush();
            writer.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
