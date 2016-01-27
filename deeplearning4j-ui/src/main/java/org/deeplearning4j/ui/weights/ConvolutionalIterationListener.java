package org.deeplearning4j.ui.weights;

import lombok.NonNull;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.UiServer;
import org.deeplearning4j.ui.UiUtils;
import org.deeplearning4j.ui.activation.PathUpdate;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author raver119@gmail.com
 */
public class ConvolutionalIterationListener implements IterationListener {
    private enum Orientation {
        LANDSCAPE,
        PORTRAIT
    }
    private int freq = 10;
    private static final Logger log = LoggerFactory.getLogger(ConvolutionalIterationListener.class);
    private int minibatchNum = 0;
    private boolean openBrowser = true;
    private String path;
    private Client client = ClientBuilder.newClient();
    private WebTarget target;
    private boolean firstIteration = true;

    private Color borderColor = new Color(140,140,140);
    private Color bgColor = new Color(255,255,255);


    public ConvolutionalIterationListener(int visualizationFrequency) {
        this(visualizationFrequency, true, "activations");
    }

    public ConvolutionalIterationListener(int iterations, boolean openBrowser, String subPath){
        int port = -1;
        try{
            UiServer server = UiServer.getInstance();
            port = server.getPort();
        }catch(Exception e){
            log.error("Error initializing UI server",e);
            throw new RuntimeException(e);
        }

        this.freq = iterations;
        this.openBrowser = openBrowser;
        path = "http://localhost:" + port + "/" + subPath;
        target = client.target("http://localhost:" + port).path(subPath).path("update");
        try{
            UiServer.getInstance();
        }catch(Exception e){
            log.error("Error initializing UI server",e);
        }
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
        if (iteration % freq == 0) {

            List<INDArray> tensors = new ArrayList<>();
            int cnt = 0;
            Random rnd = new Random();
            MultiLayerNetwork l = (MultiLayerNetwork) model;
            for (Layer layer: l.getLayers()) {
                if (layer.type() == Layer.Type.CONVOLUTIONAL) {
                    INDArray output = layer.activate();

//                    log.info("Layer output shape: " + Arrays.toString(output.shape()));

                    INDArray tad = output.tensorAlongDimension(rnd.nextInt(output.shape()[0] - 1) + 1, 3, 2, 1);
  //                  log.info("TAD(3,2,1) shape: " + Arrays.toString(tad.shape()));

                    tensors.add(tad);

                    cnt++;
                }
            }
            BufferedImage render = rasterizeConvoLayers(tensors);
            try {
                File tempFile = File.createTempFile("cnn_activations",".png");
                tempFile.deleteOnExit();

                ImageIO.write(render, "png", tempFile);

                PathUpdate update = new PathUpdate();
                //ensure path is set
                update.setPath(tempFile.getPath());
                //ensure the server is hooked up with the path
                target.request(MediaType.APPLICATION_JSON).post(Entity.entity(update, MediaType.APPLICATION_JSON));
                if(openBrowser && firstIteration){
                    UiUtils.tryOpenBrowser(path, log);
                    firstIteration = false;
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
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
        int width = 0;
        int height = 0;

        int border = 1;
        int padding_row = 2;
        int padding_col = 80;

        /*
            We determine height of joint output image. We assume that first position holds maximum dimensionality
         */
        int[] shape = tensors3D.get(0).shape();
        int numImages = shape[0];
        height = (shape[2]);
        width = (shape[1]);
        log.info("Output image dimensions: {height: " + height + ", width: " + width + "}");
        int maxHeight = (height + (border * 2 ) + padding_row) * numImages;
        int totalWidth = 0;
        int iOffset = 1;

        Orientation orientation = Orientation.PORTRAIT;

        List<BufferedImage> images = new ArrayList<>();
        for (int layer = 0; layer < tensors3D.size(); layer++) {
            INDArray tad = tensors3D.get(layer);
            int zoomed = 0;

            BufferedImage image = renderMultipleImages(tad,maxHeight, width, height, orientation);
            totalWidth += image.getWidth() + padding_col;
            images.add(image);
        }


        // append some space for arrows
        totalWidth += padding_col  * 2;

        BufferedImage output = new BufferedImage(totalWidth, maxHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = output.createGraphics();

        graphics2D.setPaint(bgColor);
        graphics2D.fillRect(0, 0, output.getWidth(), output.getHeight());

        BufferedImage singleArrow = null;
        BufferedImage multipleArrows = null;

        /*
            We try to add nice flow arrow here
         */
        try {
            ClassPathResource resource = new ClassPathResource("/arrow_sing.PNG");
            ClassPathResource resource2 = new ClassPathResource("/arrow_mul.PNG");

            singleArrow = ImageIO.read(resource.getInputStream());
            multipleArrows = ImageIO.read(resource2.getInputStream());
            if (orientation == Orientation.PORTRAIT) {
                // if orientation is portrait, we should rotate arrows
                singleArrow.createGraphics().rotate(90.0);
                multipleArrows.createGraphics().rotate(90.0);
            }

                graphics2D.drawImage(singleArrow, (padding_col / 2) - (singleArrow.getWidth() / 2), (maxHeight / 2) - (singleArrow.getHeight() / 2), null);
            iOffset += padding_col;
        } catch (Exception e) {
            // if we can't load images - ignore them
            ;
        }



        /*
            now we merge all images into one big image with some offset
        */


        for (int i = 0; i < images.size(); i++) {
            BufferedImage curImage = images.get(i);
            if (orientation == Orientation.LANDSCAPE) {
                // image grows from left to right
                graphics2D.drawImage(curImage, iOffset, 1, null);
                iOffset += curImage.getWidth() + padding_col;

                if (singleArrow != null && multipleArrows != null) {
                    if (i < images.size() - 1) {
                        // draw multiple arrows here
                        graphics2D.drawImage(multipleArrows, iOffset - (padding_col / 2) - (multipleArrows.getWidth() / 2), (maxHeight / 2) - (multipleArrows.getHeight() / 2), null);
                    } else {
                        // draw single arrow
                        graphics2D.drawImage(singleArrow, iOffset - (padding_col / 2) - (singleArrow.getWidth() / 2), (maxHeight / 2) - (singleArrow.getHeight() / 2), null);
                    }
                }
            } else if (orientation == Orientation.PORTRAIT) {
                // image grows from top to bottom
                graphics2D.drawImage(curImage, 1, iOffset, null);
                iOffset += curImage.getWidth() + padding_col;

                if (singleArrow != null && multipleArrows != null) {
                    if (i < images.size() - 1) {
                        // draw multiple arrows here
                        graphics2D.drawImage(multipleArrows, iOffset - (padding_col / 2) - (multipleArrows.getWidth() / 2), (maxHeight / 2) - (multipleArrows.getHeight() / 2), null);
                    } else {
                        // draw single arrow
                        graphics2D.drawImage(singleArrow, iOffset - (padding_col / 2) - (singleArrow.getWidth() / 2), (maxHeight / 2) - (singleArrow.getHeight() / 2), null);
                    }
                }
            }
        }

        return output;
    }

    /**
     * This method renders 1 convolution layer as set of patches + multiple zoomed images
     * @param tensor3D
     * @return
     */
    private BufferedImage renderMultipleImages(INDArray tensor3D, int maxHeight, int zoomWidth, int zoomHeight, Orientation orientation) {
        /*
            first we need to determine, weight of output image.
         */
        int border = 1;
        int padding_row = 2;
        int padding_col = 2;
        int zoomPadding = 20;

        int[] tShape = tensor3D.shape();

        int numColumns = tShape[0] / tShape[2];

        int width = (numColumns * (tShape[1] + border + padding_col)) + padding_col + zoomPadding + zoomWidth;


        List<Pair<Pair<Integer, Integer>, BufferedImage>> coords = new ArrayList<>();


        BufferedImage outputImage = new BufferedImage(width, maxHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = outputImage.createGraphics();

        graphics2D.setPaint(bgColor);
        graphics2D.fillRect(0, 0, outputImage.getWidth(), outputImage.getHeight());

        int columnOffset = 0;
        int rowOffset = 0;
        int numZoomed = 0;
        int limZoomed = 5;
        int zoomSpan = maxHeight / limZoomed;
        for (int z = 0; z < tensor3D.shape()[0]; z++) {

            INDArray tad2D = tensor3D.tensorAlongDimension(z, 2, 1);

            int rWidth = tad2D.shape()[0];
            int rHeight = tad2D.shape()[1];

            int loc_height = (rHeight) + (border * 2) + padding_row;
            int loc_width = (rWidth) + (border * 2) + padding_col;



            BufferedImage currentImage = renderImageGrayscale(tad2D);

            /*
                if resulting image doesn't fit into image, we should step to next columns
             */
            if (rowOffset + loc_height > maxHeight) {
                columnOffset += loc_width;
                rowOffset = 0;
            }

            /*
                now we should place this image into output image
            */

            graphics2D.drawImage(currentImage, columnOffset+1, rowOffset + 1, null);

            /*
                we save coords for each image, for use with zoom
             */
            coords.add(Pair.makePair(Pair.makePair(columnOffset, rowOffset), currentImage));

            /*
                draw borders around each image
            */

            graphics2D.setPaint(borderColor);
            graphics2D.drawRect(columnOffset, rowOffset, tad2D.shape()[0], tad2D.shape()[1]);



            /*
                draw one of 3 zoomed images if we're not on first level
            */

            if (z % 5 == 0 && // zoom each 5th element
                    z != 0 && // do not zoom 0 element
                    numZoomed < limZoomed && // we want only few zoomed samples
                    (rHeight != zoomHeight && rWidth != zoomWidth ) // do not zoom if dimensions match
                    ) {

                int cY = (zoomSpan * numZoomed) + (zoomHeight);

                graphics2D.drawImage(currentImage, width - zoomWidth -1 , cY - 1, zoomWidth, zoomHeight, null);
                graphics2D.drawRect(width - zoomWidth -2, cY -2, zoomWidth, zoomHeight);

                // draw line to connect this zoomed pic with its original
                graphics2D.drawLine(columnOffset + rWidth, rowOffset + rHeight, width - zoomWidth -2, cY - 2 + zoomHeight );
                numZoomed++;
            }

            rowOffset += loc_height;
        }
        return outputImage;
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
