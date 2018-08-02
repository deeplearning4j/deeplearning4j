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

package org.deeplearning4j.ui.weights;

import lombok.NonNull;
import lombok.val;
import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.ui.UiConnectionInfo;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;

/**
 * @author raver119@gmail.com
 */
public class ConvolutionalIterationListener extends BaseTrainingListener {

    private enum Orientation {
        LANDSCAPE, PORTRAIT
    }

    private int freq = 10;
    private static final Logger log = LoggerFactory.getLogger(ConvolutionalIterationListener.class);
    private int minibatchNum = 0;
    private boolean openBrowser = true;
    private String path;
    private boolean firstIteration = true;

    private Color borderColor = new Color(140, 140, 140);
    private Color bgColor = new Color(255, 255, 255);

    private final StatsStorageRouter ssr;
    private final String sessionID;
    private final String workerID;


    public ConvolutionalIterationListener(UiConnectionInfo connectionInfo, int visualizationFrequency) {
        this(new MapDBStatsStorage(), visualizationFrequency, true);
    }

    public ConvolutionalIterationListener(int visualizationFrequency) {
        this(visualizationFrequency, true);
    }

    public ConvolutionalIterationListener(int iterations, boolean openBrowser) {
        this(new MapDBStatsStorage(), iterations, openBrowser);
    }

    public ConvolutionalIterationListener(StatsStorageRouter ssr, int iterations, boolean openBrowser) {
        this(ssr, iterations, openBrowser, null, null);
    }

    public ConvolutionalIterationListener(StatsStorageRouter ssr, int iterations, boolean openBrowser, String sessionID,
                    String workerID) {
        this.ssr = ssr;
        if (sessionID == null) {
            //TODO handle syncing session IDs across different listeners in the same model...
            this.sessionID = UUID.randomUUID().toString();
        } else {
            this.sessionID = sessionID;
        }
        if (workerID == null) {
            this.workerID = UIDProvider.getJVMUID() + "_" + Thread.currentThread().getId();
        } else {
            this.workerID = workerID;
        }

        String subPath = "activations";

        this.freq = iterations;
        this.openBrowser = openBrowser;
        path = "http://localhost:" + UIServer.getInstance().getPort() + "/" + subPath;

        if (openBrowser && ssr instanceof StatsStorage) {
            UIServer.getInstance().attach((StatsStorage) ssr);
        }

        System.out.println("ConvolutionTrainingListener path: " + path);
    }

    /**
     * Event listener for each iteration
     *
     * @param model     the model iterating
     * @param iteration the iteration number
     */
    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (iteration % freq == 0) {

            List<INDArray> tensors = new ArrayList<>();
            int cnt = 0;
            Random rnd = new Random();
            BufferedImage sourceImage = null;
            if (model instanceof MultiLayerNetwork) {
                MultiLayerNetwork l = (MultiLayerNetwork) model;
                for (Layer layer : l.getLayers()) {
                    if (layer.type() == Layer.Type.CONVOLUTIONAL) {
                        INDArray output = layer.activate(layer.input(), true, LayerWorkspaceMgr.noWorkspaces());
                        // FIXME: int cast
                        int sampleDim = output.shape()[0] == 1 ? 0 : rnd.nextInt((int) output.shape()[0] - 1) + 1;
                        if (cnt == 0) {
                            INDArray inputs = layer.input();

                            try {
                                sourceImage = restoreRGBImage(
                                                inputs.tensorAlongDimension(sampleDim, new int[] {3, 2, 1}));
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        }

                        INDArray tad = output.tensorAlongDimension(sampleDim, 3, 2, 1);

                        tensors.add(tad);

                        cnt++;
                    }
                }

            } else if (model instanceof ComputationGraph) {
                ComputationGraph l = (ComputationGraph) model;
                for (Layer layer : l.getLayers()) {
                    if (layer.type() == Layer.Type.CONVOLUTIONAL) {
                        INDArray output = layer.activate(layer.input(), true, LayerWorkspaceMgr.noWorkspaces());

                        // FIXME: int cast
                        int sampleDim = output.shape()[0] == 1 ? 0 : rnd.nextInt((int) output.shape()[0] - 1) + 1;
                        if (cnt == 0) {
                            INDArray inputs = layer.input();

                            try {
                                sourceImage = restoreRGBImage(
                                                inputs.tensorAlongDimension(sampleDim, new int[] {3, 2, 1}));
                            } catch (Exception e) {
                                throw new RuntimeException(e);
                            }
                        }

                        INDArray tad = output.tensorAlongDimension(sampleDim, 3, 2, 1);

                        tensors.add(tad);

                        cnt++;
                    }
                }
            }
            BufferedImage render = rasterizeConvoLayers(tensors, sourceImage);
            Persistable p = new ConvolutionListenerPersistable(sessionID, workerID, System.currentTimeMillis(), render);
            ssr.putStaticInfo(p);

            minibatchNum++;

        }
    }

    /**
     * We visualize set of tensors as vertically aligned set of patches
     *
     * @param tensors3D list of tensors retrieved from convolution
     */
    private BufferedImage rasterizeConvoLayers(@NonNull List<INDArray> tensors3D, BufferedImage sourceImage) {
        long width = 0;
        long height = 0;

        int border = 1;
        int padding_row = 2;
        int padding_col = 80;

        /*
            We determine height of joint output image. We assume that first position holds maximum dimensionality
         */
        val shape = tensors3D.get(0).shape();
        val numImages = shape[0];
        height = (shape[2]);
        width = (shape[1]);
        //        log.info("Output image dimensions: {height: " + height + ", width: " + width + "}");
        int maxHeight = 0; //(height + (border * 2 ) + padding_row) * numImages;
        int totalWidth = 0;
        int iOffset = 1;

        Orientation orientation = Orientation.LANDSCAPE;
        /*
            for debug purposes we'll use portait only now
         */
        if (tensors3D.size() > 3) {
            orientation = Orientation.PORTRAIT;
        }



        List<BufferedImage> images = new ArrayList<>();
        for (int layer = 0; layer < tensors3D.size(); layer++) {
            INDArray tad = tensors3D.get(layer);
            int zoomed = 0;

            BufferedImage image = null;
            if (orientation == Orientation.LANDSCAPE) {
                maxHeight = (int) ((height + (border * 2) + padding_row) * numImages);
                image = renderMultipleImagesLandscape(tad, maxHeight, (int) width, (int) height);
                totalWidth += image.getWidth() + padding_col;
            } else if (orientation == Orientation.PORTRAIT) {
                totalWidth = (int) ((width + (border * 2) + padding_row) * numImages);
                image = renderMultipleImagesPortrait(tad, totalWidth, (int) width, (int) height);
                maxHeight += image.getHeight() + padding_col;
            }

            images.add(image);
        }

        if (orientation == Orientation.LANDSCAPE) {
            // append some space for arrows
            totalWidth += padding_col * 2;
        } else if (orientation == Orientation.PORTRAIT) {
            maxHeight += padding_col * 2;
            maxHeight += sourceImage.getHeight() + (padding_col * 2);
        }

        BufferedImage output = new BufferedImage(totalWidth, maxHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = output.createGraphics();

        graphics2D.setPaint(bgColor);
        graphics2D.fillRect(0, 0, output.getWidth(), output.getHeight());

        BufferedImage singleArrow = null;
        BufferedImage multipleArrows = null;

        /*
            We try to add nice flow arrow here
         */
        try {

            if (orientation == Orientation.LANDSCAPE) {
                try {
                    ClassPathResource resource = new ClassPathResource("arrow_sing.PNG");
                    ClassPathResource resource2 = new ClassPathResource("arrow_mul.PNG");

                    singleArrow = ImageIO.read(resource.getInputStream());
                    multipleArrows = ImageIO.read(resource2.getInputStream());
                } catch (Exception e) {
                }

                graphics2D.drawImage(sourceImage, (padding_col / 2) - (sourceImage.getWidth() / 2),
                                (maxHeight / 2) - (sourceImage.getHeight() / 2), null);

                graphics2D.setPaint(borderColor);
                graphics2D.drawRect((padding_col / 2) - (sourceImage.getWidth() / 2),
                                (maxHeight / 2) - (sourceImage.getHeight() / 2), sourceImage.getWidth(),
                                sourceImage.getHeight());

                iOffset += sourceImage.getWidth();

                if (singleArrow != null)
                    graphics2D.drawImage(singleArrow, iOffset + (padding_col / 2) - (singleArrow.getWidth() / 2),
                                    (maxHeight / 2) - (singleArrow.getHeight() / 2), null);
            } else {
                try {
                    ClassPathResource resource = new ClassPathResource("arrow_singi.PNG");
                    ClassPathResource resource2 = new ClassPathResource("arrow_muli.PNG");

                    singleArrow = ImageIO.read(resource.getInputStream());
                    multipleArrows = ImageIO.read(resource2.getInputStream());
                } catch (Exception e) {
                }

                graphics2D.drawImage(sourceImage, (totalWidth / 2) - (sourceImage.getWidth() / 2),
                                (padding_col / 2) - (sourceImage.getHeight() / 2), null);

                graphics2D.setPaint(borderColor);
                graphics2D.drawRect((totalWidth / 2) - (sourceImage.getWidth() / 2),
                                (padding_col / 2) - (sourceImage.getHeight() / 2), sourceImage.getWidth(),
                                sourceImage.getHeight());

                iOffset += sourceImage.getHeight();
                if (singleArrow != null)
                    graphics2D.drawImage(singleArrow, (totalWidth / 2) - (singleArrow.getWidth() / 2),
                                    iOffset + (padding_col / 2) - (singleArrow.getHeight() / 2), null);

            }
            iOffset += padding_col;
        } catch (Exception e) {
            // if we can't load images - ignore them
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
                        if (multipleArrows != null)
                            graphics2D.drawImage(multipleArrows,
                                            iOffset - (padding_col / 2) - (multipleArrows.getWidth() / 2),
                                            (maxHeight / 2) - (multipleArrows.getHeight() / 2), null);
                    } else {
                        // draw single arrow
                        //    graphics2D.drawImage(singleArrow, iOffset - (padding_col / 2) - (singleArrow.getWidth() / 2), (maxHeight / 2) - (singleArrow.getHeight() / 2), null);
                    }
                }
            } else if (orientation == Orientation.PORTRAIT) {
                // image grows from top to bottom
                graphics2D.drawImage(curImage, 1, iOffset, null);
                iOffset += curImage.getHeight() + padding_col;

                if (singleArrow != null && multipleArrows != null) {
                    if (i < images.size() - 1) {
                        // draw multiple arrows here
                        if (multipleArrows != null)
                            graphics2D.drawImage(multipleArrows, (totalWidth / 2) - (multipleArrows.getWidth() / 2),
                                            iOffset - (padding_col / 2) - (multipleArrows.getHeight() / 2), null);
                    } else {
                        // draw single arrow
                        //   graphics2D.drawImage(singleArrow, (totalWidth / 2) - (singleArrow.getWidth() / 2),  iOffset - (padding_col / 2) - (singleArrow.getHeight() / 2) , null);
                    }
                }
            }
        }

        return output;
    }


    private BufferedImage renderMultipleImagesPortrait(INDArray tensor3D, int maxWidth, int zoomWidth, int zoomHeight) {
        int border = 1;
        int padding_row = 2;
        int padding_col = 2;
        int zoomPadding = 20;

        val tShape = tensor3D.shape();

        val numRows = tShape[0] / tShape[2];

        val height = (numRows * (tShape[1] + border + padding_col)) + padding_col + zoomPadding + zoomWidth;

        // FIXME: int cast
        BufferedImage outputImage = new BufferedImage(maxWidth, (int) height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = outputImage.createGraphics();

        graphics2D.setPaint(bgColor);
        graphics2D.fillRect(0, 0, outputImage.getWidth(), outputImage.getHeight());

        int columnOffset = 0;
        int rowOffset = 0;
        int numZoomed = 0;
        int limZoomed = 5;
        int zoomSpan = maxWidth / limZoomed;

        for (int z = 0; z < tensor3D.shape()[0]; z++) {

            INDArray tad2D = tensor3D.tensorAlongDimension(z, 2, 1);

            val rWidth = tad2D.shape()[0];
            val rHeight = tad2D.shape()[1];

            val loc_height = (rHeight) + (border * 2) + padding_row;
            val loc_width = (rWidth) + (border * 2) + padding_col;



            BufferedImage currentImage = renderImageGrayscale(tad2D);

            /*
                if resulting image doesn't fit into image, we should step to next columns
             */
            if (columnOffset + loc_width > maxWidth) {
                rowOffset += loc_height;
                columnOffset = 0;
            }

            /*
                now we should place this image into output image
            */

            graphics2D.drawImage(currentImage, columnOffset + 1, rowOffset + 1, null);


            /*
                draw borders around each image
            */

            graphics2D.setPaint(borderColor);
            graphics2D.drawRect(columnOffset, rowOffset, (int) tad2D.shape()[0], (int) tad2D.shape()[1]);



            /*
                draw one of 3 zoomed images if we're not on first level
            */

            if (z % 7 == 0 && // zoom each 5th element
                            z != 0 && // do not zoom 0 element
                            numZoomed < limZoomed && // we want only few zoomed samples
                            (rHeight != zoomHeight && rWidth != zoomWidth) // do not zoom if dimensions match
            ) {

                int cY = (zoomSpan * numZoomed) + (zoomHeight);
                int cX = (zoomSpan * numZoomed) + (zoomWidth);

                graphics2D.drawImage(currentImage, cX - 1, (int) height - zoomWidth - 1, zoomWidth, zoomHeight, null);
                graphics2D.drawRect(cX - 2, (int) height - zoomWidth - 2, zoomWidth, zoomHeight);

                // draw line to connect this zoomed pic with its original
                graphics2D.drawLine(columnOffset + (int) rWidth, rowOffset + (int) rHeight, cX - 2, (int) height - zoomWidth - 2);
                numZoomed++;

            }

            columnOffset += loc_width;
        }

        return outputImage;
    }

    /**
     * This method renders 1 convolution layer as set of patches + multiple zoomed images
     * @param tensor3D
     * @return
     */
    private BufferedImage renderMultipleImagesLandscape(INDArray tensor3D, int maxHeight, int zoomWidth,
                    int zoomHeight) {
        /*
            first we need to determine, weight of output image.
         */
        int border = 1;
        int padding_row = 2;
        int padding_col = 2;
        int zoomPadding = 20;

        val tShape = tensor3D.shape();

        val numColumns = tShape[0] / tShape[1];

        val width = (numColumns * (tShape[1] + border + padding_col)) + padding_col + zoomPadding + zoomWidth;

        BufferedImage outputImage = new BufferedImage((int) width, maxHeight, BufferedImage.TYPE_BYTE_GRAY);
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

            val rWidth = tad2D.shape()[0];
            val rHeight = tad2D.shape()[1];

            val loc_height = (rHeight) + (border * 2) + padding_row;
            val loc_width = (rWidth) + (border * 2) + padding_col;



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

            graphics2D.drawImage(currentImage, columnOffset + 1, rowOffset + 1, null);


            /*
                draw borders around each image
            */

            graphics2D.setPaint(borderColor);
            // FIXME: int cast
            graphics2D.drawRect(columnOffset, rowOffset, (int) tad2D.shape()[0], (int) tad2D.shape()[1]);



            /*
                draw one of 3 zoomed images if we're not on first level
            */

            if (z % 5 == 0 && // zoom each 5th element
                            z != 0 && // do not zoom 0 element
                            numZoomed < limZoomed && // we want only few zoomed samples
                            (rHeight != zoomHeight && rWidth != zoomWidth) // do not zoom if dimensions match
            ) {

                int cY = (zoomSpan * numZoomed) + (zoomHeight);

                graphics2D.drawImage(currentImage, (int) width - zoomWidth - 1, cY - 1, zoomWidth, zoomHeight, null);
                graphics2D.drawRect((int) width - zoomWidth - 2, cY - 2, zoomWidth, zoomHeight);

                // draw line to connect this zoomed pic with its original
                graphics2D.drawLine(columnOffset + (int) rWidth, rowOffset + (int) rHeight, (int) width - zoomWidth - 2,
                                cY - 2 + zoomHeight);
                numZoomed++;
            }

            rowOffset += loc_height;
        }
        return outputImage;
    }

    /**
     * Returns RGB image out of 3D tensor
     *
     * @param tensor3D
     * @return
     */
    private BufferedImage restoreRGBImage(INDArray tensor3D) {
        INDArray arrayR = null;
        INDArray arrayG = null;
        INDArray arrayB = null;

        // entry for 3D input vis
        if (tensor3D.shape()[0] == 3) {
            arrayR = tensor3D.tensorAlongDimension(2, 2, 1);
            arrayG = tensor3D.tensorAlongDimension(1, 2, 1);
            arrayB = tensor3D.tensorAlongDimension(0, 2, 1);
        } else {
            // for all other cases input is just black & white, so we just assign the same channel data to RGB, and represent everything as RGB
            arrayB = tensor3D.tensorAlongDimension(0, 2, 1);
            arrayG = arrayB;
            arrayR = arrayB;
        }

        BufferedImage imageToRender = new BufferedImage(arrayR.columns(), arrayR.rows(), BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < arrayR.columns(); x++) {
            for (int y = 0; y < arrayR.rows(); y++) {
                Color pix = new Color((int) (255 * arrayR.getRow(y).getDouble(x)),
                                (int) (255 * arrayG.getRow(y).getDouble(x)),
                                (int) (255 * arrayB.getRow(y).getDouble(x)));
                int rgb = pix.getRGB();
                imageToRender.setRGB(x, y, rgb);
            }
        }
        return imageToRender;
    }

    /**
     * Renders 2D INDArray into BufferedImage
     *
     * @param array
     */
    private BufferedImage renderImageGrayscale(INDArray array) {
        BufferedImage imageToRender = new BufferedImage(array.columns(), array.rows(), BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < array.columns(); x++) {
            for (int y = 0; y < array.rows(); y++) {
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
