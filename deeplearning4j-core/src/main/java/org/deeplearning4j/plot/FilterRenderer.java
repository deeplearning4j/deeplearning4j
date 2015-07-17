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

package org.deeplearning4j.plot;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.Image;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.Map;
import java.util.TreeMap;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adapted from:
 * https://github.com/jpatanooga/Metronome/blob/master/src/main/java/tv/floe/metronome/deeplearning/rbm/visualization/RBMRenderer.java
 * @author Adam Gibson
 *
 */
@Deprecated
public class FilterRenderer {

    public  JFrame frame;
    BufferedImage img;
    private int width = 28;
    private int height = 28;
    public String title = "TEST";
    private int heightOffset = 0;
    private int widthOffset = 0;
    private static final Logger log = LoggerFactory.getLogger(FilterRenderer.class);


    public FilterRenderer() { }

    public void renderHiddenBiases(int heightOffset, int widthOffset, INDArray render_data, String filename) {

        this.width = render_data.columns();
        this.height = render_data.rows();

        img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.heightOffset = heightOffset;
        this.widthOffset = widthOffset;
        WritableRaster r = img.getRaster();
        int[] equiv = new int[ render_data.length()];

        for (int i = 0; i < equiv.length; i++) {

            equiv[i] = (int) Math.round(render_data.getDouble(i) * 256 );
            log.debug( "> " + equiv[i] );

        }

        log.debug( "hbias size: Cols: " + render_data.columns() + ", Rows: " + render_data.rows()  );

        r.setDataElements(0, 0, width, height, equiv);

        this.saveToDisk(filename);

    }



    public int computeHistogramBucketIndex(double min, double stepSize, double value, int numberBins) {

        for ( int x = 0; x < numberBins; x++ ) {

            double tmp = (x * stepSize) + min;

            if ( value >= tmp && value <= (tmp + stepSize) ) {
                return x;
            }

        }

        return -10;

    }



    public static double round(double unrounded, int precision, int roundingMode)
    {
        BigDecimal bd = new BigDecimal(unrounded);
        BigDecimal rounded = bd.setScale(precision, roundingMode);
        return rounded.doubleValue();
    }

    private String buildBucketLabel(int bucketIndex, double stepSize, double min) {

        double val = min + (bucketIndex * stepSize);
        String ret = "" + round(val, 2, BigDecimal.ROUND_HALF_UP);


        return ret;

    }

    /**
     * Take some matrix input data and a bucket count and compute:
     *
     * - a list of N buckets, each with:
     * 1. a bucket label
     * 2. a bucket count
     *
     * over the input dataset
     *
     * @param data
     * @param numberBins
     * @return
     */
    public Map<Integer, Integer> generateHistogramBuckets(INDArray data, int numberBins) {

        Map<Integer, Integer> mapHistory = new TreeMap<>();

        double min =  data.min(Integer.MAX_VALUE).getDouble(0);
        double max =  data.max(Integer.MAX_VALUE).getDouble(0);

        double range = max - min;
        double stepSize = range / numberBins;

		/*
		log.debug( "min: " + min );
		log.debug( "max: " + max );
		log.debug( "range: " + range );
		log.debug( "stepSize: " + stepSize );
		log.debug( "numberBins: " + numberBins );
		 */
        //stepSize = 1;

        for ( int row = 0; row < data.rows(); row++ ) {

            for (int col = 0; col < data.columns(); col++ ) {

                double matrix_value = data.getScalar( row, col ).getDouble(0);

                // at this point we need round values into bins

                int bucket_key = this.computeHistogramBucketIndex(min, stepSize, matrix_value, numberBins);

                int entry = 0;

                if (mapHistory.containsKey( bucket_key )) {

                    // entry exists, increment

                    entry = mapHistory.get( bucket_key );
                    entry++;

                    mapHistory.put( bucket_key, entry );

                } else {

                    // entry does not exit, createComplex, insert

                    // createComplex new key
                    String bucket_label = buildBucketLabel(bucket_key, stepSize, min);

                    // new entry
                    entry = 1; // new Pair<String, Integer>(bucket_label, 1);

                    // update data structure
                    mapHistory.put( bucket_key, entry );
                }

            }
        }

        return mapHistory;


    }


    /**
     * Groups values into 1 of 10 bins, sums, and renders
     *
     * NOTE: this is "render histogram BS code";
     * - I'm not exactly concerned with how pretty it is.
     *
     * @param data
     * @param numberBins
     */
    public void renderHistogram(INDArray data, String filename, int numberBins) {

        Map<Integer, Integer> mapHistory = this.generateHistogramBuckets( data, numberBins );

        double min =  data.min(Integer.MAX_VALUE).getDouble(0); //data.getFromOrigin(0, 0);
        double max =  data.max(Integer.MAX_VALUE).getDouble(0); //data.getFromOrigin(0, 0);

        double range = max - min;
        double stepSize = range / numberBins;


        int xOffset = 50;
        int yOffset = -50;

        int graphWidth = 600;
        int graphHeight = 400;

        BufferedImage img = new BufferedImage( graphWidth, graphHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = img.createGraphics();


        final int BAR_WIDTH = 40;
        final int X_POSITION = 0;
        final int Y_POSITION = 200;
        int MIN_BAR_WIDTH = 4;

        g2d.setColor(Color.LIGHT_GRAY);
        //g2d.drawRect(xOffset, yOffset, graphWidth, graphHeight);
        g2d.fillRect(0, 0, graphWidth, graphHeight);
        //g2d.fill(new Rectangle(x, y, width, height));

        //     int barWidth = Math.max(MIN_BAR_WIDTH,
        //             (int) Math.floor((double) graphWidth
        //             / (double) mapHistory.size()));
        int barWidth = BAR_WIDTH;

        //       log.debug("width = " + graphWidth + "; size = "
        //             + mapHistory.size() + "; barWidth = " + barWidth);

        int maxValue = 0;
        for (Integer key : mapHistory.keySet()) {
            int value = mapHistory.get(key);
            maxValue = Math.max(maxValue, value);
        }

        // draw Y-scale

        //log.debug( "max-value: " + maxValue );

        double plotAreaHeight = (graphHeight + yOffset);

        double yScaleStepSize = plotAreaHeight / 4;

        double yLabelStepSize = (double)maxValue / 4.0f;

        for ( int yStep = 0; yStep < 5; yStep++ ) {

            double curLabel = yStep * yLabelStepSize ;

            long curY =  (graphHeight + yOffset) - Math.round(( (int) (curLabel)
                    / (double) maxValue) * (graphHeight + yOffset - 20));


            g2d.setColor(Color.BLACK);
            g2d.drawString("" + curLabel, 10, curY );

        }

        int xPos = xOffset;

        for (Integer key : mapHistory.keySet()) {

            long value = mapHistory.get(key);

            String bucket_label = this.buildBucketLabel(key, stepSize, min);

            long barHeight = Math.round(((double) value
                    / (double) maxValue) * (graphHeight + yOffset - 20));

            //g2d.setColor(new Color(key, key, key));
            g2d.setColor(Color.BLUE);

            long yPos = graphHeight + yOffset - barHeight;

            g2d.fillRect( xPos, (int) yPos, barWidth, (int) barHeight);
            g2d.setColor(Color.DARK_GRAY);
            g2d.drawRect(xPos, (int) yPos,barWidth, (int) barHeight);

            g2d.setColor(Color.BLACK);
            g2d.drawString(bucket_label, xPos + ((barWidth / 2) - 10), barHeight + 20 + yPos);
            //g2d.draw(bar);
            xPos += barWidth + 10;
        }

        try {
            saveImageToDisk( img, filename );
        } catch (IOException e) {
            e.printStackTrace();
        }

        g2d.dispose();

    }

    /**
     *
     * Once the probability image and weight histograms are
     * behaving satisfactorily, we plot the learned filter
     * for each hidden neuron, one per column of W. Each filter
     * is of the same dimension as the input data, and it is
     * most useful to visualize the filters in the same way
     * as the input data is visualized.
     * @throws Exception
     *
     */
    public BufferedImage renderFilters( INDArray data, String filename, int patchWidth, int patchHeight,int patchesPerRow) throws Exception {

        int[] equiv = new int[ data.length()  ];

        int numberCols = data.columns();

        double approx = (double) numberCols / (double) patchesPerRow;
        int numPatchRows = (int) Math.round(approx);
        if(numPatchRows < 1)
            numPatchRows = 1;

        int patchBorder = 2;

        int filterImgWidth = ( patchWidth + patchBorder ) * patchesPerRow;
        int filterImgHeight = numPatchRows * (patchHeight + patchBorder);

        img = new BufferedImage( filterImgWidth, filterImgHeight, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster r = img.getRaster();

        // for each hidden neuron
        // plot the learned filter (same dim as the input data)
        outer:
        for ( int col = 0; col < numberCols; col++ ) {
            int curX = (col % patchesPerRow ) * (patchWidth + patchBorder );
            int curY = col / patchesPerRow * ( patchHeight + patchBorder );

            INDArray column = data.getColumn(col);

            double col_min =  column.min(Integer.MAX_VALUE).getDouble(0);
            double col_max =  column.max(Integer.MAX_VALUE).getDouble(0);

            // reshape the column into the shape of the filter patch
            // render the filter patch
            log.debug("Rendering " + column.length() + " pixels in column " + col + " for filter patch " + patchWidth + " x " + patchHeight + ", total size: " + (patchWidth * patchHeight) + " at " + curX );

            for (int i = 0; i < column.length(); i++) {

                //double patch_normal = ( column.getFromOrigin(i) - min ) / ( max - min + 0.000001 );
                double patch_normal = (  column.getDouble(0) - col_min ) / ( col_max - col_min + 0.000001f );
                equiv[i] = (int) (255 * patch_normal);

            }

            // now draw patch to raster image
            boolean outOfBounds = false;
            if(curX >= filterImgWidth) {
                curX = filterImgWidth - 1;
                outOfBounds = true;
                break outer;

            }

            if(curY >= filterImgHeight) {
                curY = filterImgHeight - 1;
                outOfBounds = true;
                break outer;

            }

            r.setPixels( curX, curY, patchWidth, patchHeight, equiv );
            if(outOfBounds)
                break outer;


        }

        try {
            saveImageToDisk( img, filename );
            GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
            if(!ge.isHeadlessInstance()) {
                log.info("Rendering filter images...");
                JFrame frame = new JFrame();
                frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

                FilterPanel panel = new FilterPanel(img);
                frame.add(panel);
                Dimension d = new Dimension(numberCols * patchWidth , numPatchRows * patchHeight);
                frame.setSize(d);
                frame.setMinimumSize(d);
                panel.setMinimumSize(d);
                frame.pack();
                frame.setVisible(true);
                Thread.sleep(10000);
                frame.dispose();

            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return img;

    }




    public void renderActivations(int heightOffset, int widthOffset, INDArray activation_data, String filename, int scale ) {

        this.width = activation_data.columns();
        this.height = activation_data.rows();


        log.debug( "----- renderActivations ------" );

        img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        this.heightOffset = heightOffset;
        this.widthOffset = widthOffset;
        WritableRaster r = img.getRaster();
        int[] equiv = new int[ activation_data.length() ];

        double max = 0.1f * scale; //MatrixUtils.max(render_data);
        double min = -0.1f * scale; //MatrixUtils.min(render_data);
        double range = max - min;


        for (int i = 0; i < equiv.length; i++) {

            equiv[i] = (int) Math.round(activation_data.getDouble(i) * 255 );

        }


        log.debug( "activations size: Cols: " + activation_data.columns() + ", Rows: " + activation_data.rows()  );

        r.setPixels(0, 0, width, height, equiv);

        this.saveToDisk(filename);

    }





    public static void saveImageToDisk(BufferedImage img, String imageName) throws IOException {

        File outputfile = new File( imageName );
        File parentDir = outputfile.getParentFile();
        if(!parentDir.exists())
            parentDir.mkdirs();
        if(!outputfile.exists())
            outputfile.createNewFile();



        ImageIO.write(img, "png", outputfile);

    }

    public void saveToDisk(String filename) {

        try {
            saveImageToDisk( this.img, filename );
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }


    public void draw() {
        frame = new JFrame(title);
        frame.setVisible(true);
        start();
        frame.add(new JLabel(new ImageIcon(getImage())));

        frame.pack();
        // Better to DISPOSE than EXIT
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }

    public void close() {
        frame.dispose();
    }

    public Image getImage() {
        return img;
    }

    public void start(){


        int[] pixels = ((DataBufferInt)img.getRaster().getDataBuffer()).getData();
        boolean running = true;
        while(running){
            BufferStrategy bs = frame.getBufferStrategy();
            if(bs == null){
                frame.createBufferStrategy(4);
                return;
            }
            for (int i = 0; i < width * height; i++)
                pixels[i] = 0;

            Graphics g= bs.getDrawGraphics();
            g.drawImage(img, heightOffset, widthOffset, width, height, null);
            g.dispose();
            bs.show();

        }
    }


}