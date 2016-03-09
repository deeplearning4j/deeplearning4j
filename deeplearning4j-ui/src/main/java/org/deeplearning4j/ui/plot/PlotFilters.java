package org.deeplearning4j.ui.plot;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Based on the work by krizshevy et. al
 *
 * Plot filters takes in either 2d or 4d input
 *
 * 2d input represents one matrix.
 *
 * Typically for RBMs and AutoEncoders
 * this will be a transposed nout x nin
 * matrix.
 *
 * For 4d input (multiple channels and images)
 *
 * the input should of shape:Au
 * channels x number images x rows x columns
 *
 * @author Adam Gibson
 */
public class PlotFilters {
    private INDArray plot;
    private INDArray input;
    private int[] tileShape;
    private int[] tileSpacing = {0,0};
    private int[] imageShape;
    private boolean scaleRowsToInterval = true;
    private boolean outputPixels = true;

    /**
     *
     * @param input a 2d or 4d matrix (see the input above)
     * @param tileShape the tile shape (typically 10,10)
     * @param tileSpacing the tile spacing(typically 0,0 or 1,1)
     * @param imageShape the intended image shape
     */
    public PlotFilters(INDArray input, int[] tileShape, int[] tileSpacing, int[] imageShape) {
        this.input = input;
        this.tileShape = tileShape;
        this.tileSpacing = tileSpacing;
        this.imageShape = imageShape;
    }


    public INDArray getInput() {
        return input;
    }

    public void setInput(INDArray input) {
        this.input = input;
    }

    /**
     * scale the data to between 0 and 1
     * @param toScale the data to scale
     * @return the scaled version of the data passed in
     */
    public INDArray scale(INDArray toScale) {
        return toScale.sub(toScale.min(Integer.MAX_VALUE)).muli(1.0 / (Nd4j.EPS_THRESHOLD + toScale.max(Integer.MAX_VALUE).getDouble(0)));
    }


    /**
     * Plot the image
     */
    public void plot() {
        int[] retShape = {(imageShape[0] + tileSpacing[0]) * tileShape[0] - tileSpacing[0],(imageShape[1] + tileSpacing[1]) * tileShape[1] - tileSpacing[1]};
        if(input.rank() == 2) {
            plot = plotSection(input,retShape);
        }
        else {
            plot = Nd4j.zeros(retShape[0],retShape[1],4);

            for(int i = 0; i < 4; i++) {
                INDArray retSection = plotSection(input.slice(i), retShape);
                plot.putSlice(i,retSection);
            }
        }


    }

    /**
     * Returns the plot ndarray
     * to be rendered
     * @return the plot ndarray to be rendered
     */
    public INDArray getPlot() {
        if(plot == null) {
            throw new IllegalStateException("Please call plot() first.");
        }
        return plot;
    }



    private INDArray plotSection(INDArray input,int[] retShape) {
        INDArray ret = Nd4j.zeros(retShape);
        if(input.getLeadingOnes() == 2)
            input = input.reshape(input.size(-2),input.size(-1));
        int h = imageShape[0];
        int w = imageShape[1];
        int hs = tileSpacing[0];
        int ws = tileSpacing[1];
        for(int tileRow = 0; tileRow < tileShape[0]; tileRow++) {
            for(int tileCol = 0; tileCol < tileShape[1]; tileCol++) {
                if(tileRow * tileShape[1] + tileCol < input.size(0)) {
                    INDArray image = input.get(NDArrayIndex.point(tileRow * tileShape[1] + tileCol));
                    image = image.reshape(imageShape);

                    if(scaleRowsToInterval) {
                        image = scale(image);
                    }

                    if(outputPixels)
                        image.muli(255);
                    int rowBegin = tileRow * (h + hs);
                    int rowEnd = tileRow * (h + hs) + h;
                    int colBegin = tileCol * (w + ws);
                    int colEnd = tileCol * (w + ws) + w;
                    INDArrayIndex rowIndex = NDArrayIndex.interval(rowBegin,rowEnd);
                    INDArrayIndex colIndex = NDArrayIndex.interval(colBegin,colEnd);
                    ret.put(new INDArrayIndex[]{rowIndex,colIndex},image);


                }
            }
        }

        return ret;
    }




}
