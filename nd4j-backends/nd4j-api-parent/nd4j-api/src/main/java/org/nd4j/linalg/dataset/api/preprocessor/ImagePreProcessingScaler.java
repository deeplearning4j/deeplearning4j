package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 6/23/16.
 * A preprocessor specifically for images that applies min max scaling
 * Can take a range, so pixel values can be scaled from 0->255 to minRange->maxRange
 * default minRange = 0 and maxRange = 1;
 * If pixel values are not 8 bits, you can specify the number of bits as the third argument in the constructor
 * For values that are already floating point, specify the number of bits as 1
 *
 */
public class ImagePreProcessingScaler implements DataNormalization {

    private static Logger logger = LoggerFactory.getLogger(NormalizerMinMaxScaler.class);

    private double minRange, maxRange;
    private double maxPixelVal;
    private int maxBits;

    public ImagePreProcessingScaler() {
        this(0,1,8);
    }

    public ImagePreProcessingScaler(double a, double b) {
        this(a, b, 8);
    }
    /**
     * Preprocessor can take a range as minRange and maxRange
     * @param a, default = 0
     * @param b, default = 1
     * @param maxBits in the image, default = 8
     */
    public ImagePreProcessingScaler(double a, double b, int maxBits) {
        //Image values are not always from 0 to 255 though
        //some images are 16-bit, some 32-bit, integer, or float, and those BTW already come with values in [0..1]...
        //If the max expected value is 1, maxBits should be specified as 1
        maxPixelVal = Math.pow(2,maxBits) - 1;
        this.minRange = a;
        this.maxRange = b;
    }

    /**
     * Fit a dataset (only compute
     * based on the statistics from this dataset0
     *
     * @param dataSet the dataset to compute on
     */
    @Override
    public void fit(DataSet dataSet) {

    }

    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     *
     * @param iterator the iterator to use for
     *                 collecting statistics.
     */
    @Override
    public void fit(DataSetIterator iterator) {

    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray features = toPreProcess.getFeatures();
        this.preProcess(features);
    }

    public void preProcess(INDArray features) {
        features.divi(this.maxPixelVal); //Scaled to 0->1
        if (this.maxRange - this.minRange != 1)
            features.muli(this.maxRange - this.minRange); //Scaled to minRange -> maxRange
        if (this.minRange != 0)
            features.addi(this.minRange); //Offset by minRange
    }

    /**
     * Transform the data
     * @param toPreProcess the dataset to transform
     */
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    public void transform(INDArray features) {
        this.preProcess(features);
    }

    /**
     * Load the statistics
     * for the data normalizer
     *
     * @param statistics the files to persist
     * @throws IOException
     */
    @Override
    public void load(File... statistics) throws IOException {

    }

    /**
     * Save the accumulated statistics
     *
     * @param statistics the statistics to save
     * @throws IOException
     */
    @Override
    public void save(File... statistics) throws IOException {

    }

}
