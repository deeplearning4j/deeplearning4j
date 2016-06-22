package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Max;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by susaneraly on 5/25/16.
 * A preprocessor that applies min max scaling
 * Can take a range 
 * X -> (X - min/(max-min)) * (given_max - given_min) + given_min
 * default given_min,given_max is 0,1
 */
public class NormalizerMinMaxScaler implements org.nd4j.linalg.dataset.api.DataSetPreProcessor{
    private static Logger logger = LoggerFactory.getLogger(NormalizerMinMaxScaler.class);
    private INDArray min,max,maxMinusMin;
    private double minRange,maxRange;

    /**
    * Preprocessor can take a range as minRange and maxRange
    * @param minRange 
    * @param maxRange 
    */
    public NormalizerMinMaxScaler (double minRange, double maxRange) {
        setMinRange(minRange);
        setMaxRange(maxRange);
    }

    public NormalizerMinMaxScaler () {
        this(0.0,1.0);
    }

    public void setMinRange (double minRange) {
        this.minRange = minRange;
    }

    public void setMaxRange (double maxRange) {
        this.maxRange = maxRange;
    }

    public void fit(DataSet dataSet) {
        min = dataSet.getFeatureMatrix().min(0);
        max = dataSet.getFeatureMatrix().max(0);
        maxMinusMin = max.sub(min);
        maxMinusMin.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        if (maxMinusMin.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD)) 
            logger.info("API_INFO: max val minus min val found to be zero. Transform will round upto epsilon to avoid nans.");
    }

    /**
     * Fit the given model
     * @param iterator for the data to iterate over
     */
    public void fit(DataSetIterator iterator) {
        INDArray nextMax, nextMin;
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            if(min == null) {
                this.fit(next);
            }
            else {
                nextMin =  next.getFeatureMatrix().min(0);;
                min = Nd4j.getExecutioner().execAndReturn(new Min(nextMin,min,min,min.length()));

                nextMax =  next.getFeatureMatrix().max(0);
                max = Nd4j.getExecutioner().execAndReturn(new Max(nextMax,max,max,max.length()));
            }
        }
        maxMinusMin = max.sub(min).add(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        if (maxMinusMin.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD)) 
            logger.info("API_INFO: max val minus min val found to be zero. Transform will round upto epsilon to avoid nans.");
        iterator.reset();
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        if (min == null || max == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        if (maxRange - minRange < 0) throw new RuntimeException("API_USE_ERROR: The given max value minus min value has to be greater than 0");
        // subtract by dataset min
        toPreProcess.getFeatures().subiRowVector(min);
        // scale by dataset range
        toPreProcess.getFeatures().diviRowVector(maxMinusMin);
        // scale by given or default feature range
        toPreProcess.getFeatures().divi(maxRange - minRange + Nd4j.EPS_THRESHOLD);
        // offset by given min feature value
        toPreProcess.getFeatures().addi(minRange);
    }

    /**
     * Transform the data
     * @param toPreProcess the dataset to transform
     */
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    public void transform(DataSetIterator toPreProcessIter) {
        while (toPreProcessIter.hasNext()) {
            this.preProcess(toPreProcessIter.next());
        }
        toPreProcessIter.reset();
    }

    public void revertPreProcess(DataSet toPreProcess) {
        if (min == null || max == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");

        toPreProcess.getFeatures().subi(minRange);
        toPreProcess.getFeatures().muli(maxRange - minRange + Nd4j.EPS_THRESHOLD);
        toPreProcess.getFeatures().muliRowVector(maxMinusMin);
        toPreProcess.getFeatures().addiRowVector(min);
    }

    /**
     *  Revert the data to what it was before transform
     * @param toPreProcess the dataset to revert back
     */
    public void revert(DataSet toPreProcess) {this.revertPreProcess(toPreProcess);}

    public void revert(DataSetIterator toPreProcessIter) {
        while (toPreProcessIter.hasNext()) {
            this.revertPreProcess(toPreProcessIter.next());
        }
        toPreProcessIter.reset();
    }

    public INDArray getMin() {
        if (min == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return min;
    }

    public INDArray getMax() {
        if (max == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return max;
    }

    /**
     * Load the given min and max
     * @param min the min file
     * @param max the max file
     * @throws IOException
     */
    public void load(File min,File max) throws IOException {
        this.min = Nd4j.readBinary(min);
        this.max = Nd4j.readBinary(max);
    }

    /**
     * Save the current min and max
     * @param min the min
     * @param max the max
     * @throws IOException
     */
    public void save(File min,File max) throws IOException {
        Nd4j.saveBinary(this.min,min);
        Nd4j.saveBinary(this.max,max);
    }
}
