package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Max;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 5/25/16.
 * A preprocessor that applies min max scaling
 * Can take a range 
 * X -> (X - min/(max-min)) * (given_max - given_min) + given_min
 * default given_min,given_max is 0,1
 */
public class NormalizerMinMaxScaler implements DataNormalization {
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

    @Override
    public void fit(DataSet dataSet) {
        min = dataSet.getFeatures().min(0);
        max = dataSet.getFeatures().max(0);
        maxMinusMin = max.sub(min);
        maxMinusMin.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        if (maxMinusMin.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
            logger.info("API_INFO: max val minus min val found to be zero. Transform will round upto epsilon to avoid nans.");
    }

    /**
     * Fit the given model
     * @param iterator for the data to iterate over
     */
    @Override
    public void fit(DataSetIterator iterator) {
        INDArray nextMax, nextMin;
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            if(min == null) {
                this.fit(next);
            }
            else {
                nextMin =  next.getFeatures().min(0);;
                min = Nd4j.getExecutioner().execAndReturn(new Min(nextMin,min,min,min.length()));

                nextMax =  next.getFeatures().max(0);
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
        if (min == null || max == null)
            throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        if (maxRange - minRange < 0)
            throw new RuntimeException("API_USE_ERROR: The given max value minus min value has to be greater than 0");
        INDArray theFeatures = toPreProcess.getFeatures();
        preProcess(theFeatures);
    }

    public void preProcess(INDArray theFeatures) {
        // subtract by dataset min
        theFeatures.subiRowVector(min);
        // scale by dataset range
        theFeatures.diviRowVector(maxMinusMin);
        // scale by given or default feature range
        theFeatures.muli(maxRange - minRange + Nd4j.EPS_THRESHOLD);
        // offset by given min feature value
        theFeatures.addi(minRange);
    }

    /**
     * Transform the data
     * @param toPreProcess the dataset to transform
     */
    @Override
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    public void transform(INDArray theFeatures) {
        this.preProcess(theFeatures);
    }

    public void revertPreProcess(DataSet toPreProcess) {
        if (min == null || max == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");

        toPreProcess.getFeatures().subi(minRange);
        toPreProcess.getFeatures().divi(maxRange - minRange + Nd4j.EPS_THRESHOLD);
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
     * @param  statistics the statistics to load
     * @throws IOException
     */
    @Override
    public void load(File...statistics) throws IOException {
        this.min = Nd4j.readBinary(statistics[0]);
        this.max = Nd4j.readBinary(statistics[1]);
        this.maxMinusMin = max.sub(min);
    }

    /**
     * Save the current min and max
     * @param files the statistics to save
     * @throws IOException
     */
    @Override
    public void save(File...files) throws IOException {
        Nd4j.saveBinary(this.min,files[0]);
        Nd4j.saveBinary(this.max,files[1]);
    }
}
