package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Ceil;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 5/25/16.
 */
public class NormalizerMinMaxScaler implements org.nd4j.linalg.dataset.api.DataSetPreProcessor{
    private INDArray min,max,maxMinusMin;
    private double minRange,maxRange;

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
    }

    /**
     * Fit the given model
     * @param iterator for the data to iterate over
     */
    public void fit(DataSetIterator iterator) {
        INDArray maxN, nextMax, nextMaxN, nextMin;
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            if(min == null) {
                this.fit(next);
            }
            else {
                nextMin =  next.getFeatureMatrix().min(0);;
                Nd4j.getExecutioner().exec(new Ceil(nextMin,min));

                nextMax =  next.getFeatureMatrix().max(0);
                maxN = max.mul(-1.0);
                nextMaxN = nextMax.mul(-1.0);
                Nd4j.getExecutioner().exec(new Ceil(nextMaxN,maxN));
                max = maxN.mul(-1.0);
            }
        }
        maxMinusMin = max.sub(min);
        iterator.reset();
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        if (min == null || max == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");

        // subtract by dataset min
        toPreProcess.setFeatures(toPreProcess.getFeatures().subRowVector(min));
        // scale by dataset range
        toPreProcess.setFeatures(toPreProcess.getFeatures().divRowVector(maxMinusMin));
        // scale by given or default feature range
        toPreProcess.setFeatures(toPreProcess.getFeatures().div(maxRange - minRange));
        // offset by given min feature value
        toPreProcess.setFeatures(toPreProcess.getFeatures().add(minRange));
    }

    /**
     * Transform the data
     * @param toPreProcess the dataset to transform
     */
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
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
