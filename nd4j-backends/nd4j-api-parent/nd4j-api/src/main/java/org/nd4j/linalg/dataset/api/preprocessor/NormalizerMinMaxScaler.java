package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Max;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Min;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
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
    private int featureRank = 2;
    private INDArray featureMaxMin, labelMaxMin;
    private INDArray featureMin, featureMax, labelMax, labelMin;
    private boolean fitLabels = false;
    private double minRange, maxRange;

    /**
     * Preprocessor can take a range as minRange and maxRange
     *
     * @param minRange
     * @param maxRange
     */
    public NormalizerMinMaxScaler(double minRange, double maxRange) {
        setMinRange(minRange);
        setMaxRange(maxRange);
    }

    public NormalizerMinMaxScaler() {
        this(0.0, 1.0);
    }

    public void setMinRange(double minRange) {
        this.minRange = minRange;
    }

    public void setMaxRange(double maxRange) {
        this.maxRange = maxRange;
    }

    @Override
    public void fit(DataSet dataSet) {
        featureRank = dataSet.getFeatures().rank();

        INDArray theFeatures = DataSetUtil.tailor2d(dataSet, true);
        featureMaxMin = fit(theFeatures);
        featureMin = featureMaxMin.getRow(0).dup();
        featureMax = featureMaxMin.getRow(1).dup();
        featureMaxMin = featureMax.sub(featureMin);

        if (fitLabels) {
            INDArray theLabels = DataSetUtil.tailor2d(dataSet, false);
            labelMaxMin = fit(theLabels);
            labelMin = labelMaxMin.getRow(0).dup();
            labelMax = labelMaxMin.getRow(1).dup();
            labelMaxMin = labelMax.sub(labelMin);
        }
    }

    private INDArray fit(INDArray theArray) {
        INDArray maxminhere = Nd4j.zeros(2, theArray.size(1));
        maxminhere.putRow(0, theArray.min(0));
        maxminhere.putRow(1, theArray.max(0));
        if (maxminhere.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
            logger.info("API_INFO: max val minus min val found to be zero. Transform will round upto epsilon to avoid nans.");
        return maxminhere;
    }

    /**
     * Fit the given model
     *
     * @param iterator for the data to iterate over
     */
    @Override
    public void fit(DataSetIterator iterator) {
        INDArray nextMax, nextMin;
        while (iterator.hasNext()) {
            DataSet next = iterator.next();
            featureRank = next.getFeatures().rank();
            INDArray theFeatures = DataSetUtil.tailor2d(next, true);
            INDArray theLabels = null;
            if (fitLabels) {
                theLabels = DataSetUtil.tailor2d(next, false);
            }
            if (featureMin == null) {
                this.fit(next);
            } else {
                nextMin = theFeatures.min(0);
                featureMin = Nd4j.getExecutioner().execAndReturn(new Min(nextMin, featureMin, featureMin, featureMin.length()));
                nextMax = theFeatures.max(0);
                featureMax = Nd4j.getExecutioner().execAndReturn(new Max(nextMax, featureMax, featureMax, featureMax.length()));

                if (fitLabels) {
                    nextMin =  theLabels.min(0);
                    labelMin = Nd4j.getExecutioner().execAndReturn(new Min(nextMin,labelMin,labelMin,labelMin.length()));

                    nextMax = theLabels.max(0);
                    labelMax = Nd4j.getExecutioner().execAndReturn(new Max(nextMax, labelMax, labelMax, labelMax.length()));
                }
            }
        }
        featureMaxMin = featureMax.sub(featureMin).add(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        if (featureMaxMin.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
            logger.info("API_INFO: Feature max val minus min val found to be zero. Transform will round upto epsilon to avoid nans.");
        if (fitLabels) {
            labelMaxMin = labelMax.sub(labelMin).add(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
            if (labelMaxMin.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
                logger.info("API_INFO: Labels max val minus min val found to be zero. Transform will round upto epsilon to avoid nans.");
        }
        iterator.reset();
    }

    /**
     * Flag to specify if the labels/outputs in the dataset should be also normalized
     * default value is false
     *
     */
    @Override
    public void fitLabel(boolean fitLabels) {
        this.fitLabels = fitLabels;
    }

    @Override
    public boolean isFitLabel(){
        return this.fitLabels;
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        if (featureMin == null || featureMax == null)
            throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        if (maxRange - minRange < 0)
            throw new RuntimeException("API_USE_ERROR: The given max value minus min value has to be greater than 0");
        INDArray theFeatures = toPreProcess.getFeatures();
        INDArray theLabels = toPreProcess.getLabels();
        this.preProcess(theFeatures, true);
        if (fitLabels) this.preProcess(theLabels, false);
    }

    private void preProcess(INDArray theArray, boolean isFeatures) {
        INDArray min, max, maxmin;
        max = isFeatures ? featureMax : labelMax;
        min = isFeatures ? featureMin : labelMin;
        maxmin = max.sub(min);
        if (theArray.rank() == 2) {
            // subtract by dataset min
            theArray.subiRowVector(featureMin);
            // scale by dataset range
            theArray.diviRowVector(featureMaxMin.add(Nd4j.EPS_THRESHOLD));
            // scale by given or default feature range
            theArray.muli(maxRange - minRange + Nd4j.EPS_THRESHOLD);
            // offset by given min feature value
            theArray.addi(minRange);
        }
        // if feature Rank is 3 (time series) samplesxfeaturesxtimesteps
        // if feature Rank is 4 (images) samplesxchannelsxrowsxcols
        // both cases operations should be carried out in dimension 1
        else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(theArray, min, theArray, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(theArray, maxmin, theArray, 1));
            theArray.muli(maxRange - minRange + Nd4j.EPS_THRESHOLD);
            theArray.addi(minRange);
        }
    }

    /**
     * Transform the data
     *
     * @param toPreProcess the dataset to transform
     */
    @Override
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    @Override
    public void transform(INDArray theFeatures) {
        this.preProcess(theFeatures, true);
    }

    @Override
    public void transformLabel(INDArray labels){
        this.preProcess(labels, false);
    }

    public void revertPreProcess(DataSet toPreProcess) {
        if (featureMin == null || featureMax == null)
            throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");

        revertFeatures(toPreProcess.getFeatures());


        if (fitLabels) {
            revertLabels(toPreProcess.getLabels());
        }
    }

    /**
     * Revert the data to what it was before transform
     *
     * @param toPreProcess the dataset to revert back
     */
    @Override
    public void revert(DataSet toPreProcess) {
        this.revertPreProcess(toPreProcess);
    }

    @Override
    public void revertFeatures(INDArray features){
        features.subi(minRange)
                .divi(maxRange - minRange + Nd4j.EPS_THRESHOLD)
                .muliRowVector(featureMaxMin)
                .addiRowVector(featureMin);
    }

    @Override
    public void revertLabels(INDArray labels){
        if(!fitLabels) return;
        labels.subi(minRange)
                .divi(maxRange - minRange + Nd4j.EPS_THRESHOLD)
                .muliRowVector(featureMaxMin)
                .addiRowVector(featureMin);
    }

    public INDArray getMin() {
        if (featureMin == null)
            throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return featureMin;
    }

    public INDArray getMax() {
        if (featureMax == null)
            throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return featureMax;
    }

    public INDArray getLabelMin() {
        if (labelMin == null)
            throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return labelMin;
    }

    public INDArray getLabelMax() {
        if (labelMax == null)
            throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return labelMax;
    }

    /**
     * Load the given min and max
     *
     * @param statistics the statistics to load
     * @throws IOException
     */
    @Override
    public void load(File... statistics) throws IOException {
        this.featureMin = Nd4j.readBinary(statistics[0]);
        this.featureMax = Nd4j.readBinary(statistics[1]);
        this.featureMaxMin = featureMax.sub(featureMin);
        if (fitLabels) {
            this.labelMin = Nd4j.readBinary(statistics[0]);
            this.labelMax = Nd4j.readBinary(statistics[1]);
            this.labelMaxMin = labelMax.sub(labelMin);
        }
    }

    /**
     * Save the current min and max
     *
     * @param files the statistics to save
     * @throws IOException
     */
    @Override
    public void save(File... files) throws IOException {
        Nd4j.saveBinary(this.featureMin, files[0]);
        Nd4j.saveBinary(this.featureMax, files[1]);
        if (fitLabels) {
            Nd4j.saveBinary(this.labelMin, files[2]);
            Nd4j.saveBinary(this.labelMax, files[3]);
        }
    }
}
