package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DistributionStats;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly, EdeMeijer
 * variance and mean
 * Pre processor for DataSet that normalizes feature values (and optionally label values) to have 0 mean and a standard
 * deviation of 1
 */
public class NormalizerStandardize extends AbstractNormalizerStandardize implements DataNormalization {
    private DistributionStats featureStats;
    private DistributionStats labelStats;
    private boolean fitLabels = false;

    public NormalizerStandardize() {
    }

    public NormalizerStandardize(INDArray featureMean, INDArray featureStd) {
        this.featureStats = new DistributionStats(featureMean,featureStd);
        this.fitLabels = false;
    }

    public NormalizerStandardize(INDArray featureMean, INDArray featureStd, INDArray labelMean, INDArray labelStd) {
        this.featureStats = new DistributionStats(featureMean,featureStd);
        this.labelStats = new DistributionStats(labelMean,labelStd);
        this.fitLabels = true;
    }

    /**
     * Flag to specify if the labels/outputs in the dataset should be also normalized
     * default value is false
     *
     * @param fitLabels
     */
    @Override
    public void fitLabel(boolean fitLabels) {
        this.fitLabels = fitLabels;
    }


    @Override
    public boolean isFitLabel() {
        return this.fitLabels;
    }

    /**
     * Fit the given model with dataset
     * to calculate mean and std dev with
     *
     * @param dataSet
     */
    @Override
    public void fit(@NonNull DataSet dataSet) {
        featureStats = new DistributionStats.Builder().addFeatures(dataSet).build();
        if (fitLabels) {
            labelStats = new DistributionStats.Builder().addLabels(dataSet).build();
        }
    }

    /**
     * Fit the given model with a given iterator
     * to calculate mean and std dev with
     *
     * @param iterator
     */
    public void fit(@NonNull DataSetIterator iterator) {
        DistributionStats.Builder featureNormBuilder = new DistributionStats.Builder();
        DistributionStats.Builder labelNormBuilder = new DistributionStats.Builder();

        iterator.reset();
        while (iterator.hasNext()) {
            DataSet next = iterator.next();
            featureNormBuilder.addFeatures(next);
            if (fitLabels) {
                labelNormBuilder.addLabels(next);
            }
        }
        featureStats = featureNormBuilder.build();
        if (fitLabels) {
            labelStats = labelNormBuilder.build();
        }
        iterator.reset();
    }

    @Override
    public void preProcess(@NonNull DataSet toPreProcess) {
        assertIsFit();
        this.preProcess(toPreProcess.getFeatures(), featureStats);
        if (fitLabels) {
            this.preProcess(toPreProcess.getLabels(), labelStats);
        }
    }

    /**
     * Transform the given dataset
     *
     * @param toPreProcess
     */
    @Override
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    /**
     * Transform the given INDArray
     *
     * @param theFeatures
     */
    @Override
    public void transform(INDArray theFeatures) {
        this.transform(theFeatures, true);
    }

    @Override
    public void transformLabel(INDArray label){
        transform(label, false);
    }

    private void transform(INDArray theArray, boolean isFeatures) {
        this.preProcess(theArray, isFeatures ? featureStats : labelStats);
    }

    /**
     * Revert the data to what it was before transform
     *
     * @param data the dataset to revert back
     */
    @Override
    public void revert(DataSet data) {
        assertIsFit();
        revert(data.getFeatures(), featureStats);
        if (fitLabels) {
            revert(data.getLabels(), labelStats);
        }
    }

    @Override
    public void revertFeatures(INDArray features) {
        revert(features,featureStats);
    }

    @Override
    public void revertLabels(INDArray labels) {
        if (!fitLabels) return;
        revert(labels, labelStats);
    }

    public INDArray getMean() {
        assertIsFit();
        return featureStats.getMean();
    }

    public INDArray getLabelMean() {
        assertIsFit();
        return labelStats.getMean();
    }

    public INDArray getStd() {
        assertIsFit();
        return featureStats.getStd();
    }

    public INDArray getLabelStd() {
        assertIsFit();
        return labelStats.getStd();
    }


    @Override
    protected boolean isFit() {
        return featureStats != null;
    }

    /**
     * Load the means and standard deviations from the file system
     *
     * @param files the files to load from. Needs 4 files if normalizing labels, otherwise 2.
     */
    @Override
    public void load(File... files) throws IOException {
        featureStats = DistributionStats.load(files[0], files[1]);
        if (fitLabels) {
            labelStats = DistributionStats.load(files[2], files[3]);
        }
    }

    /**
     * Save the current means and standard deviations to the file system
     *
     * @param files the files to save to. Needs 4 files if normalizing labels, otherwise 2.
     */
    @Override
    public void save(File... files) throws IOException {
        featureStats.save(files[0], files[1]);
        if (fitLabels) {
            labelStats.save(files[2], files[3]);
        }
    }
}
