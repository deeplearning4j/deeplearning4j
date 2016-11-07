package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DistributionStats;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 5/25/16.
 * Standard scaler calculates a moving column wise
 * variance and mean
 * http://www.johndcook.com/blog/standard_deviation/
 */
public class NormalizerStandardize extends AbstractNormalizerStandardize implements DataNormalization {
    private DistributionStats featureStats;
    private DistributionStats labelStats;

    /**
     * Fit the given model with dataset
     * to calculate mean and std dev with
     *
     * @param dataSet
     */
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

    public void transform(INDArray theArray, boolean isFeatures) {
        this.preProcess(theArray, isFeatures ? featureStats : labelStats);
    }

    /**
     * Revert the data to what it was before transform
     *
     * @param toPreProcess the dataset to revert back
     */
    public void revert(DataSet toPreProcess) {
        assertIsFit();
        if (toPreProcess.getFeatures().rank() == 2) {
            toPreProcess.getFeatures().muliRowVector(featureStats.getStd());
            toPreProcess.getFeatures().addiRowVector(featureStats.getMean());
        } else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(toPreProcess.getFeatures(), featureStats.getStd(), toPreProcess.getFeatures(), 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(toPreProcess.getFeatures(), featureStats.getMean(), toPreProcess.getFeatures(), 1));
        }
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
     * Load the given mean and std
     *
     * @param files the statistics to laod
     * @throws IOException
     */
    @Override
    public void load(File... files) throws IOException {
        featureStats = DistributionStats.load(files[0], files[1]);
        if (fitLabels) {
            labelStats = DistributionStats.load(files[2], files[3]);
        }
    }

    /**
     * Save the current mean and std
     *
     * @param files the statistics to save
     * @throws IOException
     */
    @Override
    public void save(File... files) throws IOException {
        featureStats.save(files[0], files[1]);
        if (fitLabels) {
            labelStats.save(files[2], files[3]);
        }
    }
}
