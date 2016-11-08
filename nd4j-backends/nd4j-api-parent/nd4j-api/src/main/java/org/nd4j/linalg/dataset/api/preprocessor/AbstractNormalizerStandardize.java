package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.DistributionStats;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Base class with common functionality for standardize normalizers
 */
abstract class AbstractNormalizerStandardize {
    boolean fitLabels = false;

    /**
     * Flag to specify if the labels/outputs in the dataset should be also normalized
     * default value is false
     *
     * @param fitLabels
     */
    public void fitLabel(boolean fitLabels) {
        this.fitLabels = fitLabels;
    }

    void assertIsFit() {
        if (!isFit()) {
            throw new RuntimeException(
                "API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)"
            );
        }
    }

    protected void preProcess(INDArray theFeatures, DistributionStats stats) {
        if (theFeatures.rank() == 2) {
            theFeatures.subiRowVector(stats.getMean());
            theFeatures.diviRowVector(stats.getStd());
        }
        // if feature Rank is 3 (time series) samplesxfeaturesxtimesteps
        // if feature Rank is 4 (images) samplesxchannelsxrowsxcols
        // both cases operations should be carried out in dimension 1
        else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(theFeatures, stats.getMean(), theFeatures, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(theFeatures, stats.getStd(), theFeatures, 1));
        }
    }

    abstract protected boolean isFit();

    void revert(INDArray data, DistributionStats distribution) {
        if (data.rank() == 2) {
            data.muliRowVector(distribution.getStd());
            data.addiRowVector(distribution.getMean());
        } else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(data, distribution.getStd(), data, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(data, distribution.getMean(), data, 1));
        }
    }
}
