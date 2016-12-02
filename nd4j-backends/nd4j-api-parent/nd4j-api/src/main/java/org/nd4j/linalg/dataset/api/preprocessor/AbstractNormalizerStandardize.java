package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.DistributionStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Base class with common functionality for standardize normalizers
 */
abstract class AbstractNormalizerStandardize {

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
            theFeatures.diviRowVector(filteredStd(stats));
        }
        // if feature Rank is 3 (time series) samplesxfeaturesxtimesteps
        // if feature Rank is 4 (images) samplesxchannelsxrowsxcols
        // both cases operations should be carried out in dimension 1
        else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(theFeatures, stats.getMean(), theFeatures, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(theFeatures, filteredStd(stats), theFeatures, 1));
        }
    }

    abstract protected boolean isFit();

    void revert(INDArray data, DistributionStats distribution) {
        if (data.rank() == 2) {
            data.muliRowVector(filteredStd(distribution));
            data.addiRowVector(distribution.getMean());
        } else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(data, filteredStd(distribution), data, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(data, distribution.getMean(), data, 1));
        }
    }

    private INDArray filteredStd(DistributionStats stats) {
        /*
            To avoid division by zero when the std deviation is zero, replace by zeros by one
         */
        INDArray stdCopy = stats.getStd();
        BooleanIndexing.replaceWhere(stdCopy, 1.0, Conditions.equals(0));
        return stdCopy;
    }
}
