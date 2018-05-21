package org.nd4j.linalg.dataset.api.preprocessor.stats;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Statistics about the normal distribution of values in data (means and standard deviations).
 * Can be constructed incrementally by using the DynamicCustomOpsBuilder, which is useful for obtaining these statistics from an
 * iterator. Can also load and save from files.
 *
 * @author Ede Meijer
 */
@Getter
@EqualsAndHashCode
public class DistributionStats implements NormalizerStats {
    private static final Logger logger = LoggerFactory.getLogger(NormalizerStandardize.class);

    private final INDArray mean;
    private final INDArray std;

    /**
     * @param mean row vector of means
     * @param std  row vector of standard deviations
     */
    public DistributionStats(@NonNull INDArray mean, @NonNull INDArray std) {
        Transforms.max(std, Nd4j.EPS_THRESHOLD, false);
        if (std.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD)) {
            logger.info("API_INFO: Std deviation found to be zero. Transform will round up to epsilon to avoid nans.");
        }

        this.mean = mean;
        this.std = std;
    }

    /**
     * Load distribution statistics from the file system
     *
     * @param meanFile file containing the means
     * @param stdFile  file containing the standard deviations
     */
    public static DistributionStats load(@NonNull File meanFile, @NonNull File stdFile) throws IOException {
        return new DistributionStats(Nd4j.readBinary(meanFile), Nd4j.readBinary(stdFile));
    }

    /**
     * Save distribution statistics to the file system
     *
     * @param meanFile file to contain the means
     * @param stdFile  file to contain the standard deviations
     */
    public void save(@NonNull File meanFile, @NonNull File stdFile) throws IOException {
        Nd4j.saveBinary(getMean(), meanFile);
        Nd4j.saveBinary(getStd(), stdFile);
    }

    /**
     * DynamicCustomOpsBuilder class that can incrementally update a running mean and variance in order to create statistics for a
     * large set of data
     */
    public static class Builder implements NormalizerStats.Builder<DistributionStats> {
        private long runningCount = 0;
        private INDArray runningMean;
        private INDArray runningVariance;

        /**
         * Add the features of a DataSet to the statistics
         */
        public Builder addFeatures(@NonNull org.nd4j.linalg.dataset.api.DataSet dataSet) {
            return add(dataSet.getFeatures(), dataSet.getFeaturesMaskArray());
        }

        /**
         * Add the labels of a DataSet to the statistics
         */
        public Builder addLabels(@NonNull org.nd4j.linalg.dataset.api.DataSet dataSet) {
            return add(dataSet.getLabels(), dataSet.getLabelsMaskArray());
        }

        /**
         * Add rows of data to the statistics
         *
         * @param data the matrix containing multiple rows of data to include
         * @param mask (optionally) the mask of the data, useful for e.g. time series
         */
        public Builder add(@NonNull INDArray data, INDArray mask) {
            data = DataSetUtil.tailor2d(data, mask);

            // Using https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            if (data == null) {
                // Nothing to add. Either data is empty or completely masked. Just skip it, otherwise we will get
                // null pointer exceptions.
                return this;
            }
            INDArray mean = data.mean(0);
            INDArray variance = data.var(false, 0);
            long count = data.size(0);

            if (runningMean == null) {
                // First batch
                runningMean = mean;
                runningVariance = variance;
                runningCount = count;

                if (data.size(0) == 1) {
                    //Handle edge case: currently, reduction ops may return the same array
                    //But we don't want to modify this array in-place later
                    runningMean = runningMean.dup();
                    runningVariance = runningVariance.dup();
                }
            } else {
                // Update running variance
                INDArray deltaSquared = Transforms.pow(mean.subRowVector(runningMean), 2);
                INDArray mB = variance.muli(count);
                runningVariance.muli(runningCount).addiRowVector(mB)
                                .addiRowVector(deltaSquared
                                                .muli((float) (runningCount * count) / (runningCount + count)))
                                .divi(runningCount + count);

                // Update running count
                runningCount += count;

                // Update running mean
                INDArray xMinusMean = data.subRowVector(runningMean);
                runningMean.addi(xMinusMean.sum(0).divi(runningCount));
            }

            return this;
        }

        /**
         * Create a DistributionStats object from the data ingested so far. Can be used multiple times when updating
         * online.
         */
        public DistributionStats build() {
            if (runningMean == null) {
                throw new RuntimeException("No data was added, statistics cannot be determined");
            }
            return new DistributionStats(runningMean.dup(), Transforms.sqrt(runningVariance, true));
        }
    }
}
