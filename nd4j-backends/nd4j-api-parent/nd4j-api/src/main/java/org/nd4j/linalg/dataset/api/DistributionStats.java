package org.nd4j.linalg.dataset.api;

import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Statistics about the distribution of values in data (mean and standard deviation).
 * Can be constructed incrementally by using the Builder, which is useful for obtaining these statistics from an
 * iterator. Can also load and save from files.
 */
@Getter
public class DistributionStats {
    private static final Logger logger = LoggerFactory.getLogger(NormalizerStandardize.class);

    private final INDArray mean;
    private final INDArray std;

    public DistributionStats(@NonNull INDArray mean, @NonNull INDArray std) {
        Transforms.max(std, Nd4j.EPS_THRESHOLD, false);
        if (std.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD)) {
            logger.info("API_INFO: Std deviation found to be zero. Transform will round up to epsilon to avoid nans.");
        }

        this.mean = mean;
        this.std = std;
    }

    public static DistributionStats load(@NonNull File meanFile, @NonNull File stdFile) throws IOException {
        return new DistributionStats(Nd4j.readBinary(meanFile), Nd4j.readBinary(stdFile));
    }

    public void save(@NonNull File meanFile, @NonNull File stdFile) throws IOException {
        Nd4j.saveBinary(getMean(), meanFile);
        Nd4j.saveBinary(getStd(), stdFile);
    }

    public static class Builder {
        private int runningCount = 0;
        private INDArray runningMean;
        private INDArray runningVariance;

        public Builder addFeatures(@NonNull DataSet dataSet) {
            return add(dataSet.getFeatures(), dataSet.getFeaturesMaskArray());
        }

        public Builder addLabels(@NonNull DataSet dataSet) {
            return add(dataSet.getLabels(), dataSet.getLabelsMaskArray());
        }

        public Builder add(@NonNull INDArray data, INDArray mask) {
            data = DataSetUtil.tailor2d(data, mask);

            // Using https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            INDArray mean = data.mean(0);
            INDArray variance = data.var(false, 0);
            int count = data.size(0);

            if (runningMean == null) {
                // First batch
                runningMean = mean;
                runningVariance = variance;
                runningCount = count;
            } else {
                // Update running variance
                INDArray deltaSquared = Transforms.pow(mean.subRowVector(runningMean), 2);
                INDArray mB = variance.muli(count);
                runningVariance.muli(runningCount)
                    .addiRowVector(mB)
                    .addiRowVector(deltaSquared.muli((float)(runningCount * count) / (runningCount + count)))
                    .divi(runningCount + count);

                // Update running count
                runningCount += count;

                // Update running mean
                INDArray xMinusMean = data.subRowVector(runningMean);
                runningMean.addi(xMinusMean.sum(0).divi(runningCount));
            }

            return this;
        }

        public DistributionStats build() {
            return new DistributionStats(runningMean, Transforms.sqrt(runningVariance, false));
        }

        public static List<DistributionStats> buildList(@NonNull List<Builder> builders) {
            List<DistributionStats> result = new ArrayList<>(builders.size());
            for (Builder builder : builders) {
                result.add(builder.build());
            }
            return result;
        }
    }
}