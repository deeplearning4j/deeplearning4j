package org.nd4j.linalg.dataset.api;

import lombok.Getter;
import lombok.NonNull;
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
        private int n = 0;
        private INDArray mean;
        private INDArray M2;

        public Builder addFeatures(@NonNull DataSet dataSet) {
            return add(dataSet.getFeatures(), dataSet.getFeaturesMaskArray());
        }

        public Builder addLabels(@NonNull DataSet dataSet) {
            return add(dataSet.getLabels(), dataSet.getLabelsMaskArray());
        }

        public Builder add(@NonNull INDArray data, INDArray mask) {
            data = DataSetUtil.tailor2d(data, mask);

            // Using https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
            if (mean == null) {
                // Initialize means and deviations as zeros now that we know the width of the (tailored) data
                mean = Nd4j.zeros(data.size(1));
                M2 = Nd4j.zeros(data.size(1));
            }
            // TODO: find a way to vectorize this to speed things up
            for (int i = 0; i < data.rows(); i++) {
                INDArray x = data.getRow(i);
                n += 1;
                INDArray delta = x.subRowVector(mean);
                mean.addiRowVector(delta.div(n));
                M2.addiRowVector(delta.mul(x.sub(mean)));
            }

            return this;
        }

        public DistributionStats build() {
            return new DistributionStats(mean, Transforms.sqrt(M2.div(n), false));
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