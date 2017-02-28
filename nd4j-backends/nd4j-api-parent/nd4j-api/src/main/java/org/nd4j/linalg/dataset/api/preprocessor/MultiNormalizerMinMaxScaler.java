package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

/**
 * Pre processor for MultiDataSet that normalizes feature values (and optionally label values) to lie between a minimum
 * and maximum value (by default between 0 and 1)
 *
 * @author Ede Meijer
 */
public class MultiNormalizerMinMaxScaler extends AbstractMultiDataSetNormalizer<MinMaxStats> {
    public MultiNormalizerMinMaxScaler() {
        this(0.0, 1.0);
    }

    /**
     * Preprocessor can take a range as minRange and maxRange
     *
     * @param minRange the target range lower bound
     * @param maxRange the target range upper bound
     */
    public MultiNormalizerMinMaxScaler(double minRange, double maxRange) {
        super(new MinMaxStrategy(minRange, maxRange));
    }

    public double getTargetMin() {
        return ((MinMaxStrategy) strategy).getMinRange();
    }

    public double getTargetMax() {
        return ((MinMaxStrategy) strategy).getMaxRange();
    }

    @Override
    protected NormalizerStats.Builder newBuilder() {
        return new MinMaxStats.Builder();
    }

    public INDArray getMin(int input) {
        return getFeatureStats(input).getLower();
    }

    public INDArray getMax(int input) {
        return getFeatureStats(input).getUpper();
    }

    public INDArray getLabelMin(int output) {
        return getLabelStats(output).getLower();
    }

    public INDArray getLabelMax(int output) {
        return getLabelStats(output).getUpper();
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.MULTI_MIN_MAX;
    }
}
