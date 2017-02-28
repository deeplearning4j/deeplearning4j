package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

/**
 * Pre processor for DataSets that normalizes feature values (and optionally label values) to lie between a minimum
 * and maximum value (by default between 0 and 1)
 *
 * @author susaneraly
 * @author Ede Meijer
 */
public class NormalizerMinMaxScaler extends AbstractDataSetNormalizer<MinMaxStats> {
    public NormalizerMinMaxScaler() {
        this(0.0, 1.0);
    }

    /**
     * Preprocessor can take a range as minRange and maxRange
     *
     * @param minRange
     * @param maxRange
     */
    public NormalizerMinMaxScaler(double minRange, double maxRange) {
        super(new MinMaxStrategy(minRange, maxRange));
    }

    public void setFeatureStats(@NonNull INDArray featureMin, @NonNull INDArray featureMax) {
        setFeatureStats(new MinMaxStats(featureMin, featureMax));
    }

    public void setLabelStats(@NonNull INDArray labelMin, @NonNull INDArray labelMax) {
        setLabelStats(new MinMaxStats(labelMin, labelMax));
    }

    public double getTargetMin() {
        return ((MinMaxStrategy) strategy).getMinRange();
    }

    public double getTargetMax() {
        return ((MinMaxStrategy) strategy).getMaxRange();
    }

    public INDArray getMin() {
        return getFeatureStats().getLower();
    }

    public INDArray getMax() {
        return getFeatureStats().getUpper();
    }

    public INDArray getLabelMin() {
        return getLabelStats().getLower();
    }

    public INDArray getLabelMax() {
        return getLabelStats().getUpper();
    }

    /**
     * Load the given min and max
     *
     * @param statistics the statistics to load
     * @throws IOException
     */
    public void load(File... statistics) throws IOException {
        setFeatureStats(new MinMaxStats(Nd4j.readBinary(statistics[0]), Nd4j.readBinary(statistics[1])));
        if (isFitLabel()) {
            setLabelStats(new MinMaxStats(Nd4j.readBinary(statistics[2]), Nd4j.readBinary(statistics[3])));
        }
    }

    /**
     * Save the current min and max
     *
     * @param files the statistics to save
     * @throws IOException
     * @deprecated use {@link NormalizerSerializer instead}
     */
    public void save(File... files) throws IOException {
        Nd4j.saveBinary(getMin(), files[0]);
        Nd4j.saveBinary(getMax(), files[1]);
        if (isFitLabel()) {
            Nd4j.saveBinary(getLabelMin(), files[2]);
            Nd4j.saveBinary(getLabelMax(), files[3]);
        }
    }

    @Override
    protected NormalizerStats.Builder newBuilder() {
        return new MinMaxStats.Builder();
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.MIN_MAX;
    }
}
