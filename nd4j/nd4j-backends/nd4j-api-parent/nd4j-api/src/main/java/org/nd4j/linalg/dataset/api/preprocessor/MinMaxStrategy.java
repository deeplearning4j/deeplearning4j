package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * {@link NormalizerStrategy} implementation that will normalize and denormalize data arrays to a given range, based on
 * statistics of the upper and lower bounds of the population
 *
 * @author Ede Meijer
 */
@Getter
@EqualsAndHashCode
public class MinMaxStrategy implements NormalizerStrategy<MinMaxStats>, Serializable {
    private double minRange;
    private double maxRange;

    public MinMaxStrategy() {
        this(0, 1);
    }

    /**
     * @param minRange the target range lower bound
     * @param maxRange the target range upper bound
     */
    public MinMaxStrategy(double minRange, double maxRange) {
        this.minRange = minRange;
        this.maxRange = Math.max(maxRange, minRange + Nd4j.EPS_THRESHOLD);
    }

    /**
     * Normalize a data array
     *
     * @param array the data to normalize
     * @param stats statistics of the data population
     */
    @Override
    public void preProcess(INDArray array, INDArray maskArray, MinMaxStats stats) {
        if (array.rank() <= 2) {
            array.subiRowVector(stats.getLower());
            array.diviRowVector(stats.getRange());
        }
        // if feature Rank is 3 (time series) samplesxfeaturesxtimesteps
        // if feature Rank is 4 (images) samplesxchannelsxrowsxcols
        // both cases operations should be carried out in dimension 1
        else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(array, stats.getLower(), array, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(array, stats.getRange(), array, 1));
        }

        // Scale by target range
        array.muli(maxRange - minRange);
        // Add target range minimum values
        array.addi(minRange);

        if (maskArray != null) {
            DataSetUtil.setMaskedValuesToZero(array, maskArray);
        }
    }

    /**
     * Denormalize a data array
     *
     * @param array the data to denormalize
     * @param stats statistics of the data population
     */
    @Override
    public void revert(INDArray array, INDArray maskArray, MinMaxStats stats) {
        // Subtract target range minimum value
        array.subi(minRange);
        // Scale by target range
        array.divi(maxRange - minRange);

        if (array.rank() <= 2) {
            array.muliRowVector(stats.getRange());
            array.addiRowVector(stats.getLower());
        } else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(array, stats.getRange(), array, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(array, stats.getLower(), array, 1));
        }

        if (maskArray != null) {
            DataSetUtil.setMaskedValuesToZero(array, maskArray);
        }
    }

    /**
     * Create a new {@link NormalizerStats.Builder} instance that can be used to fit new data and of the opType that
     * belongs to the current NormalizerStrategy implementation
     *
     * @return the new builder
     */
    @Override
    public NormalizerStats.Builder newStatsBuilder() {
        return new MinMaxStats.Builder();
    }
}
