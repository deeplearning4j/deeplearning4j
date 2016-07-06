package org.nd4j.etl4j.api.transform.transform.real;

import org.nd4j.etl4j.api.io.data.DoubleWritable;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * Normalize by substracting the mean
 */
public class SubtractMeanNormalizer extends BaseDoubleTransform {

    private final double mean;

    public SubtractMeanNormalizer(String columnName, double mean){
        super(columnName);
        this.mean = mean;
    }

    @Override
    public Writable map(Writable writable) {
        return new DoubleWritable(writable.toDouble()-mean);
    }

    @Override
    public String toString() {
        return "SubstractMean(mean=" + mean + ")";
    }
}
