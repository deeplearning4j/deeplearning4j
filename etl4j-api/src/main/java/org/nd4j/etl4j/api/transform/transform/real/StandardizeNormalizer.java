package org.nd4j.etl4j.api.transform.transform.real;

import org.nd4j.etl4j.api.io.data.DoubleWritable;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * Normalize using (x-mean)/sigma.
 * Also known as a standard score, standardization etc.
 *
 * @author Alex Black
 */
public class StandardizeNormalizer extends BaseDoubleTransform {

    protected final double mean;
    protected final double sigma;

    public StandardizeNormalizer(String columnName, double mean, double sigma) {
        super(columnName);
        this.mean = mean;
        this.sigma = sigma;
    }


    @Override
    public Writable map(Writable writable) {
        double val = writable.toDouble();
        return new DoubleWritable((val - mean) / sigma);
    }

    @Override
    public String toString() {
        return "StandardizeNormalizer(mean=" + mean + ",sigma=" + sigma + ")";
    }
}
