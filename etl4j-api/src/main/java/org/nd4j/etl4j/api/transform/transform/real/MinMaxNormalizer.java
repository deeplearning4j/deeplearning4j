package org.nd4j.etl4j.api.transform.transform.real;

import org.canova.api.io.data.DoubleWritable;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.DoubleMetaData;

/**
 * Normalizer to map (min to max) -> (newMin-to newMax) linearly. <br>
 * <p>
 * Mathematically: (newMax-newMin)/(max-min) * (x-min) + newMin
 *
 * @author Alex Black
 */
public class MinMaxNormalizer extends BaseDoubleTransform {

    protected final double min;
    protected final double max;
    protected final double newMin;
    protected final double newMax;
    protected final double ratio;

    public MinMaxNormalizer(String columnName, double min, double max) {
        this(columnName, min, max, 0, 1);
    }

    public MinMaxNormalizer(String columnName, double min, double max, double newMin, double newMax) {
        super(columnName);
        this.min = min;
        this.max = max;
        this.newMin = newMin;
        this.newMax = newMax;
        this.ratio = (newMax - newMin) / (max - min);
    }

    @Override
    public Writable map(Writable writable) {
        double val = writable.toDouble();
        if (Double.isNaN(val)) return new DoubleWritable(0);
        return new DoubleWritable(ratio * (val - min) + newMin);
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnMeta) {
        return new DoubleMetaData(newMin, newMax);
    }

    @Override
    public String toString() {
        return "MinMaxNormalizer(min=" + min + ",max=" + max + ",newMin=" + newMin + ",newMax=" + newMax + ")";
    }

}
