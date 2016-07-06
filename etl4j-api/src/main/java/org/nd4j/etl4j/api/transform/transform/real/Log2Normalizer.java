package org.nd4j.etl4j.api.transform.transform.real;

import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.DoubleMetaData;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.writable.Writable;

/**
 * Normalize by taking scale * log2((in-columnMin)/(mean-columnMin) + 1)
 * Maps values in range (columnMin to infinity) to (0 to infinity)
 * Most suitable for values with a geometric/negative exponential type distribution.
 *
 * @author Alex Black
 */
public class Log2Normalizer extends BaseDoubleTransform {

    protected static final double log2 = Math.log(2);
    protected final double columnMean;
    protected final double columnMin;
    protected final double scalingFactor;

    public Log2Normalizer(String columnName, double columnMean, double columnMin, double scalingFactor) {
        super(columnName);
        if (Double.isNaN(columnMean) || Double.isInfinite(columnMean))
            throw new IllegalArgumentException("Invalid input");
        this.columnMean = columnMean;
        this.columnMin = columnMin;
        this.scalingFactor = scalingFactor;
    }

    public Writable map(Writable writable) {
        double val = writable.toDouble();
        if (Double.isNaN(val)) return new DoubleWritable(0);
        return new DoubleWritable(normMean(val));
    }

    private double log2(double x) {
        return Math.log(x) / log2;
    }

    private double normMean(double in) {
        return scalingFactor * log2((in - columnMin) / (columnMean - columnMin) + 1);
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnMeta) {
        return new DoubleMetaData(0.0, null);
    }

    @Override
    public String toString() {
        return "Log2Normalizer(columnMean=" + columnMean + ",columnMin=" + columnMin + ",scalingFactor=" + scalingFactor + ")";
    }

}
