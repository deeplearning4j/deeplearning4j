package org.nd4j.etl4j.api.transform.metadata;

import org.nd4j.etl4j.api.transform.ColumnType;
import lombok.Data;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * MetaData for a double column.
 *
 * @author Alex Black
 */
@Data
public class DoubleMetaData implements ColumnMetaData {

    //min/max are nullable: null -> no restriction on min/max values
    private final Double min;
    private final Double max;
    private final boolean allowNaN;
    private final boolean allowInfinite;

    public DoubleMetaData() {
        this(null, null, false, false);
    }

    /**
     * @param min Min allowed value. If null: no restriction on min value value in this column
     * @param max Max allowed value. If null: no restiction on max value in this column
     */
    public DoubleMetaData(Double min, Double max) {
        this(min, max, false, false);
    }

    /**
     * @param min           Min allowed value. If null: no restriction on min value value in this column
     * @param max           Max allowed value. If null: no restiction on max value in this column
     * @param allowNaN      Are NaN values ok?
     * @param allowInfinite Are +/- infinite values ok?
     */
    public DoubleMetaData(Double min, Double max, boolean allowNaN, boolean allowInfinite) {
        this.min = min;
        this.max = max;
        this.allowNaN = allowNaN;
        this.allowInfinite = allowInfinite;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Double;
    }

    @Override
    public boolean isValid(Writable writable) {
        double d;
        try {
            d = writable.toDouble();
        } catch (Exception e) {
            return false;
        }

        if (allowNaN && Double.isNaN(d)) return true;
        if (allowInfinite && Double.isInfinite(d)) return true;

        if (min != null && d < min) return false;
        if (max != null && d > max) return false;

        return true;
    }

    @Override
    public DoubleMetaData clone() {
        return new DoubleMetaData(min, max, allowNaN, allowInfinite);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("DoubleMetaData(");
        boolean needComma = false;
        if (min != null) {
            sb.append("minAllowed=").append(min);
            needComma = true;
        }
        if (max != null) {
            if (needComma) sb.append(",");
            sb.append("maxAllowed=").append(max);
            needComma = true;
        }
        if (needComma) sb.append(",");
        sb.append("allowNaN=").append(allowNaN).append(",allowInfinite=").append(allowInfinite).append(")");
        return sb.toString();
    }
}
