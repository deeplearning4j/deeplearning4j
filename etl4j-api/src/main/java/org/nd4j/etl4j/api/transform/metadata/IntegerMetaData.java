package org.nd4j.etl4j.api.transform.metadata;

import org.nd4j.etl4j.api.transform.ColumnType;
import lombok.Data;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * Metadata for an integer column
 *
 * @author Alex Black
 */
@Data
public class IntegerMetaData implements ColumnMetaData {

    //min/max are nullable: null -> no restriction on min/max values
    private final Integer minAllowedValue;
    private final Integer maxAllowedValue;

    public IntegerMetaData() {
        this(null, null);
    }

    /**
     * @param min Min allowed value. If null: no restriction on min value value in this column
     * @param max Max allowed value. If null: no restiction on max value in this column
     */
    public IntegerMetaData(Integer min, Integer max) {
        this.minAllowedValue = min;
        this.maxAllowedValue = max;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Integer;
    }

    @Override
    public boolean isValid(Writable writable) {
        int value;
        try {
            value = Integer.parseInt(writable.toString());
        } catch (NumberFormatException e) {
            return false;
        }

        if (minAllowedValue != null && value < minAllowedValue) return false;
        if (maxAllowedValue != null && value > maxAllowedValue) return false;
        return true;
    }

    @Override
    public IntegerMetaData clone() {
        return new IntegerMetaData(minAllowedValue, maxAllowedValue);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("IntegerMetaData(");
        if (minAllowedValue != null) sb.append("minAllowed=").append(minAllowedValue);
        if (maxAllowedValue != null) {
            if (minAllowedValue != null) sb.append(",");
            sb.append("maxAllowed=").append(maxAllowedValue);
        }
        sb.append(")");
        return sb.toString();
    }
}
