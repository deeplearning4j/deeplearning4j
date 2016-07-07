package org.datavec.api.transform.metadata;

import org.datavec.api.io.data.IntWritable;
import org.datavec.api.io.data.LongWritable;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.writable.Writable;
import lombok.Data;

/**
 * Metadata for an long column
 *
 * @author Alex Black
 */
@Data
public class LongMetaData implements ColumnMetaData {

    //min/max are nullable: null -> no restriction on min/max values
    private final Long min;
    private final Long max;

    public LongMetaData() {
        this(null, null);
    }

    public LongMetaData(Long min, Long max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Long;
    }

    @Override
    public boolean isValid(Writable writable) {
        long value;
        if (writable instanceof IntWritable || writable instanceof LongWritable) {
            value = writable.toLong();
        } else {
            try {
                value = Long.parseLong(writable.toString());
            } catch (NumberFormatException e) {
                return false;
            }
        }
        if (min != null && value < min) return false;
        if (max != null && value > max) return false;

        return true;
    }

    @Override
    public LongMetaData clone() {
        return new LongMetaData(min, max);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("LongMetaData(");
        if (min != null) sb.append("minAllowed=").append(min);
        if (max != null) {
            if (min != null) sb.append(",");
            sb.append("maxAllowed=").append(max);
        }
        sb.append(")");
        return sb.toString();
    }
}
